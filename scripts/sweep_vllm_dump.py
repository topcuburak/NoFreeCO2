#!/usr/bin/env python3
"""Sweep vLLM TP=1 configs, running one suspend/store/load/restore cycle per config.

For each config: launch serve.py (held alive via --repeat), wait for the KV pool to
allocate, run the dump cycle (cuda-checkpoint -> store -> load -> resume) scoped to
vLLM's GPU, kill serve, then tabulate footprint + per-leg latency & cpu_abs.

Use it two ways:
  # BATCH-SIZE sweep (footprint ~constant -> transparency-tax demo):
  sudo -E $(which python) scripts/sweep_vllm_dump.py --store-out /var/data \
    --configs "--input-len 2000 --num-prompts 8|--input-len 8000 --num-prompts 16|--input-len 12000 --num-prompts 32"

  # GPU-MEM-UTIL sweep (footprint VARIES -> cost-vs-S curve):
  sudo -E $(which python) scripts/sweep_vllm_dump.py --store-out /var/data \
    --base "--input-len 8000 --num-prompts 16" \
    --configs "--gpu-memory-utilization 0.5|--gpu-memory-utilization 0.7|--gpu-memory-utilization 0.9"

Run as root (cuda-checkpoint/RAPL). Source ford_env.sh + set HF_TOKEN first -- the
serve subprocess inherits the env (+ the vLLM workarounds set here). Each serve runs
TP=1 single-process + --enforce-eager (validated checkpoint path).
"""
from __future__ import annotations

import argparse
import os
import signal
import statistics
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import transparent_dump as td                     # noqa: E402
import timed_dump_experiment as tde               # noqa: E402
from _common import build_telemetry               # noqa: E402

try:
    import pynvml                                 # noqa: E402
    _HAVE_NVML = True
except Exception:
    _HAVE_NVML = False


def gpu_generating(min_util: int = 15) -> bool:
    """True if any GPU is actively computing (>min_util%) -- i.e. vLLM is generating,
    which only happens AFTER full init (weights + KV pool + warmup)."""
    if not _HAVE_NVML:
        return True
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            if pynvml.nvmlDeviceGetUtilizationRates(h).gpu >= min_util:
                return True
    except Exception:
        return True
    return False


def cpu_abs(rec) -> float:
    if rec is None:
        return 0.0
    for s in rec.sources:
        if s.name == "cpu_pkg_energy_rapl" and s.energy_abs_j is not None:
            return s.energy_abs_j
    return 0.0


def wait_for_ready(min_gb: float, timeout_s: float, stable_s: float = 12.0):
    """Wait until vLLM is GENUINELY ready: GPU memory >= min_gb AND it has PLATEAUED
    (KV pool allocated) AND we've observed the GPU GENERATING (util active). The
    generation gate is the key fix -- the memory plateau alone fires on the profiling
    pause (weights-only), and dumping then captures the wrong footprint and can break
    a half-initialized process. Returns (pids, used_bytes)."""
    deadline = time.monotonic() + timeout_s
    last = -1.0
    stable_since = None
    seen_gen = False
    while time.monotonic() < deadline:
        used = td.gpu_used_bytes()
        pids = tde.gpu_compute_pids()
        if gpu_generating():
            seen_gen = True                            # vLLM has started generating
        if pids and used >= min_gb * 1e9 and seen_gen:
            if abs(used - last) < 1e9:                 # memory stable within 1 GB
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= stable_s:
                    return pids, used                  # fully initialized + generating + plateaued
            else:
                stable_since = None
        last = used
        time.sleep(3)
    return None, 0.0


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)   # live logs under nohup/redirect
    ap = argparse.ArgumentParser(description="sweep vLLM TP=1 configs through a dump cycle")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--base", default="--max-model-len 16384 --max-tokens 256",
                    help="args common to every config")
    ap.add_argument("--configs", required=True, help="'|'-separated serve.py arg strings")
    ap.add_argument("--store-out", default="/var/data", help="NVMe dir for store/load")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--settle", type=float, default=20.0, help="wait after alloc before dumping")
    ap.add_argument("--min-gb", type=float, default=20.0, help="GPU mem that means 'allocated'")
    ap.add_argument("--timeout", type=float, default=300.0, help="max wait for vLLM startup")
    ap.add_argument("--cycles", type=int, default=3, help="suspend/resume cycles per config (mean±std)")
    ap.add_argument("--cycle-gap", type=float, default=5.0, help="serve-resume gap between cycles")
    args = ap.parse_args()

    pf = td.preflight(argparse.Namespace(cc_bin=None, criu_bin=None))
    if pf["problems"]:
        for p in pf["problems"]:
            print(f"[sweep] BLOCKER: {p}")
        raise SystemExit(1)

    serve_py = os.path.join(_REPO, "workloads", "a2_vllm", "serve.py")
    serve_env = dict(os.environ, VLLM_ENABLE_V1_MULTIPROCESSING="0",
                     VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_ATTENTION_BACKEND="FLASH_ATTN")
    configs = [c.strip() for c in args.configs.split("|") if c.strip()]
    rows = []

    for ci, cfg in enumerate(configs):
        cmd = ([sys.executable, serve_py, "--model", args.model,
                "--tensor-parallel-size", "1", "--enforce-eager", "--repeat", "100000"]
               + args.base.split() + cfg.split())
        print(f"\n[sweep] === config {ci}: {cfg} ===")
        print(f"[sweep] launch: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, env=serve_env, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            pids, used = wait_for_ready(args.min_gb, args.timeout)
            if not pids:
                print(f"[sweep] config {ci}: TIMEOUT waiting for GPU pool to allocate -- skipping")
                continue
            print(f"[sweep] ready (PIDs {pids}, GPU {used/1e9:.1f} GB allocated + plateaued); "
                  f"settling {args.settle:.0f}s")
            time.sleep(args.settle)
            pids = tde.gpu_compute_pids() or pids

            tele = build_telemetry(nvml_gpus=tde.pid_gpu_indices(pids))
            tele.start()
            cycles = []
            try:
                for cyc in range(args.cycles):
                    print(f"[sweep] config {ci} cycle {cyc + 1}/{args.cycles}")
                    try:
                        cycles.append(tde.dump_and_resume(
                            tele, pf["cuda_checkpoint"], pf["criu"], pids,
                            out_dir=args.store_out, mark_min=ci, baseline=args.baseline,
                            keep_images=False, skip_criu=True, store=True, store_out=args.store_out))
                    except Exception as e:
                        print(f"[sweep] config {ci} cycle {cyc + 1} FAILED: {type(e).__name__}: {e}")
                    if cyc < args.cycles - 1:
                        time.sleep(args.cycle_gap)
            finally:
                tele.stop()
            if cycles:
                rows.append((cfg, cycles))
        except Exception as e:
            print(f"[sweep] config {ci} FAILED: {type(e).__name__}: {e}")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            for _ in range(30):          # wait for GPU to free before next config
                if td.gpu_used_bytes() < 2e9:
                    break
                time.sleep(2)

    # --- summary (mean ± std over cycles) ---
    def msd(vals):
        if not vals:
            return (0.0, 0.0)
        return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)

    print(f"\n=== vLLM TP=1 DUMP-CYCLE SWEEP ({len(rows)} configs x {args.cycles} cycles) ===  mean ± std")
    print(f"{'config':34}{'foot_GB':>8}{'susp_s':>14}{'store_s':>8}{'res_s':>14}"
          f"{'dump_J':>14}{'restore_J':>15}")
    for cfg, cycles in rows:
        foot, _ = msd([c["suspend"].extra.get("gpu_freed_bytes", 0) / 1e9 for c in cycles])
        sm, ss = msd([c["suspend"].latency_s for c in cycles])
        stm, _ = msd([c["store"].latency_s for c in cycles if c["store"]])
        rm, rs = msd([c["resume"].latency_s for c in cycles])
        dm, ds = msd([cpu_abs(c["suspend"]) + cpu_abs(c["store"]) for c in cycles])
        rjm, rjs = msd([cpu_abs(c["load"]) + cpu_abs(c["resume"]) for c in cycles])
        print(f"{cfg[:34]:34}{foot:8.1f}{sm:8.2f}±{ss:<5.2f}{stm:8.2f}"
              f"{rm:8.2f}±{rs:<5.2f}{dm:8.0f}±{ds:<5.0f}{rjm:9.0f}±{rjs:<5.0f}")
    print("\ndump_J = suspend+store cpu_abs ; restore_J = load+resume cpu_abs. "
          "Batch -> foot_GB ~constant (tax); mem-util -> foot_GB varies (S-curve).")


if __name__ == "__main__":
    main()
