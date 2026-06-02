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


def cpu_abs(rec) -> float:
    if rec is None:
        return 0.0
    for s in rec.sources:
        if s.name == "cpu_pkg_energy_rapl" and s.energy_abs_j is not None:
            return s.energy_abs_j
    return 0.0


def wait_for_ready(min_gb: float, timeout_s: float, stable_s: float = 12.0):
    """Wait until a process holds >= min_gb AND GPU memory has PLATEAUED (KV pool
    fully allocated, not just weights). Dumping before the plateau captures only the
    weights, so the footprint wouldn't reflect the pool / gpu-mem-util. Returns
    (pids, used_bytes)."""
    deadline = time.monotonic() + timeout_s
    last = -1.0
    stable_since = None
    while time.monotonic() < deadline:
        used = td.gpu_used_bytes()
        pids = tde.gpu_compute_pids()
        if pids and used >= min_gb * 1e9:
            if abs(used - last) < 1e9:                 # within 1 GB of the previous sample
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= stable_s:
                    return pids, used                  # plateaued -> pool allocated
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
            try:
                recs = tde.dump_and_resume(
                    tele, pf["cuda_checkpoint"], pf["criu"], pids,
                    out_dir=args.store_out, mark_min=ci, baseline=args.baseline,
                    keep_images=False, skip_criu=True, store=True, store_out=args.store_out)
            finally:
                tele.stop()
            rows.append((cfg, recs))
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

    # --- summary ---
    print(f"\n=== vLLM TP=1 DUMP-CYCLE SWEEP ({len(rows)} configs) ===")
    print(f"{'config':42}{'foot_GB':>8}{'susp_s':>8}{'store_s':>8}{'load_s':>8}"
          f"{'res_s':>7}{'dump_J':>9}{'restore_J':>11}")
    for cfg, r in rows:
        foot = r["suspend"].extra.get("gpu_freed_bytes", 0) / 1e9
        ss, st = r["suspend"].latency_s, (r["store"].latency_s if r["store"] else 0)
        ld, rs = (r["load"].latency_s if r["load"] else 0), r["resume"].latency_s
        dump_j = cpu_abs(r["suspend"]) + cpu_abs(r["store"])
        rest_j = cpu_abs(r["load"]) + cpu_abs(r["resume"])
        print(f"{cfg[:42]:42}{foot:8.1f}{ss:8.2f}{st:8.2f}{ld:8.2f}{rs:7.2f}"
              f"{dump_j:9.0f}{rest_j:11.0f}")
    print("\ndump_J = suspend+store cpu_abs ; restore_J = load+resume cpu_abs. "
          "Batch sweep -> foot_GB ~constant (tax); mem-util sweep -> foot_GB varies (S-curve).")


if __name__ == "__main__":
    main()
