#!/usr/bin/env python3
"""Sweep the TEMPORAL suspend-to-disk mechanism cost vs FOOTPRINT, using hold_gpu as a
clean single-process target (no vLLM, no NCCL). For each footprint S: allocate S GiB on
one GPU, then run the full temporal cycle and measure per-leg latency + energy:

    suspend (HBM->host) -> store (host->disk) -> load (disk->host) -> resume (host->HBM)

The temporal mechanism cost depends only on the footprint S (bytes moved), not on the
workload's semantics, so this single sweep gives the temporal cost for ANY target
workload by its state size (A2 vLLM, A3 ViT, A4 DLRM, A5 HACC). Workloads >40 GB (A1
FSDP, large A5) exceed one A100 and are extrapolated via the linear fit; transparent
TP>1 suspend is separately infeasible (NCCL/IPC).

    sudo -E $(which python) scripts/sweep_checkpoint_size.py --gpu 0 \
        --sizes 1,2,4,8,16,24,30,36 --store-out /var/data --cycles 3 --hold-seconds 0

criu is skipped (io_uring); the store/load legs are O_DIRECT device-rate proxies. Run as
root (cuda-checkpoint + RAPL). Source ford_env.sh first. Use a FREE --gpu.
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

import transparent_dump as td                       # noqa: E402
import timed_dump_experiment as tde                 # noqa: E402
from _common import build_telemetry                 # noqa: E402


def cpu_abs(rec) -> float:
    if rec is None:
        return 0.0
    for s in rec.sources:
        if s.name == "cpu_pkg_energy_rapl" and getattr(s, "energy_abs_j", None) is not None:
            return s.energy_abs_j
    return 0.0


def gpu_abs(rec) -> float:
    if rec is None:
        return 0.0
    for s in rec.sources:
        if s.name == "nvml_gpu_pkg" and getattr(s, "energy_abs_j", None) is not None:
            return s.energy_abs_j
    return 0.0


def wait_for_alloc(min_bytes: float, timeout_s: float = 120.0):
    """Wait until hold_gpu has allocated its HBM; return the GPU PIDs."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = td.gpu_used_bytes()
        pids = tde.gpu_compute_pids()
        if pids and used >= min_bytes:
            return pids, used
        time.sleep(1)
    return None, 0.0


def lstsq(xs, ys):
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0, 0.0)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return (my - b * mx, b)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="temporal suspend-to-disk cost vs footprint (hold_gpu)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sizes", default="1,2,4,8,16,24,30,36", help="comma GiB footprints (<=~38 on 40GB A100)")
    ap.add_argument("--store-out", default="/var/data", help="tier dir for store/load")
    ap.add_argument("--cycles", type=int, default=3, help="cycles per footprint (mean±std)")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--hold-seconds", type=float, default=0.0, help="parked-state hold per cycle")
    ap.add_argument("--cycle-gap", type=float, default=3.0)
    ap.add_argument("--chunks", type=int, default=1,
                    help="hold_gpu allocation count (probe per-allocation checkpoint "
                         "overhead; raise to mimic a real workload's many tensors)")
    ap.add_argument("--chunks-list", default=None,
                    help="comma allocation counts to sweep at a FIXED footprint (=first --sizes), "
                         "e.g. 1,2,4,16,64,256,1024 -- isolates cost-vs-allocation-structure")
    ap.add_argument("--tag", default=None, help="tag written into each record (e.g. s1_chunks_nvme)")
    args = ap.parse_args()

    pf = td.preflight(argparse.Namespace(cc_bin=None, criu_bin=None))
    if pf["problems"]:
        for p in pf["problems"]:
            print(f"[ckpt-sweep] BLOCKER: {p}")
        raise SystemExit(1)

    hold_py = os.path.join(_REPO, "scripts", "hold_gpu.py")
    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    # configs = list of (gb, chunks, mark). chunks-list -> sweep allocation count at fixed footprint
    # (mark = chunk count); else -> sweep footprint at fixed --chunks (mark = GiB).
    if args.chunks_list:
        cc = [int(c) for c in args.chunks_list.split(",") if c.strip()]
        fixed = sizes[0]
        configs = [(fixed, c, c) for c in cc]
        print(f"[ckpt-sweep] CHUNKS sweep at fixed {fixed} GiB: {cc} allocations")
    else:
        configs = [(gb, args.chunks, int(gb)) for gb in sizes]
    rows = []

    for gb, chunks, mark in configs:
        label = f"{gb} GiB / {chunks} alloc"
        print(f"\n[ckpt-sweep] ===== {label} =====")
        proc = subprocess.Popen(
            [sys.executable, hold_py, "--gb", str(gb), "--gpu", str(args.gpu),
             "--chunks", str(chunks), "--seconds", "100000"],
            start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            pids, used = wait_for_alloc(gb * (1024 ** 3) * 0.9)
            if not pids:
                print(f"[ckpt-sweep] {label}: alloc TIMEOUT -- skipping")
                continue
            print(f"[ckpt-sweep] allocated (PIDs {pids}, {used/1e9:.1f} GB); settling 3s")
            time.sleep(3)
            tele = build_telemetry(nvml_gpus=tde.pid_gpu_indices(pids) or [args.gpu])
            tele.start()
            cyc = []
            try:
                for c in range(args.cycles):
                    print(f"[ckpt-sweep] {label} cycle {c+1}/{args.cycles}")
                    try:
                        cyc.append(tde.dump_and_resume(
                            tele, pf["cuda_checkpoint"], pf["criu"], pids,
                            out_dir=args.store_out, mark_min=mark, baseline=args.baseline,
                            keep_images=False, skip_criu=True, store=True, store_out=args.store_out,
                            hold_seconds=args.hold_seconds, tag=args.tag))
                    except Exception as e:
                        print(f"[ckpt-sweep] {label} cycle {c+1} FAILED: {type(e).__name__}: {e}")
                    if c < args.cycles - 1:
                        time.sleep(args.cycle_gap)
            finally:
                tele.stop()
            if cyc:
                rows.append((gb, chunks, cyc))
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=20)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            for _ in range(20):
                if td.gpu_used_bytes() < 1e9:
                    break
                time.sleep(1)

    # --- per-footprint table + affine fit of the full temporal round-trip ---
    def msd(v):
        return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) if v else (0.0, 0.0)

    def leg_lat(cyc, key):
        return [c[key].latency_s for c in cyc if c.get(key)]

    def leg_e(cyc, key):
        return [cpu_abs(c[key]) + gpu_abs(c[key]) for c in cyc if c.get(key)]

    chunks_mode = bool(args.chunks_list)
    print(f"\n=== TEMPORAL SUSPEND-TO-DISK COST vs "
          f"{'ALLOCATION COUNT' if chunks_mode else 'FOOTPRINT'} ({args.store_out}) ===  mean±std")
    print(f"{'GiB':>5}{'chunks':>7}{'foot_GB':>9}{'susp_s':>9}{'store_s':>9}{'load_s':>9}{'res_s':>9}"
          f"{'RT_s':>9}{'RT_J':>10}")
    pts = []
    for gb, chunks, cyc in rows:
        foot, _ = msd([c["suspend"].extra.get("gpu_freed_bytes", 0) / 1e9 for c in cyc])
        sl, _ = msd(leg_lat(cyc, "suspend"))
        stl, _ = msd(leg_lat(cyc, "store"))
        ldl, _ = msd(leg_lat(cyc, "load"))
        rl, _ = msd(leg_lat(cyc, "resume"))
        rt_l = sl + stl + ldl + rl
        rt_e_vals = [sum(cpu_abs(c[k]) + gpu_abs(c[k]) for k in ("suspend", "store", "load", "resume") if c.get(k))
                     for c in cyc]
        rt_e, rt_es = msd(rt_e_vals)
        print(f"{gb:5.0f}{chunks:7d}{foot:9.1f}{sl:9.2f}{stl:9.2f}{ldl:9.2f}{rl:9.2f}{rt_l:9.2f}{rt_e:10.0f}")
        # x-axis is chunk count (allocation-structure probe) or footprint
        pts.append((chunks if chunks_mode else foot, sl, rl, rt_l, rt_e))

    if len(pts) >= 2:
        X = [p[0] for p in pts]
        xname = "alloc" if chunks_mode else "GB"
        aS, bS = lstsq(X, [p[1] for p in pts])      # suspend latency vs x (per-allocation overhead)
        aR, bR = lstsq(X, [p[2] for p in pts])      # resume latency vs x
        aL, bL = lstsq(X, [p[3] for p in pts])
        aE, bE = lstsq(X, [p[4] for p in pts])
        print(f"\nAffine fit vs {xname}:")
        if chunks_mode:
            print(f"  suspend:  {aS:.2f} s + {bS*1000:.2f} ms/alloc   (per-allocation cuda-checkpoint overhead)")
            print(f"  resume:   {aR:.2f} s + {bR*1000:.2f} ms/alloc")
        print(f"  round-trip latency:  {aL:.2f} s + {bL:.4f} s/{xname}")
        print(f"  round-trip energy:   {aE:.0f} J + {bE:.2f} J/{xname}   (host+GPU, {args.store_out})")
    print("Raw per-op records -> data/timed_dump.jsonl")


if __name__ == "__main__":
    main()
