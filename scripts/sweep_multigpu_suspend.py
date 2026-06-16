#!/usr/bin/env python3
"""S1 baseline: TRANSPARENT suspend/restore cost vs footprint, across 1..N GPUs.

Launches one `hold_gpu` process per GPU (each holding `per_gpu` GiB of HBM), then runs
the cuda-checkpoint suspend(HBM->host) + resume(host->HBM) cycle over all of them with
`--multiproc` (lock-all -> checkpoint-all -> restore-all -> unlock-all). hold_gpu has NO
NCCL / CUDA-IPC, so this is the CLEAN multi-GPU GPU-leg baseline -- no destroy/reinit
needed (unlike A1 FSDP / A2 vLLM TP>1). Sweeping (gpu_count x per_gpu) gives the
suspend/restore cost as a function of TOTAL footprint, spanning a few GB to ~150 GB
(4x A100), which single-GPU can't reach (40 GB cap).

    sudo -E $(which python) scripts/sweep_multigpu_suspend.py \
        --gpu-counts 1,2,4 --sizes 8,16,24,32 --cycles 3 --tag s1_mg_nvme

Per config: n_gpus hold_gpu @ per_gpu GiB -> suspend+resume (no disk leg -- HBM<->host
only). Records to data/timed_dump.jsonl (multiproc=true), mark_min = n_gpus*1000+total_GB.
Root (cuda-checkpoint + RAPL). Use FREE GPUs (0..n-1).
"""
from __future__ import annotations

import argparse
import itertools
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


def abs_e(rec, name):
    if rec is None:
        return 0.0
    for s in rec.sources:
        if s.name == name and getattr(s, "energy_abs_j", None) is not None:
            return s.energy_abs_j
    return 0.0


def full_e(rec):
    return (rec.extra.get("full_total_j") if rec is not None else 0.0) or 0.0


def wait_for_n(n_pids, min_total_bytes, timeout_s=120.0):
    """Wait until n_pids GPU processes are up and total HBM >= min_total_bytes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pids = tde.gpu_compute_pids()
        if len(pids) >= n_pids and td.gpu_used_bytes() >= min_total_bytes:
            return pids
        time.sleep(2)
    return tde.gpu_compute_pids()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="multi-GPU transparent suspend/restore cost vs footprint")
    ap.add_argument("--gpu-counts", default="1,2,4", help="comma GPU counts to sweep")
    ap.add_argument("--sizes", default="8,16,24,32", help="comma PER-GPU GiB footprints")
    ap.add_argument("--cycles", type=int, default=4, help="cycles/config (cold cycle-0 dropped in analysis)")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--cycle-gap", type=float, default=3.0)
    ap.add_argument("--tag", default=None, help="record tag, e.g. s1_mg_nvme")
    ap.add_argument("--store-out", default=None,
                    help="if set, run the FULL cycle (suspend->store->load->resume) writing the "
                         "footprint to this dir (tier). Omit -> suspend/resume only (GPU legs). "
                         "drive coeff auto: NVMe 50 W, SATA (path has 'home') 3 W.")
    args = ap.parse_args()
    do_store = args.store_out is not None
    drive_w = 3.0 if (args.store_out and "home" in args.store_out) else 50.0

    pf = td.preflight(argparse.Namespace(cc_bin=None, criu_bin=None))
    if pf["euid"] != 0 or not pf["cuda_checkpoint"]:
        raise SystemExit("[mg-sweep] need root + cuda-checkpoint")

    hold_py = os.path.join(_REPO, "scripts", "hold_gpu.py")
    counts = [int(c) for c in args.gpu_counts.split(",") if c.strip()]
    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    rows = []

    for n, per_gpu in itertools.product(counts, sizes):
        total = n * per_gpu
        label = f"{n}gpu x {per_gpu}GiB = {total:.0f}GiB total"
        print(f"\n[mg-sweep] ===== {label} =====")
        procs = []
        for g in range(n):
            procs.append(subprocess.Popen(
                [sys.executable, hold_py, "--gb", str(per_gpu), "--gpu", str(g),
                 "--chunks", "1", "--seconds", "100000"],
                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        try:
            pids = wait_for_n(n, total * (1024 ** 3) * 0.85)
            if len(pids) < n:
                print(f"[mg-sweep] {label}: only {len(pids)}/{n} allocated -- skipping")
                continue
            print(f"[mg-sweep] up: PIDs {pids}, {td.gpu_used_bytes()/1e9:.1f} GB; settling 3s")
            time.sleep(3)
            tele = build_telemetry(nvml_gpus=list(range(n)))     # scope to the GPUs in use
            tele.start()
            cyc = []
            try:
                for c in range(args.cycles):
                    print(f"[mg-sweep] {label} cycle {c+1}/{args.cycles}")
                    try:
                        cyc.append(tde.dump_and_resume(
                            tele, pf["cuda_checkpoint"], pf["criu"], pids,
                            out_dir=args.store_out or "/tmp", mark_min=n * 1000 + int(round(total)),
                            baseline=args.baseline, keep_images=False, multiproc=True,
                            skip_criu=True, store=do_store, store_out=args.store_out,
                            drive_w=drive_w, tag=args.tag))
                    except Exception as e:
                        print(f"[mg-sweep] {label} cycle {c+1} FAILED: {type(e).__name__}: {e}")
                    if c < args.cycles - 1:
                        time.sleep(args.cycle_gap)
            finally:
                tele.stop()
            if cyc:
                rows.append((n, per_gpu, total, cyc))
        finally:
            for p in procs:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass
            for _ in range(30):
                if td.gpu_used_bytes() < 1e9:
                    break
                time.sleep(1)

    # --- summary ---
    def msd(v):
        return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0) if v else (0.0, 0.0)

    print(f"\n=== MULTI-GPU TRANSPARENT SUSPEND/RESTORE vs FOOTPRINT ===  mean±std")
    print(f"{'nGPU':>5}{'per_GB':>8}{'total_GB':>9}{'susp_s':>9}{'res_s':>9}"
          f"{'susp_GBps':>10}{'susp_kJ':>9}{'res_kJ':>8}{'FULL_kJ':>9}")
    pts = []
    for n, per_gpu, total, cyc in rows:
        foot = msd([c["suspend"].extra.get("gpu_freed_bytes", 0) / 1e9 for c in cyc])[0]
        sl = msd([c["suspend"].latency_s for c in cyc])[0]
        rl = msd([c["resume"].latency_s for c in cyc])[0]
        se = msd([full_e(c["suspend"]) / 1000 for c in cyc])[0]
        re = msd([full_e(c["resume"]) / 1000 for c in cyc])[0]
        bw = foot / sl if sl else 0.0
        print(f"{n:5d}{per_gpu:8.0f}{foot:9.1f}{sl:9.2f}{rl:9.2f}{bw:10.2f}{se:9.2f}{re:8.2f}{se+re:9.2f}")
        pts.append((foot, sl, rl, se + re))

    if len(pts) >= 2:
        X = [p[0] for p in pts]
        aS, bS = _lstsq(X, [p[1] for p in pts])
        aR, bR = _lstsq(X, [p[2] for p in pts])
        aE, bE = _lstsq(X, [p[3] for p in pts])
        print(f"\nAffine fit vs TOTAL footprint S (across all GPU counts):")
        print(f"  suspend:  {aS:.2f} s + {bS:.4f} s/GB   (= {1/bS if bS else 0:.2f} GB/s)")
        print(f"  resume:   {aR:.2f} s + {bR:.4f} s/GB   (= {1/bR if bR else 0:.2f} GB/s)")
        print(f"  energy:   {aE:.2f} kJ + {bE:.4f} kJ/GB  (FULL: meas GPU+CPU + modeled DRAM)")
        print("  -> if the per-GB slope is the SAME across GPU counts, lock-all/checkpoint-all is "
              "sequential (no contention) and S1 is one footprint-driven coefficient.")
    print("Raw records -> data/timed_dump.jsonl (multiproc=true)")


def _lstsq(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    d = n * sxx - sx * sx
    if d == 0:
        return (sy / n, 0.0)
    b = (n * sxy - sx * sy) / d
    a = (sy - b * sx) / n
    return (a, b)


if __name__ == "__main__":
    main()
