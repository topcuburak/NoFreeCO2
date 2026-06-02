#!/usr/bin/env python3
"""Characterize the HBM <-> host (PCIe) leg BIDIRECTIONALLY -- the GPU<->DRAM mirror
of characterize_storage:

  hbm_to_host (D2H) = dump / suspend EXTRACT leg
  host_to_hbm (H2D) = RESTORE leg

Per direction, reports bandwidth + GPU marginal energy (NVML, the DMA work above
idle) + CPU cpu_abs (RAPL, time term), mean ± std over repeated trials. This is the
RAW PCIe capability (cudaMemcpy); the real cuda-checkpoint extract is ~4x slower
(lock/free/per-allocation overhead) and is measured separately by timed_dump.

    sudo -E $(which python) scripts/characterize_hbm_host.py --bytes 16e9 --gpu 1 \
        --repeat 8 --warmup 2

bytes <= ~32 GB on a 40 GB A100 (allocates nbytes on GPU + nbytes pinned host).
Run as root for RAPL energy. Use a FREE --gpu if serving holds GPU 0.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from harness import measure_operation                              # noqa: E402
from harness.configload import load                                # noqa: E402
from microbench.isolation import pcie_copy, pcie_copy_h2d          # noqa: E402
from _common import build_telemetry, write_record, print_record    # noqa: E402


def gpu_marg_j(rec) -> float:
    for s in rec.sources:
        if s.name == "nvml_gpu_pkg":
            return s.energy_j or 0.0
    return 0.0


def cpu_abs_j(rec) -> float:
    for s in rec.sources:
        if s.name == "cpu_pkg_energy_rapl" and s.energy_abs_j is not None:
            return s.energy_abs_j
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="HBM<->host (PCIe) bidirectional characterization")
    ap.add_argument("--bytes", type=float, default=16e9, help="bytes moved per trial (<=~32e9)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1, help="copies per trial (1 = move S once)")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    nbytes = int(args.bytes)
    cfg = load("ford.yaml")
    dirs = [("hbm_to_host", pcie_copy), ("host_to_hbm", pcie_copy_h2d)]

    tele = build_telemetry(cfg, nvml_gpus=[args.gpu])   # scope GPU power to the one in use
    tele.start()
    rows = []
    try:
        for name, op in dirs:
            print(f"\n[hbm-host] === {name}  {args.repeat} trial(s) (+{args.warmup} warmup) ===")
            recs = []
            for i in range(args.warmup + args.repeat):
                tag = "warmup" if i < args.warmup else f"t{i - args.warmup + 1}"
                rec = measure_operation(
                    tele, workload=f"hbmhost:{name}", operation=name,
                    state_bytes=nbytes * args.iters, baseline_seconds=args.baseline,
                    op=lambda: op(nbytes, gpu=args.gpu, iters=args.iters),
                    config={"dir": name, "gpu": args.gpu, "iters": args.iters})
                print(f"  [{name} {tag}] {rec.latency_s:6.3f}s  gpu {gpu_marg_j(rec):6.0f}J  "
                      f"cpu_abs {cpu_abs_j(rec):6.0f}J")
                if i >= args.warmup:
                    recs.append(rec); write_record(rec, "hbm_host_char")
            rows.append((name, recs))
    finally:
        tele.stop()

    def msd(vals):
        if not vals:
            return 0.0, 0.0
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    moved = nbytes * args.iters
    print(f"\n=== HBM<->HOST CHARACTERIZATION ({nbytes/1e9:.0f} GB x{args.iters}, "
          f"n={args.repeat}) ===   mean ± std")
    print(f"{'direction':13}{'GB/s':>16}{'latency_s':>16}{'gpu_marg_J':>14}{'cpu_abs_J':>14}")
    for name, recs in rows:
        bwm, bws = msd([moved / r.latency_s / 1e9 for r in recs if r.latency_s])
        lm, ls = msd([r.latency_s for r in recs])
        gm, gs = msd([gpu_marg_j(r) for r in recs])
        cm, cs = msd([cpu_abs_j(r) for r in recs])
        print(f"{name:13}{bwm:8.2f}±{bws:<7.2f}{lm:8.3f}±{ls:<7.3f}"
              f"{gm:7.0f}±{gs:<6.0f}{cm:7.0f}±{cs:<6.0f}")
    print("\nRAW PCIe capability (cudaMemcpy). Real cuda-checkpoint extract is ~4x slower"
          " (lock/free/per-alloc overhead) -- see timed_dump. GPU=marginal (DMA above idle),"
          " cpu_abs=time term. DRAM modeled.")


if __name__ == "__main__":
    main()
