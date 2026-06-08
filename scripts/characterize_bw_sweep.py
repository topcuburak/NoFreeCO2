#!/usr/bin/env python3
"""Sweep the HBM<->host PCIe transfer at CONTROLLED bandwidths and DECOMPOSE the dump
energy into its two physical terms:

    E(S, BW) = e_byte * S   +   P_hold * (S / BW)
               \___DMA work__/   \___time term___/
               BW-independent     grows as BW drops

Holding S fixed and sweeping BW sweeps the transfer time t = S/BW. Fitting **E vs t**
(least squares over the swept points) gives:
    slope     = P_hold   [W]      -- the effective holding power (the time term)
    intercept = e_byte*S [J]      -- the irreducible per-byte data-movement energy

That model predicts the dump/restore cost at ANY effective bandwidth -- the
cuda-checkpoint ~4.6 GB/s, the raw ~25 GB/s ceiling, or a contended per-rank TP=4 rate
-- and says which term dominates where (the paper's "time term dominates" claim, now
fit from data instead of asserted).

    sudo -E $(which python) scripts/characterize_bw_sweep.py --bytes 16e9 --gpu 0 \
        --rates 2,4,6,8,12,16,0 --dir both --repeat 3 --warmup 1

target rate 0 = unthrottled (hardware ceiling). Run as root for RAPL. Use a FREE --gpu.
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
from microbench.isolation import pcie_copy_rate                    # noqa: E402
from _common import build_telemetry, write_record                  # noqa: E402


def _abs_j(rec, name: str) -> float:
    for s in rec.sources:
        if s.name == name and getattr(s, "energy_abs_j", None) is not None:
            return s.energy_abs_j
    return 0.0


def gpu_abs(rec) -> float:
    return _abs_j(rec, "nvml_gpu_pkg")


def cpu_abs(rec) -> float:
    return _abs_j(rec, "cpu_pkg_energy_rapl")


def lstsq(xs, ys):
    """Fit y = a + b*x. Returns (intercept a, slope b)."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0, 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    return (a, b)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="HBM<->host PCIe bandwidth sweep + energy decomposition")
    ap.add_argument("--bytes", type=float, default=16e9, help="bytes moved per trial (S)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--rates", default="2,4,6,8,12,16,0",
                    help="comma target GB/s; 0 = unthrottled ceiling")
    ap.add_argument("--dir", choices=["d2h", "h2d", "both"], default="both",
                    help="d2h=suspend/extract, h2d=restore")
    ap.add_argument("--chunk-mb", type=int, default=128)
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    S = int(args.bytes)
    S_gb = S / 1e9
    rates = [float(r) for r in args.rates.split(",") if r.strip() != ""]
    dirs = ["d2h", "h2d"] if args.dir == "both" else [args.dir]

    tele = build_telemetry(nvml_gpus=[args.gpu])
    tele.start()
    results = {}
    try:
        for d in dirs:
            leg = "suspend/extract (HBM->host)" if d == "d2h" else "restore (host->HBM)"
            print(f"\n[bw] ===== {d}  {leg}  ({S_gb:.0f} GB/trial) =====")
            rows = []
            for r in rates:
                tag = "ceiling" if r == 0 else f"{r:g}GB/s"
                lat, gj, cj = [], [], []
                for i in range(args.warmup + args.repeat):
                    rec = measure_operation(
                        tele, workload=f"bwsweep:{d}", operation=f"pcie_{d}",
                        state_bytes=S, baseline_seconds=args.baseline,
                        op=lambda r=r, d=d: pcie_copy_rate(
                            S, gpu=args.gpu, direction=d,
                            target_gbps=(r or None), chunk_mb=args.chunk_mb),
                        config={"dir": d, "target_gbps": r, "gpu": args.gpu})
                    if i >= args.warmup:
                        lat.append(rec.latency_s); gj.append(gpu_abs(rec)); cj.append(cpu_abs(rec))
                        write_record(rec, "bw_sweep")
                achieved = S_gb / statistics.mean(lat) if lat else 0.0
                row = dict(target=r, tag=tag, bw=achieved,
                           lat=statistics.mean(lat),
                           gpu=statistics.mean(gj), cpu=statistics.mean(cj),
                           tot=statistics.mean(gj) + statistics.mean(cj))
                rows.append(row)
                print(f"  [{tag:>8}] achieved {achieved:5.1f} GB/s  lat {row['lat']:6.2f}s  "
                      f"gpu_abs {row['gpu']:6.0f}J  cpu_abs {row['cpu']:6.0f}J  "
                      f"total {row['tot']:6.0f}J")
            results[d] = rows
    finally:
        tele.stop()

    # --- fit E = intercept + P_hold * latency, for gpu / cpu / total ---
    print(f"\n=== DECOMPOSITION  E = e_byte*S + P_hold*t   (fit over {len(rates)} BW points, S={S_gb:.0f} GB) ===")
    print(f"{'dir':5}{'domain':8}{'P_hold(W)':>12}{'intercept(J)':>14}{'e_byte(J/GB)':>14}")
    for d, rows in results.items():
        xs = [r["lat"] for r in rows]
        for dom, key in (("gpu", "gpu"), ("cpu", "cpu"), ("total", "tot")):
            a, b = lstsq(xs, [r[key] for r in rows])
            print(f"{d:5}{dom:8}{b:12.1f}{a:14.0f}{a / S_gb:14.1f}")
    print("\nP_hold (slope) = watts paid per second the transfer takes = the TIME TERM.")
    print("intercept = e_byte*S = BW-independent DMA energy. e_byte = intercept/S (J/GB).")
    print("Predict dump energy at any BW:  E(S,BW) = e_byte*S + P_hold*(S/BW).")
    print("Raw per-trial records -> data/bw_sweep.jsonl")


if __name__ == "__main__":
    main()
