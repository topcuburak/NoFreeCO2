#!/usr/bin/env python3
"""Per-job mechanism PROFILE: saveable-carbon overhead AND execution-latency overhead, both legs.

The scheduler does not apply one global policy. For each job it computes TWO per-job overheads from
measured constants and decides smartly, because every job experiences a DIFFERENT overhead:

  CARBON axis   r* = E_mech / E_compute        (break-even relative CI gain per cycle; mechanism_breakeven)
  LATENCY axis  l* = T_mech / runtime          (fraction of runtime added as STALL per cycle)

  - Temporal: E_mech = dump+restore (kJ), T_mech = dump+restore wall time (s), N = K suspends.
  - Spatial : E_mech += e_net*S, T_mech += S / BW_net (the WAN transfer stall), N = M migrations (~1).

Net saveable carbon = f - N*r* (f = predicted relative CI prize). Added completion latency = N*l*.
A smart scheduler shifts a job only where BOTH pay: net carbon > 0 AND added latency fits the deadline
slack. The two axes are NOT correlated across jobs, which is exactly why per-job profiling matters.

  python scripts/mechanism_profile.py --gb-per-s 0.067   # ~15 s/GB single tuned WAN stream
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from carbon_spatial import WORKLOADS, E_NET_KJ_PER_GB

# MEASURED full-cycle (dump+restore) wall time per tier, seconds (temporal_overhead.md / real_workload_validation.md).
TMECH_S = {
    "A1": dict(nvme=71.7,  sata=630.6),
    "A2": dict(nvme=20.2,  sata=173.0),
    "A3": dict(nvme=18.5,  sata=157.3),
    "A4": dict(nvme=18.0,  sata=146.9),
    "A5": dict(nvme=55.5,  sata=311.7),
    "A6": dict(nvme=66.7,  sata=383.6),
    "A7": dict(nvme=88.4,  sata=573.1),
    "A8": dict(nvme=72.0,  sata=495.6),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb-per-s", type=float, default=0.067, help="WAN throughput GB/s (0.067=~15 s/GB single stream)")
    args = ap.parse_args()
    spg = 1.0 / args.gb_per_s
    print(f"per-cycle overhead. CARBON r*=E_mech/E_compute (CI gain needed). LATENCY l*=T_mech/runtime (stall).")
    print(f"WAN throughput {args.gb_per_s} GB/s ({spg:.0f} s/GB).  Multiply by N cycles (temporal K, spatial M~1).\n")
    print(f"{'WL':3}{'C':>3}{'S':>5}{'tier':>5} | {'TEMPORAL':>17} | {'SPATIAL':>22}")
    print(f"{'':3}{'':>3}{'':>5}{'':>5} | {'carbon r*':>9}{'lat l*':>8} | {'carbon r*':>9}{'lat l*':>8}{'  (xfer s)':>11}")
    for wl, w in WORKLOADS.items():
        C, P, S = w["C"], w["P"], w["S"]
        Ecomp = P * C * 3600.0
        run_s = C * 3600.0
        net = E_NET_KJ_PER_GB * S
        xfer = S * spg                                          # WAN transfer stall, s
        for tier in ("nvme", "sata"):
            mech = w[f"dump_{tier}"] + w[f"rest_{tier}"]
            tmech = TMECH_S[wl][tier]
            rt = mech / Ecomp                                  # temporal carbon r*
            lt = tmech / run_s                                 # temporal latency l*
            rs = (mech + net) / Ecomp                          # spatial carbon r*
            ls = (tmech + xfer) / run_s                        # spatial latency l*
            print(f"{wl:3}{C:>3}{S:>5}{tier:>5} | {100*rt:>8.2f}%{100*lt:>7.1f}% | "
                  f"{100*rs:>8.2f}%{100*ls:>7.1f}%{xfer:>10.0f}s")

    print("\nReading (the heterogeneity that forces per-job scheduling):")
    print(" - A3 (long/small): cheap on BOTH axes (r*<1%, l*<2%) -> shift freely, temporal or spatial.")
    print(" - A7/A8 (short/large): expensive on BOTH (spatial r*~40%, spatial l* >50% of runtime) -> shift")
    print("   only with a huge CI prize AND deep deadline slack; otherwise run-now.")
    print(" - A1 (short-ish/high-power/huge state): big absolute carbon prize but spatial l*~20% -> the")
    print("   carbon may pay while latency does not. The two axes disagree -> needs the per-job decision.")


if __name__ == "__main__":
    main()
