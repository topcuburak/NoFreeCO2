#!/usr/bin/env python3
"""Kill zone: where suspend/restore overhead EXCEEDS the carbon it saves (net < 0).

Reuses the carbon_temporal model (oracle, split mechanism pricing). For each workload x tier x
horizon, over all ~50 Electricity Maps grids x start hours, reports the fraction of instances where
net < 0 -- i.e. the dump/restore overhead is larger than the gross carbon saving, so suspending LOSES.
Also splits flat (CV<15%) vs volatile (CV>=25%) grids. Short jobs (C=1) are immune (K=0, never suspend).

  python scripts/carbon_killzone.py --year 2023 --step 12
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from carbon_temporal import WORKLOADS, eval_instance, load_ci

HORIZONS = [4, 6, 8, 12, 16, 24, 36, 48]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--step", type=int, default=12)
    args = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "carbon_data")

    traces = {}
    for c in sorted(os.listdir(root)):
        h = glob.glob(os.path.join(root, c, f"**/*_{args.year}_hourly.csv"), recursive=True)
        if h:
            ci = load_ci(h[0])
            if len(ci) >= 8000:
                traces[c] = ci
    cv = {c: 100 * st.pstdev([v for v in ci if v]) / st.mean([v for v in ci if v]) for c, ci in traces.items()}
    flat = [c for c in traces if cv[c] < 15]

    def kill(wl, tier, H, grids=None):
        w = WORKLOADS[wl]; dk = w[f"dump_{tier}"] / 3600; rk = w[f"rest_{tier}"] / 3600
        neg = tot = 0
        for c in (grids or traces):
            ci = traces[c]
            for t in range(0, len(ci) - H, args.step):
                r = eval_instance(ci, t, H, w["C"], w["P"], dk, rk)
                if r is None or r["naive"] <= 0:
                    continue
                tot += 1; neg += 1 if r["net"] < -1e-9 else 0
        return (100 * neg / tot) if tot else None

    print(f"{len(traces)} grids ({len(flat)} flat CV<15%); %% of instances with net<0 (suspend/restore LOSES)")
    print(f"{'WL':3}{'C':>3}{'P_W':>6}{'tier':>6} | " + " ".join(f"H{h:>2}" for h in HORIZONS) + "  | flat worst")
    for wl in ["A4", "A5", "A7", "A8", "A2", "A1", "A6", "A3"]:
        w = WORKLOADS[wl]
        for tier in ("nvme", "sata"):
            cells = []
            for H in HORIZONS:
                if H <= w["C"]:
                    cells.append("  -"); continue
                p = kill(wl, tier, H)
                cells.append(f"{p:3.0f}%" if p is not None else "  -")
            Hf = w["C"] + 4
            fp = kill(wl, tier, Hf, flat) if Hf <= 48 else None
            print(f"{wl:3}{w['C']:>3}{int(w['P']*1000):>6}{tier:>6} | " + " ".join(f"{c:>4}" for c in cells)
                  + f"  | {('%3.0f%%(H%d)' % (fp, Hf)) if fp is not None else '  -'}")


if __name__ == "__main__":
    main()
