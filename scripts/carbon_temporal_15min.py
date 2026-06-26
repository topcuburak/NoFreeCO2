#!/usr/bin/env python3
"""Carbon benefit of temporal scheduling at 15-MINUTE granularity, net of suspend/restore cost.

Same deadline-budget model as carbon_temporal.py, but the scheduler decides every 15 min instead of
hourly. Per the brief, we REINTERPRET each hourly CI sample as one 15-min slot (no interpolation), so
a 4-hour span is now 16 CI values. Keeping the workload COMPUTE hours unchanged, a job that needs C
compute hours now needs 4C slots and a deadline of H hours is 4H slots.

Two schedulers are compared, both relative to "run now" (naive, contiguous from t):
  - NO suspend/resume : may delay the start for free, but must run the 4C slots CONTIGUOUSLY.
                        -> picks the single cleanest contiguous 4C-slot window. Mechanism cost = 0.
  - WITH suspend/resume: may also suspend mid-run to skip dirty slots -> picks the 4C cleanest slots
                        anywhere in the window (may be scattered). Pays a measured dump+restore
                        (split-priced: dump half at the suspend-slot CI, restore half at the
                        resume-slot CI) for every internal gap.

The marginal value of suspend/resume is (savings_with - savings_without). When it is NEGATIVE the
fragmentation gain does not pay for the mechanism, i.e. suspend/restore KILLS the saving and you are
better off with the free contiguous shift. At 15-min granularity even C=1h jobs (4 slots) can fragment,
so they are no longer automatically immune.

Representative regions are auto-selected stratified by CI volatility (CV). For each region we draw N
random start times per workload and average. Both NVMe and SATA tiers reported.

  python scripts/carbon_temporal_15min.py --year 2023 --regions 18 --samples 80
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, _HERE)
from carbon_temporal import WORKLOADS, load_ci, run_blocks

SLOT_H = 0.25  # each CI sample now represents 15 min = 0.25 h


def best_contiguous(win, cslots):
    """Min sum over any contiguous cslots-length window (the free, no-suspend shift)."""
    s = sum(win[:cslots])
    best = s
    for i in range(cslots, len(win)):
        s += win[i] - win[i - cslots]
        if s < best:
            best = s
    return best


def eval15(win, cslots, P, dump_kwh, rest_kwh):
    """win: list of CI for the H-slot window (no None). Returns the two schedules' carbon."""
    e_slot = P * SLOT_H                                  # kWh delivered per 15-min slot
    naive = sum(win[:cslots]) * e_slot                   # run now, contiguous from t

    # NO suspend: best contiguous block (delayed start is free, run is one piece)
    nosusp = best_contiguous(win, cslots) * e_slot

    # WITH suspend: 4C cleanest slots anywhere, pay mechanism per internal gap
    order = sorted(range(len(win)), key=lambda i: win[i])
    sel = sorted(order[:cslots])
    compute = sum(win[i] for i in sel) * e_slot
    blocks = run_blocks(sel)
    K = len(blocks) - 1
    mech = 0.0
    for b in range(K):
        ci_dump = win[blocks[b][1]]                      # suspend+store priced here
        ci_rest = win[blocks[b + 1][0]]                  # load+resume priced here
        mech += dump_kwh * ci_dump + rest_kwh * ci_rest
    susp = compute + mech

    return dict(naive=naive, nosusp=nosusp, susp=susp, compute=compute, mech=mech, K=K)


def valid_starts(ci, hslots):
    """t such that ci[t:t+hslots] has no None, via a None-count prefix."""
    n = len(ci)
    pre = [0] * (n + 1)
    for i, v in enumerate(ci):
        pre[i + 1] = pre[i] + (1 if v is None else 0)
    return [t for t in range(0, n - hslots + 1) if pre[t + hslots] - pre[t] == 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--regions", type=int, default=18, help="how many representative regions")
    ap.add_argument("--samples", type=int, default=80, help="random start times per region per workload")
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    root = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "carbon_data")
    traces = {}
    for c in sorted(os.listdir(root)):
        hits = glob.glob(os.path.join(root, c, f"**/*_{args.year}_hourly.csv"), recursive=True)
        if hits:
            ci = load_ci(hits[0])
            if len(ci) >= 8000:
                traces[c] = ci

    # volatility (CV%) per region, then stratified pick across the CV range
    cv = {c: 100 * st.pstdev([v for v in ci if v]) / st.mean([v for v in ci if v]) for c, ci in traces.items()}
    by_cv = sorted(traces, key=lambda c: cv[c])
    k = min(args.regions, len(by_cv))
    idx = [round(i * (len(by_cv) - 1) / (k - 1)) for i in range(k)]
    regions = [by_cv[i] for i in sorted(set(idx))]
    print(f"[15min] {len(traces)} traces; {len(regions)} representative regions (stratified by CV), "
          f"{args.samples} random starts each, slot=15min")
    print("  regions (CV%): " + ", ".join(f"{c}={cv[c]:.0f}" for c in regions))

    # horizons in HOURS: a tight (2xC) and a loose (4xC) slack, capped at 48 h
    rows = []
    for wl, w in WORKLOADS.items():
        C = w["C"]
        cslots = int(round(C / SLOT_H))
        for slack, Hh in (("tight", min(48, 2 * C)), ("loose", min(48, 4 * C))):
            hslots = int(round(Hh / SLOT_H))
            if hslots <= cslots:
                continue
            for tier in ("nvme", "sata"):
                dump_kwh = w[f"dump_{tier}"] / 3600.0
                rest_kwh = w[f"rest_{tier}"] / 3600.0
                sv_no, sv_su, dlt, ovh, Ks, kill = [], [], [], [], [], []
                for c in regions:
                    ci = traces[c]
                    starts = valid_starts(ci, hslots)
                    if not starts:
                        continue
                    picks = starts if len(starts) <= args.samples else rng.sample(starts, args.samples)
                    for t in picks:
                        win = ci[t:t + hslots]
                        r = eval15(win, cslots, w["P"], dump_kwh, rest_kwh)
                        if r["naive"] <= 0:
                            continue
                        s_no = 100 * (r["naive"] - r["nosusp"]) / r["naive"]
                        s_su = 100 * (r["naive"] - r["susp"]) / r["naive"]
                        gross = r["naive"] - r["compute"]
                        sv_no.append(s_no)
                        sv_su.append(s_su)
                        dlt.append(s_su - s_no)
                        ovh.append(100 * r["mech"] / gross if gross > 0 else 0.0)
                        Ks.append(r["K"])
                        kill.append(1 if (s_su - s_no) < -1e-9 else 0)
                if not sv_no:
                    continue
                rows.append(dict(wl=wl, name=w["name"], tier=tier, C=C, slack=slack, H=Hh,
                                 nosusp=st.mean(sv_no), susp=st.mean(sv_su), delta=st.mean(dlt),
                                 ovh=st.mean(ovh), K=st.mean(Ks), kill=100 * st.mean(kill), n=len(sv_no)))

    # print: one row per wl/slack/tier
    print(f"\nCOMPUTE C unchanged; slots = 4*C (15-min). save% vs run-now. delta = value of suspend.")
    print(f"{'WL':3} {'C':>2}h {'slack':5} {'H':>3}h {'tier':4} | {'no-susp%':>8} {'susp-net%':>9} "
          f"{'Δsusp%':>7} {'mech%ovh':>8} {'K':>5} {'kill%':>6}")
    for wl in ["A4", "A5", "A7", "A8", "A2", "A1", "A6", "A3"]:
        for slack in ("tight", "loose"):
            for tier in ("nvme", "sata"):
                r = next((x for x in rows if x["wl"] == wl and x["slack"] == slack and x["tier"] == tier), None)
                if not r:
                    continue
                print(f"{wl:3} {r['C']:>2}  {slack:5} {r['H']:>3}  {tier:4} | "
                      f"{r['nosusp']:8.1f} {r['susp']:9.1f} {r['delta']:7.2f} {r['ovh']:8.1f} "
                      f"{r['K']:5.2f} {r['kill']:6.1f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"\n[15min] -> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
