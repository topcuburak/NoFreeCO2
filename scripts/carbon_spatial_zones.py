#!/usr/bin/env python3
"""Zone-limited spatial migration with OUTLIER TRIMMING + region/subregion granularity.

carbon_spatial.py: migrate overhead ~negligible (kill ~0%) with all 45 zones, because you can always
reach an ultra-clean grid (sweden ~3) so the ~92% prize swamps e_net*S. Real migration is REACHABILITY-
limited (residency / provider footprint), AND in practice you cannot count on the single greenest grid.
So we (1) TRIM the too-green and too-dirty OUTLIER grids globally, then (2) re-run the region-wise test
at REGION and finer SUBREGION granularity. e_net*S is unchanged by zoning, so as the in-zone CI spread
shrinks the migrate overhead (fixed pp) eventually exceeds the prize -> the spatial kill zone.

Infinite capacity assumed (reachability, not capacity). Same pricing as carbon_spatial.py (one-way,
stage-and-forward, source-charged egress). Greedy chaser only (DP optimal ~= greedy, dropped for speed).

  python scripts/carbon_spatial_zones.py --trim 5 --samples 40
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from carbon_spatial import WORKLOADS, E_NET_KJ_PER_GB
from carbon_temporal import load_ci

# Coarse regions (continents) and fine subregions. Names match carbon_data/ folders.
REGIONS = {
    "EUROPE":       ["austria","belgium","bulgaria","crotia","cyprus","czechia","denmark","estonia",
                     "finland","france","germany","great_britain","greece","hungary","ireland","italy",
                     "latvia","lithuania","netherlands","norway","poland","portugal","romania","serbia",
                     "slovakia","slovenia","spain","sweden","switzerland","turkey"],
    "N_AMERICA":    ["USA","canada"],
    "LATAM":        ["brazil","chile","nicaragua","peru"],
    "ASIA_PACIFIC": ["australia","india","japan","new_zeland","philippines","singapore","south_korea","taiwan"],
}
SUBREGIONS = {
    "NORDIC":     ["sweden","norway","finland","denmark"],
    "BALTIC":     ["estonia","latvia","lithuania"],
    "WEST_EU":    ["france","germany","netherlands","belgium","ireland","great_britain","switzerland","austria"],
    "SOUTH_EU":   ["spain","portugal","italy","greece","cyprus","crotia","slovenia"],
    "EAST_EU":    ["poland","czechia","slovakia","hungary","romania","bulgaria","serbia"],
    "EAST_ASIA":  ["japan","south_korea","taiwan"],
    "SE_ASIA":    ["singapore","philippines","indonesia"],
    "OCEANIA":    ["australia","new_zeland"],
    "N_AMERICA":  ["USA","canada"],
    "LATAM":      ["brazil","chile","nicaragua","peru"],
    "MIDEAST":    ["israel","turkey","cyprus"],
}


def eval_spatial(hourly, home, regions, P, egress, rest_kwh):
    """One instance: stay-home vs ideal-free vs greedy-chase carbon over C hours (greedy only)."""
    C = len(hourly)
    base = sum(hourly[h][home] for h in range(C)) * P
    ideal = sum(min(hourly[h][r] for r in regions) for h in range(C)) * P
    comp = 0.0; mech = 0.0; mg = 0; prev = home
    for h in range(C):
        r = min(regions, key=lambda x: hourly[h][x])
        comp += hourly[h][r] * P
        if r != prev:
            mech += egress * hourly[h][prev] + rest_kwh * hourly[h][r]
            mg += 1
        prev = r
    return base, ideal, comp + mech, mg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--trim", type=int, default=5, help="drop this many cleanest AND dirtiest grids (by mean CI)")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260626)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    root = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "carbon_data")
    traces = {}
    for c in sorted(os.listdir(root)):
        hits = glob.glob(os.path.join(root, c, f"**/*_{args.year}_hourly.csv"), recursive=True)
        if not hits:
            continue
        ci = load_ci(hits[0])
        if len(ci) < 8000:
            continue
        nz = [v for v in ci if v]
        if nz and 100 * st.pstdev(nz) / st.mean(nz) >= 1.0:
            traces[c] = ci
    n_hours = min(len(traces[c]) for c in traces)
    meanci = {c: st.mean([v for v in traces[c] if v]) for c in traces}

    # ---- TRIM outliers: drop the N greenest and N dirtiest grids by mean CI ----
    order = sorted(traces, key=lambda c: meanci[c])
    greenest = order[:args.trim]
    dirtiest = order[-args.trim:] if args.trim else []
    keep = set(order) - set(greenest) - set(dirtiest)
    print(f"loaded {len(traces)} grids; trim={args.trim} each tail -> {len(keep)} kept")
    print(f"  dropped GREENEST: " + ", ".join(f"{c}({meanci[c]:.0f})" for c in greenest))
    print(f"  dropped DIRTIEST: " + ", ".join(f"{c}({meanci[c]:.0f})" for c in dirtiest))

    def run(members, wl, tier, samples):
        w = WORKLOADS[wl]; C, P, S = w["C"], w["P"], w["S"]
        regions = [r for r in members if r in traces and r in keep]
        if len(regions) < 2:
            return None
        egress = w[f"dump_{tier}"] / 3600.0 + E_NET_KJ_PER_GB * S / 3600.0
        rest_kwh = w[f"rest_{tier}"] / 3600.0
        I, Gd, OV, KL, MG = [], [], [], [], []
        for home in regions:
            cand = [t for t in range(0, n_hours - C)
                    if all(traces[r][t + h] is not None for r in regions for h in range(C))]
            picks = cand if len(cand) <= samples else rng.sample(cand, samples)
            for t in picks:
                hourly = [{r: traces[r][t + h] for r in regions} for h in range(C)]
                base, ideal, greedy, mg = eval_spatial(hourly, home, regions, P, egress, rest_kwh)
                if base <= 0:
                    continue
                I.append(100 * (base - ideal) / base)
                Gd.append(100 * (base - greedy) / base)
                OV.append(100 * (greedy - ideal) / base)        # positive = overhead in pp
                KL.append(1 if greedy > base + 1e-12 else 0)
                MG.append(mg)
        if not I:
            return None
        return dict(nreg=len(regions), ideal=st.mean(I), greedy=st.mean(Gd),
                    ovh=st.mean(OV), kill=100 * st.mean(KL), Mg=st.mean(MG))

    # ---- Summary across all regions + subregions, sorted by shrinking prize ----
    # A7 = short + large-state (the hard case); A3 = long + small-state (the easy case)
    print(f"\n=== PRIZE + KILL by zone (trimmed), sorted by prize. A7=1h/100GB hard, A3=12h/35GB easy ===")
    print(f"{'zone':12}{'gran':>5}{'#reg':>5}{'CI range':>11}{'idl%avg':>8} | "
          f"{'A7 net%':>8}{'A7 ovh':>7}{'A7 kill%':>9} | {'A3 net%':>8}{'A3 kill%':>9}")
    cards = [("region", k, v) for k, v in REGIONS.items()] + [("sub", k, v) for k, v in SUBREGIONS.items()]
    summ = []
    for gran, name, members in cards:
        regs = [r for r in members if r in traces and r in keep]
        if len(regs) < 2:
            continue
        ideals = [run(members, wl, "nvme", max(12, args.samples // 2)) for wl in WORKLOADS]
        ideals = [x["ideal"] for x in ideals if x]
        a7 = run(members, wl="A7", tier="sata", samples=args.samples)
        a3 = run(members, wl="A3", tier="sata", samples=args.samples)
        cis = sorted(meanci[r] for r in regs)
        summ.append((st.mean(ideals), gran, name, len(regs), cis[0], cis[-1], a7, a3))
    for idl, gran, name, nreg, lo, hi, a7, a3 in sorted(summ):
        print(f"{name:12}{gran:>5}{nreg:>5}{f'{lo:.0f}-{hi:.0f}':>11}{idl:>7.1f}% | "
              f"{a7['greedy']:>8.1f}{a7['ovh']:>7.1f}{a7['kill']:>8.1f}% | {a3['greedy']:>8.1f}{a3['kill']:>8.1f}%")

    # ---- Detailed per-workload tables for the tightest (kill-prone) subregions ----
    for name in ["EAST_ASIA", "EAST_EU", "WEST_EU"]:
        members = SUBREGIONS[name]
        regs = [r for r in members if r in traces and r in keep]
        if len(regs) < 2:
            continue
        cis = sorted(meanci[r] for r in regs)
        print(f"\n=== {name} (trimmed: {', '.join(regs)} | CI {cis[0]:.0f}-{cis[-1]:.0f}) ===")
        print(f"{'WL':3}{'C':>3}{'S_GB':>6}{'tier':>6} | {'B2idl%':>7}{'B3grd%':>7}{'ovh':>6}{'kill%':>7}{'Mg':>6}")
        for wl in ["A4", "A5", "A7", "A8", "A2", "A1", "A6", "A3"]:
            for tier in ("nvme", "sata"):
                r = run(members, wl, tier, args.samples)
                if not r:
                    continue
                print(f"{wl:3}{WORKLOADS[wl]['C']:>3}{WORKLOADS[wl]['S']:>6}{tier:>6} | "
                      f"{r['ideal']:7.1f}{r['greedy']:7.1f}{r['ovh']:6.1f}{r['kill']:7.1f}{r['Mg']:6.2f}")


if __name__ == "__main__":
    main()
