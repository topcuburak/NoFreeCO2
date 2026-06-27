#!/usr/bin/env python3
"""Spatial (region-migration) carbon scheduling, net of the WAN migrate-leg cost.

Spatial analogue of carbon_temporal.py. A job needs C contiguous COMPUTE hours and runs continuously
(no suspend-in-place), but each hour it MAY relocate to a cleaner region. The cleanest region itself
moves over the day (t=1 cleanest is A, t=2 is B, ...), so a multi-hour job can chase it A->B->C, paying
a MIGRATE leg per hop. Migration = stage-and-forward: dump at source + ship S GB over WAN + restore at
dest. One-way (results stream back, negligible); no forced return home.

Per hop r' -> r at hour h (split-priced, like the temporal dump/restore):
  migrate_carbon = (dump_kWh + e_net*S) * CI[r'][h]   +   restore_kWh * CI[r][h]
                    \_ egress: dump + network, charged to SOURCE grid _/   \_ ingress at DEST _/
  e_net = 3.6 kJ/GB (full-path marginal, inter-DC; endpoints-only is ~50-100 J, negligible -> ignored).

Four baselines, carbon over the whole C-hour job, vs B1 (stay home):
  B1 home   = sum_h CI[home][t+h] * P                         # reference
  B2 ideal  = sum_h min_r CI[r][t+h] * P                      # cleanest region each hour, FREE migration
  B3 greedy = chase cleanest region each hour, PAY migration  # mechanism-blind; can lose
  B4 dp     = migration-aware min-carbon region path (DP)     # only hops when worth it; >= home always
savings% vs B1.  overhead = B2 - B3 (migration cost in the greedy chase).  kill% = B3 < 0 (chasing
LOSES to staying home).  M = number of migrations.  CI prize across regions is 10-200x (vs ~2-3x
diurnal), but e_net*S is heavy and state-size-linear -> the tradeoff this leg quantifies.

  python scripts/carbon_spatial.py --year 2023 --homes 18 --samples 80
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from carbon_temporal import load_ci

E_NET_KJ_PER_GB = 3.6          # full-path marginal inter-DC (network_coefficients_lit.md)

# C compute h, P kW, S state GB, dump/restore halves per tier in kJ (from the measured campaign).
WORKLOADS = {
    "A1": dict(name="FSDP Llama-8B FT",     C=4,  P=1.471, S=148, dump_nvme=25.66, rest_nvme=10.35, dump_sata=143.0, rest_sata=121.7),
    "A2": dict(name="vLLM batch inference", C=2,  P=0.497, S=41,  dump_nvme=3.70,  rest_nvme=1.60,  dump_sata=19.18, rest_sata=16.50),
    "A3": dict(name="ViT-Huge train",       C=12, P=0.526, S=35,  dump_nvme=3.39,  rest_nvme=1.39,  dump_sata=20.83, rest_sata=17.92),
    "A4": dict(name="DLRM train",           C=1,  P=0.312, S=33,  dump_nvme=3.31,  rest_nvme=1.35,  dump_sata=19.40, rest_sata=16.87),
    "A5": dict(name="GAPBS graph",          C=1,  P=0.255, S=71,  dump_nvme=8.53,  rest_nvme=3.91,  dump_sata=35.21, rest_sata=24.60),
    "A6": dict(name="gem5 sim",             C=8,  P=0.148, S=50,  dump_nvme=9.40,  rest_nvme=5.25,  dump_sata=32.76, rest_sata=30.61),
    "A7": dict(name="DuckDB multi-proc",    C=1,  P=0.311, S=100, dump_nvme=28.14, rest_nvme=7.06,  dump_sata=29.51, rest_sata=44.90),
    "A8": dict(name="DuckDB multi-thread",  C=1,  P=0.302, S=100, dump_nvme=19.72, rest_nvme=10.17, dump_sata=55.19, rest_sata=41.50),
}


def cleanest_path_carbon(traces_at, regions, P):
    """Greedy: each hour pick the globally cleanest region. Returns (path, compute_carbon_no_mech)."""
    path = []
    comp = 0.0
    for hourvals in traces_at:                      # hourvals: dict region->CI for this hour
        r = min(regions, key=lambda x: hourvals[x])
        path.append(r)
        comp += hourvals[r] * P
    return path, comp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--homes", type=int, default=18, help="representative home regions (stratified by CV)")
    ap.add_argument("--samples", type=int, default=80, help="random start hours per home per workload")
    ap.add_argument("--seed", type=int, default=20260626)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    root = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "carbon_data")
    traces, dropped = {}, []
    for c in sorted(os.listdir(root)):
        hits = glob.glob(os.path.join(root, c, f"**/*_{args.year}_hourly.csv"), recursive=True)
        if not hits:
            continue
        ci = load_ci(hits[0])
        if len(ci) < 8000:
            continue
        nz = [v for v in ci if v]
        if not nz or 100 * st.pstdev(nz) / st.mean(nz) < 1.0:
            dropped.append(c); continue
        traces[c] = ci
    regions = sorted(traces)
    n_hours = min(len(traces[c]) for c in regions)
    if dropped:
        print(f"[spatial] dropped constant traces: {', '.join(dropped)}")

    cv = {c: 100 * st.pstdev([v for v in traces[c] if v]) / st.mean([v for v in traces[c] if v]) for c in regions}
    by_cv = sorted(regions, key=lambda c: cv[c])
    k = min(args.homes, len(by_cv))
    idx = sorted(set(round(i * (len(by_cv) - 1) / (k - 1)) for i in range(k)))
    homes = [by_cv[i] for i in idx]
    print(f"[spatial] {len(regions)} reachable regions, {len(homes)} home regions, {args.samples} starts; "
          f"e_net={E_NET_KJ_PER_GB} kJ/GB (full-path marginal)")

    def valid(t, C):
        return all(traces[r][t + h] is not None for r in regions for h in range(C))

    rows = []
    for wl, w in WORKLOADS.items():
        C, P, S = w["C"], w["P"], w["S"]
        net_kwh = E_NET_KJ_PER_GB * S / 3600.0
        for tier in ("nvme", "sata"):
            dump_kwh = w[f"dump_{tier}"] / 3600.0
            rest_kwh = w[f"rest_{tier}"] / 3600.0
            egress = dump_kwh + net_kwh                 # source-charged half (dump + network)
            ideal_s, greedy_s, dp_s, ovh, Mg, Md, kill = [], [], [], [], [], [], []
            for home in homes:
                ci_h = traces[home]
                # candidate start hours with full C-hour validity for ALL regions
                cand = [t for t in range(0, n_hours - C) if ci_h[t] is not None]
                picks = cand if len(cand) <= args.samples * 3 else rng.sample(cand, args.samples * 3)
                got = 0
                for t in picks:
                    if got >= args.samples:
                        break
                    if not valid(t, C):
                        continue
                    got += 1
                    hourly = [{r: traces[r][t + h] for r in regions} for h in range(C)]
                    E = P
                    base = sum(hourly[h][home] for h in range(C)) * E          # B1 stay home
                    ideal = sum(min(hourly[h].values()) for h in range(C)) * E # B2 free migration

                    # B3 greedy chaser: cleanest region each hour, pay migration on switch
                    path, comp = cleanest_path_carbon(hourly, regions, E)
                    mech_g = 0.0; mg = 0
                    prev = home
                    for h in range(C):
                        if path[h] != prev:
                            mech_g += egress * hourly[h][prev] + rest_kwh * hourly[h][path[h]]
                            mg += 1
                        prev = path[h]
                    greedy = comp + mech_g

                    # B4 DP-optimal region path (migration-aware), start anchored at home
                    INF = float("inf")
                    cost = {r: hourly[0][r] * E + (0.0 if r == home else
                            egress * hourly[0][home] + rest_kwh * hourly[0][r]) for r in regions}
                    back = [{r: home for r in regions}]
                    for h in range(1, C):
                        ncost, bp = {}, {}
                        # precompute best stay vs best switch source
                        for r in regions:
                            # stay in r
                            best = cost[r]; bsrc = r
                            cr = hourly[h][r]
                            for rp in regions:
                                if rp == r:
                                    continue
                                cand_c = cost[rp] + egress * hourly[h][rp] + rest_kwh * cr
                                if cand_c < best:
                                    best = cand_c; bsrc = rp
                            ncost[r] = hourly[h][r] * E + best
                            bp[r] = bsrc
                        cost = ncost; back.append(bp)
                    rstar = min(regions, key=lambda r: cost[r])
                    dp = cost[rstar]
                    # recover migrations
                    md = 0; cur = rstar
                    for h in range(C - 1, 0, -1):
                        src = back[h][cur]
                        if src != cur:
                            md += 1
                        cur = src
                    if back[0][cur] != cur:  # initial home->cur counts if cur != home
                        pass
                    if cur != home:
                        md += 1

                    ideal_s.append(100 * (base - ideal) / base if base else 0)
                    greedy_s.append(100 * (base - greedy) / base if base else 0)
                    dp_s.append(100 * (base - dp) / base if base else 0)
                    ovh.append(100 * mech_g / base if base else 0)
                    Mg.append(mg); Md.append(md)
                    kill.append(1 if greedy > base + 1e-12 else 0)
            if not ideal_s:
                continue
            rows.append(dict(wl=wl, name=w["name"], tier=tier, C=C, P=P, S=S,
                             ideal=st.mean(ideal_s), greedy=st.mean(greedy_s), dp=st.mean(dp_s),
                             ovh=st.mean(ovh), Mg=st.mean(Mg), Md=st.mean(Md),
                             kill=100 * st.mean(kill), n=len(ideal_s)))

    print(f"\nORACLE spatial.  save% vs B1 (stay home).  B2 ideal=cleanest region/h FREE migration "
          f"(upper bound).  B3 greedy=chase+pay.  B4 dp=migration-aware optimal.")
    print(f"{'WL':3} {'C':>2}h {'S_GB':>5} {'tier':4} | {'B2idl%':>7} {'B3grd%':>7} {'ovh':>6} "
          f"{'Mg':>5} {'kill%':>6} || {'B4dp%':>6} {'Md':>5}")
    for wl in ["A4", "A5", "A7", "A8", "A2", "A1", "A6", "A3"]:
        for tier in ("nvme", "sata"):
            r = next((x for x in rows if x["wl"] == wl and x["tier"] == tier), None)
            if not r:
                continue
            print(f"{wl:3} {r['C']:>2}  {r['S']:>5} {tier:4} | {r['ideal']:7.1f} {r['greedy']:7.1f} "
                  f"{r['ovh']:6.1f} {r['Mg']:5.2f} {r['kill']:6.1f} || {r['dp']:6.1f} {r['Md']:5.2f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"\n[spatial] -> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
