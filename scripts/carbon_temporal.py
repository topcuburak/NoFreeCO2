#!/usr/bin/env python3
"""Carbon benefit of temporal (suspend-in-place) scheduling, net of suspend/restore mechanism cost.

Model (deadline-budget shifting): a workload is kicked at hour t; it needs C hours of COMPUTE and
has a deadline window of H hours, so it sees H hourly CI values. Carbon-aware runs the C *cleanest*
hours in [t, t+H] and suspends the rest; each run<->suspend transition is a dump+restore = E_mech.

  naive_gco2  = sum(CI[t : t+C]) * P_kw                         # run now, C contiguous hours
  compute     = sum(C cleanest CI in window) * P_kw            # shifted compute
  K           = (# run-blocks) - 1                              # internal suspend gaps
  mech_gco2   = sum over gaps of dump_kWh*CI_suspend_hr + restore_kWh*CI_resume_hr  # split pricing
  aware_gco2  = compute + mech_gco2
  net savings = naive - aware ;  gross = naive - compute ;  overhead = mech / gross

Swept Monte-Carlo over start hours, per workload, per tier (NVMe/SATA), per country (hourly traces
in carbon_data/<country>/*_<year>_hourly.csv, 'direct' CI). Outputs a summary table.

  python scripts/carbon_temporal.py --year 2023 --countries germany,france --step 3
  python scripts/carbon_temporal.py --year 2023 --all --step 6 --out results/carbon_temporal_2023.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st

# P in kW; C = compute hours; energy split into DUMP half (suspend+store) and RESTORE half
# (load+resume) per tier, in kJ -- so the dump is priced at the suspend-hour CI and the restore at
# the resume-hour CI (not their average). Measured from the campaign (A2-A8 raw; A1 NVMe from its
# doc, A1 SATA estimated from the GPU-SATA dump/restore ratio). dump_X / rest_X.
WORKLOADS = {
    "A1": dict(name="FSDP Llama-8B FT",     C=4,  P=1.471, dump_nvme=25.66, rest_nvme=10.35, dump_sata=143.0, rest_sata=121.7),
    "A2": dict(name="vLLM batch inference", C=2,  P=0.497, dump_nvme=3.70,  rest_nvme=1.60,  dump_sata=19.18, rest_sata=16.50),
    "A3": dict(name="ViT-Huge train",       C=12, P=0.526, dump_nvme=3.39,  rest_nvme=1.39,  dump_sata=20.83, rest_sata=17.92),
    "A4": dict(name="DLRM train",           C=1,  P=0.312, dump_nvme=3.31,  rest_nvme=1.35,  dump_sata=19.40, rest_sata=16.87),
    "A5": dict(name="GAPBS graph",          C=1,  P=0.255, dump_nvme=8.53,  rest_nvme=3.91,  dump_sata=35.21, rest_sata=24.60),
    "A6": dict(name="gem5 sim",             C=8,  P=0.148, dump_nvme=9.40,  rest_nvme=5.25,  dump_sata=32.76, rest_sata=30.61),
    "A7": dict(name="DuckDB multi-proc",    C=1,  P=0.311, dump_nvme=28.14, rest_nvme=7.06,  dump_sata=29.51, rest_sata=44.90),
    "A8": dict(name="DuckDB multi-thread",  C=1,  P=0.302, dump_nvme=19.72, rest_nvme=10.17, dump_sata=55.19, rest_sata=41.50),
}
HORIZONS = [4, 6, 8, 12, 16, 24, 36, 48]


def load_ci(path):
    out = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)                                   # header
        for row in r:
            try:
                v = float(row[4])                       # CI gCO2eq/kWh (direct)
                out.append(v if v > 0 else None)
            except (IndexError, ValueError):
                out.append(None)
    return out


def run_blocks(sorted_idx):
    """Group sorted hour-indices into contiguous run-blocks; return list of (start,end)."""
    blocks = []
    s = p = sorted_idx[0]
    for i in sorted_idx[1:]:
        if i == p + 1:
            p = i
        else:
            blocks.append((s, p)); s = p = i
    blocks.append((s, p))
    return blocks


def eval_instance(ci, t, H, C, P, dump_kwh, rest_kwh):
    win = ci[t:t + H]
    if any(v is None for v in win):
        return None
    naive = sum(win[:C]) * P
    order = sorted(range(H), key=lambda i: win[i])      # cleanest-first
    sel = sorted(order[:C])
    compute = sum(win[i] for i in sel) * P
    blocks = run_blocks(sel)
    K = len(blocks) - 1
    mech = 0.0
    for b in range(K):                                  # gap between block b and b+1
        ci_dump = win[blocks[b][1]]                     # CI_first: suspend+store (DUMP) happen here
        ci_rest = win[blocks[b + 1][0]]                 # CI_second: load+resume (RESTORE) happen here
        mech += dump_kwh * ci_dump + rest_kwh * ci_rest
    aware = compute + mech
    gross = naive - compute
    return dict(naive=naive, compute=compute, mech=mech, aware=aware,
                gross=gross, net=naive - aware, K=K)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--countries", default="", help="comma list (folder names); default with --all = every country")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--step", type=int, default=6, help="start-hour stride for the Monte Carlo")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "carbon_data")
    if args.all:
        countries = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    else:
        countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    # load CI per country (first matching *_<year>_hourly.csv anywhere under the country dir)
    traces = {}
    for c in countries:
        hits = glob.glob(os.path.join(root, c, f"**/*_{args.year}_hourly.csv"), recursive=True) \
            or glob.glob(os.path.join(root, c, f"*_{args.year}_hourly.csv"))
        if hits:
            ci = load_ci(hits[0])
            if len(ci) >= 8000:
                traces[c] = ci
    print(f"[carbon] {len(traces)} country traces ({args.year}), step={args.step}h")

    rows = []
    for wl, w in WORKLOADS.items():
        for tier in ("nvme", "sata"):
            dump_kwh = w[f"dump_{tier}"] / 3600.0       # kJ -> kWh, dump half (suspend+store)
            rest_kwh = w[f"rest_{tier}"] / 3600.0       # restore half (load+resume)
            for H in HORIZONS:
                if H < w["C"]:
                    continue
                # accumulate across all countries' instances
                net_pct, gross_pct, ovh_pct, Ks, pos = [], [], [], [], []
                for c, ci in traces.items():
                    for t in range(0, len(ci) - H, args.step):
                        r = eval_instance(ci, t, H, w["C"], w["P"], dump_kwh, rest_kwh)
                        if r is None or r["naive"] <= 0:
                            continue
                        net_pct.append(100 * r["net"] / r["naive"])
                        gross_pct.append(100 * r["gross"] / r["naive"])
                        ovh_pct.append(100 * r["mech"] / r["gross"] if r["gross"] > 0 else 0.0)
                        Ks.append(r["K"]); pos.append(1 if r["net"] > 0 else 0)
                if not net_pct:
                    continue
                rows.append(dict(wl=wl, name=w["name"], tier=tier, C=w["C"], H=H,
                                 net=st.mean(net_pct), gross=st.mean(gross_pct),
                                 ovh=st.mean(ovh_pct), K=st.mean(Ks),
                                 pos=100 * st.mean(pos), n=len(net_pct)))

    # print compact: one block per workload, NVMe vs SATA net% across horizons
    print(f"\nassumed compute C (h): " + "  ".join(f"{k}={v['C']}" for k, v in WORKLOADS.items()))
    hdr = "H=" + " ".join(f"{h:>5}" for h in HORIZONS)
    for wl in WORKLOADS:
        print(f"\n{wl} {WORKLOADS[wl]['name']} (C={WORKLOADS[wl]['C']}h, P={WORKLOADS[wl]['P']}kW)   net carbon savings %")
        for tier in ("nvme", "sata"):
            cells = {r["H"]: r["net"] for r in rows if r["wl"] == wl and r["tier"] == tier}
            kk = {r["H"]: r["K"] for r in rows if r["wl"] == wl and r["tier"] == tier}
            line = " ".join(f"{cells[h]:5.1f}" if h in cells else "   - " for h in HORIZONS)
            kline = " ".join(f"{kk[h]:5.1f}" if h in kk else "   - " for h in HORIZONS)
            print(f"  {tier:4} net% {hdr.replace('H=','')}\n       {line}   (K susp: {kline})")

    if args.out:
        with open(args.out, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"\n[carbon] -> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
