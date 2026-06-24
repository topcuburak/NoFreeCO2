#!/usr/bin/env python3
"""Forecast-driven temporal scheduling: decide on a REAL predictor's CI, pay on actual CI.

Uses CarbonCast (open-source CNN-LSTM, pretrained) predicted-vs-actual CI per region. The scheduler
picks the C cleanest hours of the window by the PREDICTED CI but the carbon emitted (and the
mechanism overhead) is computed on the ACTUAL CI. Compared head-to-head with the ORACLE (pick by
actual). Quantifies: savings lost to misprediction, and how forecast error changes K (suspends) and
mechanism overhead -- using our measured per-suspend dump/restore energy.

Data: carboncast_forecasts/<REGION>/<REGION>_direct_96hr_CI_forecasts_0.csv, columns
[datetime, carbon_intensity_actual, avg_carbon_intensity_forecast]. Layout = consecutive 96-hour
daily forecast blocks (lead time 0..95 within a block); each block = one scheduling decision at the
forecast-refresh time, using that day's 96h forecast (window = first H of the block).

  python scripts/carbon_temporal_forecast.py --data carboncast_forecasts --out results/carbon_forecast.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from carbon_temporal import WORKLOADS, run_blocks, HORIZONS    # reuse params + block grouping


def load_blocks(path, blk=96):
    rows = list(csv.DictReader(open(path)))
    act = [float(r["carbon_intensity_actual"]) for r in rows]
    prd = [float(r["avg_carbon_intensity_forecast"]) for r in rows]
    blocks = []
    for i in range(0, len(rows) - blk + 1, blk):
        a = act[i:i + blk]; p = prd[i:i + blk]
        if all(x > 0 for x in a) and all(x > 0 for x in p):
            blocks.append((p, a))
    return blocks


# CarbonCast measured per-96h-forecast inference energy = 16.3 J = 4.527e-6 kWh (CPU, ford, 2155
# predicts/60s -> 4.07 J/predict x4). Charged once per scheduling decision for the forecast path only.
PRED_KWH = 16.3 / 3.6e6


def eval_one(sel_ci, act_ci, C, P, dump_kwh, rest_kwh, pred_kwh=0.0):
    """Pick C cleanest by sel_ci; emit + price mechanism on act_ci. pred_kwh = forecaster cost."""
    H = len(act_ci)
    naive = sum(act_ci[:C]) * P
    sel = sorted(sorted(range(H), key=lambda i: sel_ci[i])[:C])
    compute = sum(act_ci[i] for i in sel) * P
    blocks = run_blocks(sel)
    K = len(blocks) - 1
    mech = 0.0
    for b in range(K):
        mech += dump_kwh * act_ci[blocks[b][1]] + rest_kwh * act_ci[blocks[b + 1][0]]
    pred = pred_kwh * act_ci[0]                                   # one forecast at decision time
    return naive, naive - (compute + mech + pred), naive - compute, K   # naive, net, gross, K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="carboncast_forecasts")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), args.data) \
        if not os.path.isabs(args.data) else args.data

    regions = {}
    for d in sorted(glob.glob(os.path.join(root, "*", ""))):
        f = glob.glob(os.path.join(d, "*_CI_forecasts_0.csv"))
        if f:
            regions[os.path.basename(d.rstrip("/"))] = load_blocks(f[0])
    print(f"[fcast] {len(regions)} regions, {sum(len(b) for b in regions.values())} forecast blocks total")

    rows = []
    for wl, w in WORKLOADS.items():
        for tier in ("nvme", "sata"):
            dump_kwh = w[f"dump_{tier}"] / 3600.0
            rest_kwh = w[f"rest_{tier}"] / 3600.0
            for H in HORIZONS:
                if H < w["C"]:
                    continue
                onet, fnet, ok, fk = [], [], [], []
                for reg, blocks in regions.items():
                    for prd, act in blocks:
                        if len(act) < H:
                            continue
                        sp, sa = prd[:H], act[:H]
                        nv, on, og, oK = eval_one(sa, sa, w["C"], w["P"], dump_kwh, rest_kwh)   # ORACLE
                        _,  fn, fg, fK = eval_one(sp, sa, w["C"], w["P"], dump_kwh, rest_kwh,
                                                  pred_kwh=PRED_KWH)                            # FORECAST + pred cost
                        if nv <= 0:
                            continue
                        onet.append(100 * on / nv); fnet.append(100 * fn / nv)
                        ok.append(oK); fk.append(fK)
                if not onet:
                    continue
                rows.append(dict(wl=wl, tier=tier, C=w["C"], H=H,
                                 oracle=st.mean(onet), forecast=st.mean(fnet),
                                 lost=st.mean(onet) - st.mean(fnet),
                                 Ko=st.mean(ok), Kf=st.mean(fk), n=len(onet)))

    print(f"\n{'WL':4}{'C':>3}{'tier':>6}  oracle%  forecast%  lost(pp)   Ko -> Kf   capture%")
    for wl in WORKLOADS:
        for tier in ("nvme", "sata"):
            r = [x for x in rows if x["wl"] == wl and x["tier"] == tier and x["H"] == 24]
            if not r:
                continue
            r = r[0]
            cap = 100 * r["forecast"] / r["oracle"] if r["oracle"] else 0
            print(f"{wl:4}{r['C']:>3}{tier:>6}  {r['oracle']:6.1f}   {r['forecast']:7.1f}   {r['lost']:6.2f}   "
                  f"{r['Ko']:.2f}->{r['Kf']:.2f}   {cap:5.1f}%   (H=24)")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\n[fcast] -> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
