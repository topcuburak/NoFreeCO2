#!/usr/bin/env python3
"""Carbon benefit of temporal scheduling at 15-MINUTE granularity, net of suspend/restore cost.

Same deadline-budget model as carbon_temporal.py, but the scheduler decides every 15 min instead of
hourly. Per the brief, we REINTERPRET each hourly CI sample as one 15-min slot (no interpolation), so
a 4-hour span is now 16 CI values. Keeping the workload COMPUTE hours unchanged, a job that needs C
compute hours now needs 4C slots and a deadline of H hours is 4H slots.

THREE baselines are compared, all carbon measured over the same job:

  B1  NO temporal scheduling. Run now: the 4C slots contiguously from start t. This is the REFERENCE
      everything else is saved against (its own savings is 0 by definition).

  B2  IDEAL temporal scheduling, mechanism FREE. Place the 4C compute slots on the CLEANEST slots
      anywhere in the [t, t+4H] window (carbon-optimal placement), and charge ZERO suspend/restore.
      This is the theoretical upper bound on temporal savings.

  B3  REAL temporal scheduling, mechanism CHARGED. Same cleanest-slot placement as B2, but pay a
      MEASURED dump+restore for every internal gap, split-priced (dump half at the suspend-slot CI,
      restore half at the resume-slot CI), separately for NVMe and SATA.

Savings% are vs B1. Then per instance:
  ideal  = 100*(B1 - B2_carbon)/B1                # B2 savings, no overhead (upper bound)
  net    = 100*(B1 - B3_carbon)/B1                # B3 savings, after the mechanism
  ovh    = ideal - net = 100*mechanism/B1         # carbon the suspend/restore eats (percentage points)
  KILL   <=> net < 0                              # real temporal scheduling is WORSE than running now
B2 >= B3 always (mechanism >= 0); but B3 can fall BELOW B1 (net < 0) when the mechanism exceeds the
entire shifting benefit. At 15-min granularity even C=1h jobs (4 slots) can fragment, so they are no
longer automatically immune. B2 is tier-independent; B3 and ovh/kill are reported per tier.

DATA CLEANING: traces that are effectively CONSTANT (CV < 1%, e.g. hong_kong flat 360, indonesia flat
580) are placeholder / non-physical CI and are DROPPED. CI values <= 0 are treated as missing (load_ci).

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


def place(win, cslots, rankby):
    """Pick the cslots cleanest slots BY rankby (actual=oracle, predicted=forecast); cost on actual win.

    Returns (compute_ci_sum, K, Sdump, Srest) all in raw gCO2/kWh-sum units (multiply by e_slot for
    carbon). Sdump/Srest are the summed ACTUAL CI at the suspend/resume slot of every internal gap, so
    the mechanism is priced per tier OUTSIDE: mech = dump_kwh*Sdump + rest_kwh*Srest.
    """
    order = sorted(range(len(win)), key=lambda i: rankby[i])
    sel = sorted(order[:cslots])
    compute = sum(win[i] for i in sel)
    blocks = run_blocks(sel)
    K = len(blocks) - 1
    Sdump = sum(win[blocks[b][1]] for b in range(K))     # actual CI at each suspend+store
    Srest = sum(win[blocks[b + 1][0]] for b in range(K)) # actual CI at each load+resume
    return compute, K, Sdump, Srest


def region_mape(cv_pct):
    """Per-region forecast MAPE (fraction) from CI volatility, calibrated to MEASURED CarbonCast:
    flat grids (FPL/PJM/PL) 3-5%, volatile (CISO/DE/BPAT/ES) 13-19%. Linear in CV, clamped 3-20%."""
    return min(0.20, max(0.03, 0.0035 * cv_pct))


PHI = 0.9   # AR(1) autocorrelation of forecast error: real forecasters err with a slow bias, not
            # independent per-slot noise, so the predicted-cleanest slots stay CLUSTERED (CarbonCast
            # smooths -> suspends less). White noise (phi=0) would shred autocorrelation and inflate K.


def ar1_error(rng, n, sd, phi=PHI):
    """Length-n AR(1) multiplicative-error series with marginal sd ~= sd (stationary start)."""
    innov = sd * (1.0 - phi * phi) ** 0.5
    e = [0.0] * n
    prev = rng.gauss(0.0, sd)                      # stationary initial draw
    for i in range(n):
        prev = phi * prev + rng.gauss(0.0, innov)
        e[i] = prev
    return e


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
    traces, dropped = {}, []
    for c in sorted(os.listdir(root)):
        hits = glob.glob(os.path.join(root, c, f"**/*_{args.year}_hourly.csv"), recursive=True)
        if hits:
            ci = load_ci(hits[0])                       # CI<=0 -> None already
            if len(ci) < 8000:
                continue
            nz = [v for v in ci if v]
            if not nz:
                dropped.append((c, "empty")); continue
            cvv = 100 * st.pstdev(nz) / st.mean(nz)
            if cvv < 1.0:                               # effectively constant -> placeholder, not real CI
                dropped.append((c, f"constant CV={cvv:.1f}%")); continue
            traces[c] = ci
    if dropped:
        print("[15min] dropped non-physical traces: " + ", ".join(f"{c} ({why})" for c, why in dropped))

    # volatility (CV%) per region, then stratified pick across the CV range
    cv = {c: 100 * st.pstdev([v for v in ci if v]) / st.mean([v for v in ci if v]) for c, ci in traces.items()}
    by_cv = sorted(traces, key=lambda c: cv[c])
    k = min(args.regions, len(by_cv))
    idx = [round(i * (len(by_cv) - 1) / (k - 1)) for i in range(k)]
    regions = [by_cv[i] for i in sorted(set(idx))]
    print(f"[15min] {len(traces)} traces; {len(regions)} representative regions (stratified by CV), "
          f"{args.samples} random starts each, slot=15min")
    print("  regions (CV%): " + ", ".join(f"{c}={cv[c]:.0f}" for c in regions))

    sigma = {c: 1.25 * region_mape(cv[c]) for c in regions}   # gauss sd so E|err| ~= MAPE
    TIERS = ("nvme", "sata")

    # horizons in HOURS: a tight (2xC) and a loose (4xC) slack, capped at 48 h
    rows = []
    for wl, w in WORKLOADS.items():
        C = w["C"]
        cslots = int(round(C / SLOT_H))
        P = w["P"]
        dk = {tier: w[f"dump_{tier}"] / 3600.0 for tier in TIERS}    # kJ -> kWh
        rk = {tier: w[f"rest_{tier}"] / 3600.0 for tier in TIERS}
        for slack, Hh in (("tight", min(48, 2 * C)), ("loose", min(48, 4 * C))):
            hslots = int(round(Hh / SLOT_H))
            if hslots <= cslots:
                continue
            ideal = []                                              # B2 oracle free-shift savings%
            Ko, Kf = [], []                                         # K oracle / forecast
            mis_kill = []                                           # MISPREDICTION-ONLY kill (mech free), tier-indep
            # per mode (oracle 'o' / forecast 'f') per tier: savings list + kill loss list
            net = {m: {t: [] for t in TIERS} for m in "of"}
            ovh = {m: {t: [] for t in TIERS} for m in "of"}
            klist = {m: {t: [] for t in TIERS} for m in "of"}       # net of KILLED instances only
            for c in regions:
                ci = traces[c]
                sd = sigma[c]
                starts = valid_starts(ci, hslots)
                if not starts:
                    continue
                picks = starts if len(starts) <= args.samples else rng.sample(starts, args.samples)
                for t in picks:
                    win = ci[t:t + hslots]
                    e = P * SLOT_H
                    naive = sum(win[:cslots]) * e
                    if naive <= 0:
                        continue
                    err = ar1_error(rng, hslots, sd)                                # autocorrelated
                    pred = [max(1.0, v * (1.0 + err[i])) for i, v in enumerate(win)] # forecast CI
                    co, ko, sdmp_o, srst_o = place(win, cslots, win)                 # oracle: rank by actual
                    cf, kf, sdmp_f, srst_f = place(win, cslots, pred)                # forecast: rank by predicted
                    ideal.append(100 * (naive - co * e) / naive)
                    Ko.append(ko); Kf.append(kf)
                    misfree = 100 * (naive - cf * e) / naive        # forecast placement, mechanism FREE
                    mis_kill.append(1 if misfree < -1e-9 else 0)    # only misprediction can flip this
                    for (m, comp, sdmp, srst) in (("o", co, sdmp_o, srst_o), ("f", cf, sdmp_f, srst_f)):
                        for tier in TIERS:
                            mech = dk[tier] * sdmp + rk[tier] * srst
                            n = 100 * (naive - comp * e - mech) / naive
                            net[m][tier].append(n)
                            ovh[m][tier].append(100 * mech / naive)
                            if n < -1e-9:
                                klist[m][tier].append(n)
            if not ideal:
                continue
            row = dict(wl=wl, name=w["name"], C=C, slack=slack, H=Hh,
                       ideal=st.mean(ideal), Ko=st.mean(Ko), Kf=st.mean(Kf),
                       kill_mis=100 * st.mean(mis_kill), n=len(ideal))
            for m in "of":
                for tier in TIERS:
                    row[f"net_{m}_{tier}"] = st.mean(net[m][tier])
                    row[f"ovh_{m}_{tier}"] = st.mean(ovh[m][tier])
                    nn = len(net[m][tier]); kk = klist[m][tier]
                    row[f"kill_{m}_{tier}"] = 100 * len(kk) / nn if nn else 0.0
                    row[f"closs_{m}_{tier}"] = st.mean(kk) if kk else 0.0      # conditional mean loss
            rows.append(row)

    def table(mode, title):
        tag = "oracle" if mode == "o" else "forecast"
        kcol = "Ko" if mode == "o" else "Kf"
        print(f"\n=== {title} ===")
        print(f"save% vs B1 (run-now).  B2 ideal = oracle free-shift (upper bound).  B3 = shift + "
              f"measured suspend/restore ({tag} placement).")
        print(f"ovh=B2-B3 (pp).  kill%=B3<0 freq.  cml=conditional mean loss (avg net | killed).")
        print(f"{'WL':3} {'C':>2}h {'slack':5} {'H':>3}h | {'B2idl%':>7} || "
              f"{'nvB3%':>6} {'ovh':>4} {'kil%':>5} {'cml':>5} || "
              f"{'saB3%':>6} {'ovh':>4} {'kil%':>5} {'cml':>5} | {'K':>4}")
        for wl in ["A4", "A5", "A7", "A8", "A2", "A1", "A6", "A3"]:
            for slack in ("tight", "loose"):
                r = next((x for x in rows if x["wl"] == wl and x["slack"] == slack), None)
                if not r:
                    continue
                print(f"{wl:3} {r['C']:>2}  {slack:5} {r['H']:>3}  | {r['ideal']:7.1f} || "
                      f"{r[f'net_{mode}_nvme']:6.1f} {r[f'ovh_{mode}_nvme']:4.2f} "
                      f"{r[f'kill_{mode}_nvme']:5.1f} {r[f'closs_{mode}_nvme']:5.1f} || "
                      f"{r[f'net_{mode}_sata']:6.1f} {r[f'ovh_{mode}_sata']:4.2f} "
                      f"{r[f'kill_{mode}_sata']:5.1f} {r[f'closs_{mode}_sata']:5.1f} | {r[kcol]:4.2f}")

    table("o", "ORACLE (perfect foresight)")
    table("f", "FORECAST (CarbonCast-calibrated misprediction, decide-on-predicted / pay-on-actual)")

    # KILL DECOMPOSITION: separate the two flippers.
    #   mech-only  = oracle placement + real mechanism  (perfect foresight; only the mechanism flips it)
    #   mis-only   = forecast placement + FREE mechanism (no mechanism; only misprediction flips it)
    #   combined   = forecast placement + real mechanism
    print(f"\n=== KILL DECOMPOSITION: mechanism vs misprediction (kill% = B3<0 frequency) ===")
    print(f"mis-only is tier-independent (mechanism set free).  dom = larger single flipper per tier.")
    print(f"{'WL':3} {'C':>2}h {'slack':5} | {'mis-only':>8} || {'nv mech':>7} {'nv comb':>7} {'nv dom':>7} "
          f"|| {'sa mech':>7} {'sa comb':>7} {'sa dom':>7}")
    for wl in ["A4", "A5", "A7", "A8", "A2", "A1", "A6", "A3"]:
        for slack in ("tight", "loose"):
            r = next((x for x in rows if x["wl"] == wl and x["slack"] == slack), None)
            if not r:
                continue
            mis = r["kill_mis"]
            cells = ""
            for tier in TIERS:
                mech = r[f"kill_o_{tier}"]
                comb = r[f"kill_f_{tier}"]
                dom = "MECH" if mech > mis else "mispred"
                cells += f" {mech:7.1f} {comb:7.1f} {dom:>7} ||"
            print(f"{wl:3} {r['C']:>2}  {slack:5} | {mis:8.1f} ||{cells}".rstrip(" |"))

    if args.out:
        with open(args.out, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"\n[15min] -> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
