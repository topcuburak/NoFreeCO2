#!/usr/bin/env python3
"""N2 simulator v2: EuroSys'24 policies + MEASURED mechanism cost injection.

Policies follow Sukprasert et al. (EuroSys'24, artifact umassos/decarbonization-potential,
Apache-2.0) exactly, per arrival hour i, zone, job length C, absolute slack s (window =
C + s slots):
  run_now    : contiguous [i, i+C)                                    (their slack_0)
  defer      : best contiguous start within the window                (their non_interrupt)
  interrupt  : the C lowest-CI slots within the window                (their interrupt)
Their bounds charge NOTHING for deferring or fragmenting. We inject the N1 measured costs:
  - every run-gap costs min(park_w x gap, E_mech dump+resume) at the gap-start CI
  - theta (aggressiveness gate): a slot-swap into a cheaper slot is only taken if its
    relative CI saving > theta. theta=0 == their ideal (max aggressive).
  - granularity g (min): decision/CI interval; 60 = their hourly; finer from carbon/ tree.
Verification: --verify imports THEIR task() and asserts our theta=0 interrupt == theirs.

  python scripts/sim_n2.py --verify --artifact ~/decarbonization-potential
  python scripts/sim_n2.py --carbon artifact --artifact ~/dp --zones DE,PL --jobs A4_dlrm,A6_gem5 \
      --thetas 0,0.02,0.05,0.1 --slacks 24 --out n2_sweep.csv
Regimes: infinite (v2, this file). Limited-capacity fleet = v3 (--utilization reserved).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
CARBON = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "carbon")

# measured workload catalog (ford, NVMe): compute hours, active W, footprint GB,
# suspend+resume round-trip J (measured), park W (GPU 74/board; CPU DRAM-hold).
WORKLOADS = {
    "A1_llama_fsdp": dict(C_h=4,  P_w=1471, S_gb=148, Emech_j=36000, park_w=4 * 74),
    "A2_vllm":       dict(C_h=2,  P_w=497,  S_gb=41,  Emech_j=5300,  park_w=74),
    "A3_vit":        dict(C_h=12, P_w=526,  S_gb=35,  Emech_j=4790,  park_w=74),
    "A4_dlrm":       dict(C_h=1,  P_w=312,  S_gb=33,  Emech_j=4700,  park_w=74),
    "A5_gapbs":      dict(C_h=1,  P_w=255,  S_gb=71,  Emech_j=12400, park_w=0.05 * 71),
    "A6_gem5":       dict(C_h=8,  P_w=148,  S_gb=50,  Emech_j=14700, park_w=0.05 * 50),
    "A7_duck_mp":    dict(C_h=1,  P_w=311,  S_gb=100, Emech_j=22600, park_w=0.05 * 100),
    "A8_duck_mt":    dict(C_h=1,  P_w=302,  S_gb=100, Emech_j=16400, park_w=0.05 * 100),
}


# ---------- carbon sources ----------
def load_artifact_carbon(art: str, year: int) -> dict[str, list[float]]:
    """Their combined_carbon.csv (hourly, 123 zones, 2020-2022)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(os.path.expanduser(art), "shared_data", "combined_carbon.csv"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"].dt.year == year].reset_index(drop=True)
    return {z: df[z].tolist() for z in df.columns if z != "datetime"}


def load_tree_carbon(zone: str, year: int, g: int) -> list[float]:
    """Our carbon/ tree at granularity g minutes (60 = real hourly, else synthetic)."""
    pat = (f"{CARBON}/real_60min/**/{zone}_{year}_hourly.csv" if g == 60
           else f"{CARBON}/synthetic_{g}min/**/{zone}_{year}_{g}min.csv")
    hits = glob.glob(pat, recursive=True)
    if not hits:
        raise SystemExit(f"[sim] no CI file: zone={zone} year={year} g={g}")
    out = []
    with open(hits[0], newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            out.append(float(row[4]) if row[4] else float("nan"))
    return out


# ---------- policies (their definitions; theta generalizes interrupt) ----------
def interrupt_slots(ci, i, k, window, theta):
    """theta=0: exactly the k lowest-CI slots in [i, i+window) (their 'interrupt').
    theta>0: start from run-now, accept cheapest swaps only while relative saving > theta."""
    end = min(i + window, len(ci))
    idx = list(range(i, end))
    if len(idx) < k:
        return None
    chosen = set(idx[:k])
    rest = sorted((ci[j], j) for j in idx[k:])
    for ci_new, j_new in rest:
        ci_old, j_old = max((ci[j], j) for j in chosen)
        if ci_old <= 0 or (ci_old - ci_new) / ci_old <= theta:
            break
        chosen.remove(j_old); chosen.add(j_new)
    return sorted(chosen)


def defer_start(ci, i, k, window):
    """Best contiguous start in the window (their 'non_interrupt')."""
    end = min(i + window, len(ci))
    if end - i < k:
        return None
    best, bs = None, i
    run = sum(ci[i:i + k])
    best = run
    for s in range(i + 1, end - k + 1):
        run += ci[s + k - 1] - ci[s - 1]
        if run < best:
            best, bs = run, s
    return bs


# ---------- cost injection ----------
def gap_cost_j(wl, gap_slots, dt_h):
    """Energy to bridge one execution gap: park in place vs dump+resume (break-even)."""
    return min(wl["park_w"] * gap_slots * dt_h * 3600.0, wl["Emech_j"])


def simulate(ci, i, wl, slots, dt_h):
    """Carbon (g CO2) for a chosen slot set, with mechanism/hold overhead at event CI."""
    E_slot = wl["P_w"] * dt_h * 3600.0
    exec_g = sum(ci[s] * E_slot for s in slots) / 3.6e6
    over_g = 0.0
    n_gap = 0
    prev = i - 1                                           # arrival; lead deferral is a gap too
    for s in slots:
        gap = s - prev - 1
        if gap > 0:
            n_gap += 1
            over_g += gap_cost_j(wl, gap, dt_h) * ci[prev + 1] / 3.6e6
        prev = s
    return exec_g, over_g, n_gap


# ---------- verification vs their artifact ----------
def verify(art):
    import importlib, sys, tempfile, pandas as pd
    os.chdir(os.path.join(os.path.expanduser(art), "sim_temporal")); sys.path.insert(0, ".")
    m = importlib.import_module("experiment_vary_job_slack")
    out = tempfile.mkdtemp(); bad = 0
    for zone, job, slack in [("DE", 6, 24), ("FR", 24, 24), ("PL", 1, 24)]:
        m.task(zone, job, slack, out)
        theirs = pd.read_csv(f"{out}/{zone}.csv")
        ci = m.carbon_df[zone].tolist()
        for i in range(len(theirs)):
            slots = interrupt_slots(ci, i, job, job + slack, 0.0)
            ours = round(sum(ci[s] for s in slots), 2)
            if abs(ours - theirs.interrupt[i]) > 0.05:
                bad += 1
                if bad < 5:
                    print(f"  MISMATCH {zone} i={i}: ours={ours} theirs={theirs.interrupt[i]}")
        print(f"  {zone} job={job} slack={slack}: {len(theirs)} arrivals checked")
    print("[verify] PASS -- theta=0 interrupt == artifact" if not bad
          else f"[verify] FAIL: {bad} mismatches")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="assert theta=0 == artifact task()")
    ap.add_argument("--artifact", default=None, help="decarbonization-potential clone path")
    ap.add_argument("--carbon", choices=["artifact", "tree"], default="artifact")
    ap.add_argument("--zones", default="DE,FR,GB,PL,US-CAL-CISO,US-MIDA-PJM")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--gran", type=int, default=60, help="minutes; tree source only for <60")
    ap.add_argument("--jobs", default=",".join(WORKLOADS), help="workload names")
    ap.add_argument("--thetas", default="0,0.02,0.05,0.1,0.2")
    ap.add_argument("--slacks", default="24", help="absolute slack HOURS beyond job length")
    ap.add_argument("--arrive-every-h", type=float, default=1.0)
    ap.add_argument("--utilization", type=float, default=None,
                    help="RESERVED v3: limited-capacity fleet regime")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.verify:
        raise SystemExit(0 if verify(args.artifact) else 1)
    if args.utilization is not None:
        raise SystemExit("[sim] limited-capacity fleet regime lands in v3")

    dt_h = args.gran / 60.0
    if args.carbon == "artifact":
        assert args.gran == 60, "artifact carbon is hourly"
        cimap = load_artifact_carbon(args.artifact, args.year)
        zones = [z for z in args.zones.split(",") if z in cimap]
    else:
        zones = args.zones.split(","); cimap = {}
    thetas = [float(x) for x in args.thetas.split(",")]
    slacks = [int(x) for x in args.slacks.split(",")]
    wls = {w: WORKLOADS[w] for w in args.jobs.split(",")}
    step = max(1, round(args.arrive_every_h / dt_h))

    rows = []
    for zone in zones:
        ci = cimap.get(zone) or load_tree_carbon(zone, args.year, args.gran)
        ci = [c for c in ci]                                # NaN slots stay (skipped jobs)
        for wname, wl in wls.items():
            k = max(1, round(wl["C_h"] / dt_h))
            for slack_h in slacks:
                window = k + round(slack_h / dt_h)
                for theta in thetas:
                    tot = defaultdict(float); n = 0
                    for i in range(0, len(ci) - window, step):
                        slots = interrupt_slots(ci, i, k, window, theta)
                        if slots is None or any(ci[s] != ci[s] for s in slots):
                            continue
                        E_slot = wl["P_w"] * dt_h * 3600.0
                        base_g = sum(ci[i + j] * E_slot for j in range(k)) / 3.6e6
                        exec_g, over_g, n_gap = simulate(ci, i, wl, slots, dt_h)
                        tot["base"] += base_g; tot["exec"] += exec_g
                        tot["over"] += over_g; tot["gaps"] += n_gap; n += 1
                    if not n or tot["base"] <= 0:
                        continue
                    gross = 100 * (tot["base"] - tot["exec"]) / tot["base"]
                    net = gross - 100 * tot["over"] / tot["base"]
                    rows.append(dict(zone=zone, wl=wname, gran=args.gran, slack_h=slack_h,
                                     theta=theta, gross_pct=round(gross, 3),
                                     net_pct=round(net, 3), gaps_per_job=round(tot["gaps"] / n, 2),
                                     n_jobs=n))
                    print(f"{zone:14} {wname:13} g={args.gran:3} s={slack_h:4} th={theta:4.2f} "
                          f"gross={gross:6.2f} net={net:6.2f} gaps/job={tot['gaps']/n:5.2f}")
    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        print(f"[sim] -> {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
