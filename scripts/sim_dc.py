#!/usr/bin/env python3
"""sim_dc: discrete-event datacenter simulator for carbon-aware scheduling under
MEASURED mechanism cost (N1). Built ON the EuroSys'24 framework (Sukprasert et al.,
artifact umassos/decarbonization-potential): their carbon data, their GCP latency
matrix, their latency-limit + idle-capacity axes -- with an actual ONLINE scheduler
and measured transition costs instead of zero-overhead oracle bounds.

Layers (composable):
  CI view    : --forecast oracle | noise:<MAPE%>          (per-zone)
  Capacity   : --utilization U -> per-zone K = ceil(peak_blind / U) (blind-peak
               calibrated). PARK holds a slot, dumped SUSP frees it. U=0 infinite.
  Temporal   : pause/resume within slack window, theta-gated; pause bills
               min(park, measured E_mech pair) at event CI.
  Spatial    : --latency-limit MS enables migration to zones within the GCP latency
               limit of the job's HOME zone (their SLO semantics). Migration bills
               E_mech pair + WAN transfer (S_gb x --wan-j-per-gb) at src/dst mean CI
               and inflates completion time by S/--wan-bw-gbps + measured latencies.
  Policy     : blind | ca_costblind (their ideal, online) | ca_costaware (N1 in the
               loop: pause if marginal slot benefit > this transition's bill;
               migrate if remaining-work benefit > migration bill)

  python scripts/sim_dc.py --carbon artifact --artifact ~/dp --zones spatial34 \
      --policy ca_costaware --theta 0.05 --slack-h 24 --utilization 0.9 \
      --latency-limit 100 --out dc_spatial.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
CARBON = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "carbon")

WORKLOADS = {
    "A1_llama_fsdp": dict(C_h=4,  P_w=1471, S_gb=148, Emech_j=36000, rt_lat_s=72,  park_w=296),
    "A2_vllm":       dict(C_h=2,  P_w=497,  S_gb=41,  Emech_j=5300,  rt_lat_s=20,  park_w=74),
    "A3_vit":        dict(C_h=12, P_w=526,  S_gb=35,  Emech_j=4790,  rt_lat_s=19,  park_w=74),
    "A4_dlrm":       dict(C_h=1,  P_w=312,  S_gb=33,  Emech_j=4700,  rt_lat_s=18,  park_w=74),
    "A5_gapbs":      dict(C_h=1,  P_w=255,  S_gb=71,  Emech_j=12400, rt_lat_s=56,  park_w=3.6),
    "A6_gem5":       dict(C_h=8,  P_w=148,  S_gb=50,  Emech_j=14700, rt_lat_s=67,  park_w=2.5),
    "A7_duck_mp":    dict(C_h=1,  P_w=311,  S_gb=100, Emech_j=22600, rt_lat_s=88,  park_w=5.0),
    "A8_duck_mt":    dict(C_h=1,  P_w=302,  S_gb=100, Emech_j=16400, rt_lat_s=72,  park_w=5.0),
}
RUN, PARK, SUSP, DONE = "RUN", "PARK", "SUSP", "DONE"

# N1 closed-form (ford-fit, NVMe): the ANALYTICAL model as the simulator's decision data.
# The A1-A8 catalog above is only the model's VALIDATION set; trace jobs get their cost
# from these formulas given (S_gb, ngpu). BW GB/s: suspend 5.5, store 6.3, load 12.8,
# resume 15.5; power W: GPU legs 140+100n, storage legs 140+74n (boards parked).
def n1_gpu_cost(S_gb, ngpu, P_active_w=None):
    t_susp, t_store = S_gb / 5.5, S_gb / 6.3
    t_load, t_res = S_gb / 12.8, S_gb / 15.5
    e = (140 + 100 * ngpu) * (t_susp + t_res) + (140 + 74 * ngpu) * (t_store + t_load)
    return dict(C_h=None, P_w=P_active_w or 250.0 * ngpu, S_gb=S_gb,
                Emech_j=round(e, 1), rt_lat_s=round(t_susp + t_store + t_load + t_res, 1),
                park_w=74.0 * ngpu)


def load_trace(path, limit=0):
    """Generic execution-trace CSV: arrive_h, duration_h, ngpu, mem_gb[, power_w].
    Returns [(arrive_h, wname, wl)] with N1-derived costs; wname buckets by ngpu."""
    rows = []
    with open(os.path.expanduser(path), newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            ngpu = max(1, int(float(r["ngpu"])))
            wl = n1_gpu_cost(float(r["mem_gb"]), ngpu,
                             float(r["power_w"]) if r.get("power_w") else None)
            wl["C_h"] = max(0.25, float(r["duration_h"]))
            rows.append((float(r["arrive_h"]), f"trace_g{min(ngpu, 8)}", wl))
    return rows


# ---------------- data loading ----------------
def load_artifact(args):
    """carbon (per zone) + latency matrix, intersected like their capacity_latency."""
    import pandas as pd
    art = os.path.expanduser(args.artifact)
    cdf = pd.read_csv(os.path.join(art, "shared_data", "combined_carbon.csv"))
    cdf["datetime"] = pd.to_datetime(cdf["datetime"])
    cdf = cdf[cdf["datetime"].dt.year == args.year].reset_index(drop=True)
    lat = pd.read_csv(os.path.join(art, "shared_data", "gcp_latency_matrix.csv"),
                      index_col="origin")
    lat = lat.loc[:, ~lat.columns.duplicated()]
    common = [z for z in lat.columns if z in cdf.columns and z in lat.index]
    if args.zones == "spatial34":
        zones = common
    else:
        zones = [z for z in args.zones.split(",") if z in cdf.columns]
    ci = {z: cdf[z].tolist() for z in zones}
    latms = {(a, b): float(lat.loc[a, b]) for a in common for b in common} if common else {}
    return ci, latms


def load_tree_zone(zone, year, g):
    sub = {5: "real_5min", 15: "real_15min", 30: "real_30min", 60: "real_60min"}
    pats = ([f"{CARBON}/{sub[g]}/**/{zone}_{year}_*.csv"] if g in sub else []) + \
        [f"{CARBON}/synthetic_{g}min/**/{zone}_{year}_{g}min.csv"]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            out = []
            with open(hits[0], newline="") as f:
                r = csv.reader(f); next(r)
                for row in r:
                    out.append(float(row[4]) if row[4] else float("nan"))
            return out
    raise SystemExit(f"[sim_dc] no CI for {zone} {year} g={g}")


class CiView:
    def __init__(self, ci, mode, seed):
        self.ci = ci; self.rng = random.Random(seed)
        self.mape = float(mode.split(":")[1]) / 100.0 if mode.startswith("noise:") else None
    def now(self, t):
        return self.ci[t]
    def ahead(self, t, h):
        v = self.ci[min(t + h, len(self.ci) - 1)]
        if self.mape is None or h == 0:
            return v
        sd = self.mape * math.sqrt(min(h, 48) / 24.0)
        return v * max(0.05, 1.0 + self.rng.gauss(0.0, sd * 1.2533))


# ---------------- engine ----------------
class Job:
    _n = 0
    def __init__(self, wname, wl, t, zone, dt_h):
        Job._n += 1; self.id = Job._n
        self.w, self.wl = wname, wl
        self.arrive, self.home, self.zone = t, zone, zone
        self.slots_left = max(1, round(wl["C_h"] / dt_h))
        self.deadline = None
        self.state = SUSP                       # queued: no resources, no cost
        self.exec_g = self.over_g = 0.0
        self.transitions = self.migrations = 0
        self.lat_s = 0.0


def blind_peak(arr, dt_h, T):
    """Peak concurrency if every job runs immediately (K calibration baseline)."""
    delta = [0] * (T + 2)
    for (t, _w, wl) in arr:
        k = max(1, round(wl["C_h"] / dt_h))
        delta[t] += 1; delta[min(t + k, T + 1)] -= 1
    peak = cur = 0
    for d in delta:
        cur += d; peak = max(peak, cur)
    return max(1, peak)


def simulate(args):
    if args.carbon == "artifact":
        assert args.gran == 60
        cimap, latms = load_artifact(args)
    else:
        cimap = {z: load_tree_zone(z, args.year, args.gran) for z in args.zones.split(",")}
        latms = {}
    zones = list(cimap)
    T = min(len(c) for c in cimap.values())
    dt_h = args.gran / 60.0
    wls = {w: WORKLOADS[w] for w in args.jobs.split(",")}
    views = {z: CiView(cimap[z], args.forecast, args.seed + i) for i, z in enumerate(zones)}
    horizon = round(args.slack_h / dt_h)
    arrive_every = max(1, round(args.arrive_every_h / dt_h))
    wnames = list(wls)

    # job population: real execution trace (N1 closed-form costs) or synthetic catalog
    if args.trace:
        rows = load_trace(args.trace, args.trace_limit)
        arrivals = {z: [] for z in zones}
        for i, (ah, wname, wl) in enumerate(sorted(rows)):
            t = round(ah / dt_h)
            if t + round(wl["C_h"] / dt_h) + horizon + 4 < T:
                arrivals[zones[i % len(zones)]].append((t, wname, wl))   # RR zone placement
        print(f"[sim_dc] trace {args.trace}: {sum(len(a) for a in arrivals.values())} jobs")
    else:
        arrivals = {z: [(t, wnames[(t // arrive_every) % len(wnames)],
                         wls[wnames[(t // arrive_every) % len(wnames)]])
                        for t in range(0, T - horizon - 60, arrive_every)] for z in zones}
    K = {z: (math.inf if not args.utilization else
             math.ceil(blind_peak(arrivals[z], dt_h, T) / args.utilization)) for z in zones}

    jobs, active = [], []
    occ = {z: set() for z in zones}             # jobs holding a slot (RUN/PARK) per zone
    arr_idx = {z: 0 for z in zones}
    wan_bw = args.wan_bw_gbps

    for t in range(T):
        for z in zones:                          # arrivals
            ai = arr_idx[z]
            while ai < len(arrivals[z]) and arrivals[z][ai][0] == t:
                _t, wname, wl = arrivals[z][ai]
                j = Job(wname, wl, t, z, dt_h)
                j.deadline = t + j.slots_left + horizon
                jobs.append(j); active.append(j); ai += 1
            arr_idx[z] = ai

        for j in list(active):
            v = views[j.zone]
            if v.ci[t] != v.ci[t]:               # NaN slot in this zone: hold state
                continue
            slack_slots = j.deadline - t - j.slots_left
            must_run = slack_slots <= 0
            want_run, mig_to = True, None
            if args.policy != "blind" and not must_run:
                # --- temporal: cheaper slot ahead in CURRENT zone?
                ahead = [v.ahead(t, h) for h in range(1, slack_slots + 1)]
                best_ahead = min(ahead) if ahead else v.now(t)
                gain = (v.now(t) - best_ahead) / max(v.now(t), 1e-9)
                if gain > args.theta:
                    if args.policy == "ca_costblind":
                        want_run = False
                    else:                        # marginal a* rule for a pause
                        E_slot = j.wl["P_w"] * dt_h * 3600.0
                        benefit = (v.now(t) - best_ahead) * E_slot / 3.6e6
                        exp_gap_h = max(dt_h, (slack_slots * dt_h) / 2)
                        pause_j = (j.wl["park_w"] * dt_h * 3600.0
                                   if j.wl["park_w"] * exp_gap_h * 3600.0 < j.wl["Emech_j"]
                                   else (j.wl["Emech_j"] if j.state == RUN else 0.0))
                        want_run = benefit <= pause_j * v.now(t) / 3.6e6
                # --- spatial: latency-feasible cheaper zone? (their SLO semantics: vs HOME)
                if args.latency_limit and latms:
                    cands = [z for z in zones if z != j.zone
                             and latms.get((j.home, z), 1e9) <= args.latency_limit
                             and (K[z] is math.inf or len(occ[z]) < K[z])]
                    if cands:
                        dst = min(cands, key=lambda z: views[z].now(t))
                        dci, sci = views[dst].now(t), v.now(t)
                        if (sci - dci) / max(sci, 1e-9) > args.theta:
                            E_rest = j.wl["P_w"] * j.slots_left * dt_h * 3600.0
                            benefit = (sci - dci) * E_rest / 3.6e6      # stock decision
                            mig_j = j.wl["Emech_j"] + j.wl["S_gb"] * args.wan_j_per_gb
                            cost = mig_j * (sci + dci) / 2 / 3.6e6
                            if args.policy == "ca_costblind" or benefit > cost:
                                mig_to = dst
            # --- apply migration
            if mig_to is not None:
                mig_j = j.wl["Emech_j"] + j.wl["S_gb"] * args.wan_j_per_gb
                j.over_g += mig_j * (views[j.zone].now(t) + views[mig_to].now(t)) / 2 / 3.6e6
                j.lat_s += j.wl["rt_lat_s"] + j.wl["S_gb"] / max(wan_bw, 1e-6)
                occ[j.zone].discard(j.id)
                j.zone = mig_to; j.migrations += 1; j.transitions += 1
                v = views[j.zone]
            # --- capacity gate (deadline-forced runs bypass: emergency overflow)
            if want_run and not must_run and j.id not in occ[j.zone] \
                    and K[j.zone] is not math.inf and len(occ[j.zone]) >= K[j.zone]:
                want_run = False
            # --- apply temporal transition + billing
            if want_run and j.state != RUN:
                j.state = RUN; j.transitions += 1
                occ[j.zone].add(j.id)
            elif not want_run and j.state == RUN:
                exp_gap_h = max(dt_h, (max(slack_slots, 1) * dt_h) / 2)
                if j.wl["park_w"] * exp_gap_h * 3600.0 < j.wl["Emech_j"]:
                    j.state = PARK
                    j.over_g += j.wl["park_w"] * dt_h * 3600.0 * v.now(t) / 3.6e6
                else:
                    j.state = SUSP
                    j.over_g += j.wl["Emech_j"] * v.now(t) / 3.6e6
                    j.lat_s += j.wl["rt_lat_s"]
                    occ[j.zone].discard(j.id)
                j.transitions += 1
            elif j.state == PARK:
                j.over_g += j.wl["park_w"] * dt_h * 3600.0 * v.now(t) / 3.6e6
            # --- advance if running (truth CI)
            if j.state == RUN:
                j.exec_g += j.wl["P_w"] * dt_h * 3600.0 * cimap[j.zone][t] / 3.6e6
                j.slots_left -= 1
                if j.slots_left == 0:
                    j.state = DONE
                    occ[j.zone].discard(j.id)
                    active.remove(j)

    done = [j for j in jobs if j.state == DONE]
    for j in done:                               # baseline: run-now at HOME zone truth CI
        k = max(1, round(j.wl["C_h"] / dt_h))
        vals = [cimap[j.home][j.arrive + x] for x in range(k)]
        j.base_g = sum(vv for vv in vals if vv == vv) * j.wl["P_w"] * dt_h * 3600.0 / 3.6e6
    per = defaultdict(lambda: defaultdict(float))
    for j in done:
        d = per[j.w]
        d["base"] += j.base_g; d["exec"] += j.exec_g; d["over"] += j.over_g
        d["tr"] += j.transitions; d["mig"] += j.migrations; d["lat"] += j.lat_s; d["n"] += 1
    kdesc = "inf" if not args.utilization else f"peak/{args.utilization}"
    print(f"# sim_dc zones={len(zones)} g={args.gran} policy={args.policy} theta={args.theta} "
          f"slack={args.slack_h} fc={args.forecast} U={args.utilization or 'inf'}({kdesc}) "
          f"L={args.latency_limit or '-'}ms done={len(done)}")
    print(f"{'wl':14} {'gross%':>7} {'net%':>7} {'tr/job':>7} {'mig/job':>8} {'lat_s':>7}")
    rows = []
    for w, d in sorted(per.items()):
        gross = 100 * (d["base"] - d["exec"]) / d["base"]
        net = gross - 100 * d["over"] / d["base"]
        rows.append(dict(zones=len(zones), wl=w, policy=args.policy, theta=args.theta,
                         gran=args.gran, slack_h=args.slack_h, forecast=args.forecast,
                         utilization=args.utilization or 0, latency_ms=args.latency_limit or 0,
                         gross_pct=round(gross, 3), net_pct=round(net, 3),
                         tr_per_job=round(d["tr"] / d["n"], 2), mig_per_job=round(d["mig"] / d["n"], 2),
                         lat_s_per_job=round(d["lat"] / d["n"], 1), n=int(d["n"])))
        print(f"{w:14} {gross:7.2f} {net:7.2f} {d['tr']/d['n']:7.2f} {d['mig']/d['n']:8.2f} "
              f"{d['lat']/d['n']:7.1f}")
    if args.out and rows:
        new = not os.path.exists(args.out)
        with open(args.out, "a", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=rows[0].keys())
            if new:
                wtr.writeheader()
            wtr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carbon", choices=["artifact", "tree"], default="artifact")
    ap.add_argument("--artifact", default=None)
    ap.add_argument("--zones", default="DE", help="comma zones, or 'spatial34' (latency-matrix set)")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--gran", type=int, default=60)
    ap.add_argument("--jobs", default=",".join(WORKLOADS))
    ap.add_argument("--policy", choices=["blind", "ca_costblind", "ca_costaware"],
                    default="ca_costblind")
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--slack-h", type=float, default=24)
    ap.add_argument("--forecast", default="oracle")
    ap.add_argument("--utilization", type=float, default=0.0, help="0=infinite; K=ceil(blind_peak/U)")
    ap.add_argument("--latency-limit", type=float, default=0.0, help="ms; 0 = temporal-only")
    ap.add_argument("--wan-j-per-gb", type=float, default=3600.0, help="WAN energy (band 300-6000)")
    ap.add_argument("--wan-bw-gbps", type=float, default=1.0, help="migration transfer rate GB/s")
    ap.add_argument("--arrive-every-h", type=float, default=1.0)
    ap.add_argument("--trace", default=None,
                    help="execution-trace CSV (arrive_h,duration_h,ngpu,mem_gb[,power_w]); "
                         "jobs get N1 closed-form costs; replaces the synthetic catalog")
    ap.add_argument("--trace-limit", type=int, default=0, help="cap trace rows (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    simulate(args)


if __name__ == "__main__":
    main()
