#!/usr/bin/env python3
"""sim_dc: discrete-event datacenter simulator for carbon-aware scheduling under
MEASURED mechanism cost (N1). The unified engine behind N2: an actual online
scheduler (not an oracle bound analysis), with pluggable realism layers.

Layers (all composable):
  CI view    : --forecast oracle | noise:<MAPE%>   (file-based CarbonCast later)
  Capacity   : --utilization U  -> K = ceil(steady_jobs / U) slots per zone; park
               OCCUPIES a slot, suspended (dumped) FREES it. U=0 -> infinite.
  Granularity: --gran minutes (60 hourly; real 5/15/30-min zones via carbon tree)
  Policy     : --policy blind | ca_costblind | ca_costaware
      blind        run immediately (carbon-unaware FIFO)
      ca_costblind at each epoch: suspend if a cheaper slot exists later in the
                   window (theta-gated), resume in the cheapest slots -- the
                   EuroSys'24 ideal made online; pays measured costs implicitly
      ca_costaware same, but acts ONLY if predicted CI benefit of the remaining
                   work exceeds the mechanism cost of the transition pair (N1
                   predictor in the loop) -- the paper's design guideline
Costs charged at event-time CI: suspend+resume pair E_mech (measured per workload),
park hold power, dump frees resources. Completion-time inflation tracked from
measured leg latencies. Bridge test: --utilization 0 --forecast oracle --policy
ca_costblind reproduces sim_n2/artifact-bound behavior (same slot preferences).

  python scripts/sim_dc.py --carbon artifact --artifact ~/dp --zone DE \
      --policy ca_costaware --theta 0.05 --slack-h 24 --utilization 0.9
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

# measured catalog (N1): compute h, active W, footprint GB, suspend+resume RT J,
# RT latency s, park W. arrival_weight ~ mix share (uniform default).
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


# ---------------- carbon + forecast ----------------
def load_ci(args) -> list[float]:
    if args.carbon == "artifact":
        import pandas as pd
        df = pd.read_csv(os.path.join(os.path.expanduser(args.artifact),
                                      "shared_data", "combined_carbon.csv"))
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df[df["datetime"].dt.year == args.year]
        assert args.gran == 60, "artifact carbon is hourly"
        return df[args.zone].tolist()
    sub = {5: "real_5min", 15: "real_15min", 30: "real_30min", 60: "real_60min"}
    pats = ([f"{CARBON}/{sub[args.gran]}/**/{args.zone}_{args.year}_*.csv"] if args.gran in sub else []) \
        + [f"{CARBON}/synthetic_{args.gran}min/**/{args.zone}_{args.year}_{args.gran}min.csv"]
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            out = []
            with open(hits[0], newline="") as f:
                r = csv.reader(f); next(r)
                for row in r:
                    out.append(float(row[4]) if row[4] else float("nan"))
            return out
    raise SystemExit(f"[sim_dc] no CI for {args.zone} {args.year} g={args.gran}")


class CiView:
    """What the scheduler SEES. oracle = truth; noise:<MAPE%> = truth x lognormal-ish
    error growing with horizon (fresh draw per (epoch,horizon): no free averaging)."""
    def __init__(self, ci, mode, seed=7):
        self.ci = ci; self.rng = random.Random(seed)
        self.mape = None
        if mode.startswith("noise:"):
            self.mape = float(mode.split(":")[1]) / 100.0
    def now(self, t):
        return self.ci[t]
    def ahead(self, t, h):                       # forecast CI at t+h, seen from t
        v = self.ci[min(t + h, len(self.ci) - 1)]
        if self.mape is None or h == 0:
            return v
        sd = self.mape * math.sqrt(min(h, 48) / 24.0)
        return v * max(0.05, 1.0 + self.rng.gauss(0.0, sd * 1.2533))   # E|err| ~ mape*sqrt(h/24)


# ---------------- engine ----------------
class Job:
    _n = 0
    def __init__(self, wname, wl, t, dt_h):
        Job._n += 1; self.id = Job._n
        self.w = wname; self.wl = wl
        self.arrive = t
        self.slots_left = max(1, round(wl["C_h"] / dt_h))
        self.deadline = None                     # set by engine (slack)
        self.state = SUSP                        # queued: holds NO resources, costs nothing
                                                 # (fresh job: no dump was billed, start is free)
        self.exec_g = 0.0; self.over_g = 0.0; self.base_g = None
        self.transitions = 0; self.lat_s = 0.0


def simulate(args):
    ci = load_ci(args)
    view = CiView(ci, args.forecast, args.seed)
    dt_h = args.gran / 60.0
    horizon = round(args.slack_h / dt_h)         # lookahead / deadline slack (slots)
    wls = {w: WORKLOADS[w] for w in args.jobs.split(",")}
    rng = random.Random(args.seed)
    arrive_every = max(1, round(args.arrive_every_h / dt_h))

    # capacity: steady-state running jobs ~ sum(C_h)/arrive_every_h; K = ceil(that / U)
    steady = sum(w["C_h"] for w in wls.values()) / (args.arrive_every_h * len(wls))
    K = math.inf if not args.utilization else max(1, math.ceil(steady * len(wls) / args.utilization))

    jobs, active = [], []
    for t in range(len(ci)):
        if any(ci[s] != ci[s] for s in [t]):     # NaN slot: skip accounting epoch
            continue
        # arrivals (round-robin over catalog)
        if t % arrive_every == 0 and t + horizon + 40 < len(ci):
            wname = list(wls)[ (t // arrive_every) % len(wls) ]
            j = Job(wname, wls[wname], t, dt_h)
            j.deadline = t + j.slots_left + horizon
            j.base_g = None
            jobs.append(j); active.append(j)

        occupied = [j for j in active if j.state in (RUN, PARK)]   # PARK holds its slot
        # ---- policy: decide state for each active job ----
        for j in list(active):
            slack_slots = j.deadline - t - j.slots_left
            must_run = slack_slots <= 0
            want_run = True
            if args.policy != "blind" and not must_run:
                # cheaper slot ahead within remaining window?
                ahead = [view.ahead(t, h) for h in range(1, slack_slots + 1)]
                best_ahead = min(ahead) if ahead else view.now(t)
                gain = (view.now(t) - best_ahead) / max(view.now(t), 1e-9)
                if gain > args.theta:
                    if args.policy == "ca_costblind":
                        want_run = False
                    else:                        # ca_costaware: N1 predictor in the loop
                        E_rest = j.wl["P_w"] * j.slots_left * dt_h * 3600.0
                        benefit_g = (view.now(t) - best_ahead) * E_rest / 3.6e6
                        pause_j = min(j.wl["park_w"] * dt_h * 3600.0, j.wl["Emech_j"])
                        cost_g = pause_j * view.now(t) / 3.6e6
                        want_run = benefit_g <= cost_g
            # capacity gate: RUN/PARK need a slot, SUSP does not
            if want_run and j not in occupied and len(occupied) >= K:
                want_run = False                 # blocked by capacity (the green rush)
            # apply transition + billing
            if want_run and j.state != RUN:
                j.state = RUN; j.transitions += 1   # resume: pair cost billed at dump time
                if j not in occupied:
                    occupied.append(j)
            elif not want_run and j.state == RUN:
                # pause: park (keep slot) vs dump (free slot), break-even on expected gap
                exp_gap_h = max(dt_h, (slack_slots * dt_h) / 2)
                if j.wl["park_w"] * exp_gap_h * 3600.0 < j.wl["Emech_j"]:
                    j.state = PARK               # keeps its slot
                    j.over_g += j.wl["park_w"] * dt_h * 3600.0 * view.now(t) / 3.6e6
                else:
                    j.state = SUSP               # dump+resume pair billed now, slot freed
                    j.over_g += j.wl["Emech_j"] * view.now(t) / 3.6e6
                    j.lat_s += j.wl["rt_lat_s"]
                    occupied.remove(j)
                j.transitions += 1
            elif j.state == PARK:
                j.over_g += j.wl["park_w"] * dt_h * 3600.0 * view.now(t) / 3.6e6
            running = [x for x in active if x.state == RUN]
        # ---- advance running jobs (truth CI for accounting) ----
        for j in running:
            j.exec_g += j.wl["P_w"] * dt_h * 3600.0 * ci[t] / 3.6e6
            j.slots_left -= 1
            if j.slots_left == 0:
                j.state = DONE; active.remove(j)

    done = [j for j in jobs if j.state == DONE]
    # baseline: run-now contiguous at truth CI
    for j in done:
        k = max(1, round(j.wl["C_h"] / dt_h))
        j.base_g = sum(ci[j.arrive + x] * j.wl["P_w"] * dt_h * 3600.0
                       for x in range(k) if ci[j.arrive + x] == ci[j.arrive + x]) / 3.6e6
    per = defaultdict(lambda: defaultdict(float))
    for j in done:
        d = per[j.w]
        d["base"] += j.base_g; d["exec"] += j.exec_g; d["over"] += j.over_g
        d["tr"] += j.transitions; d["lat"] += j.lat_s; d["n"] += 1
    print(f"# sim_dc zone={args.zone} g={args.gran} policy={args.policy} theta={args.theta} "
          f"slack={args.slack_h}h forecast={args.forecast} U={args.utilization or 'inf'} "
          f"K={K} jobs_done={len(done)}")
    print(f"{'wl':14} {'gross%':>7} {'net%':>7} {'tr/job':>7} {'lat_s/job':>9}")
    rows = []
    for w, d in sorted(per.items()):
        gross = 100 * (d["base"] - d["exec"]) / d["base"]
        net = gross - 100 * d["over"] / d["base"]
        rows.append(dict(zone=args.zone, wl=w, policy=args.policy, theta=args.theta,
                         gran=args.gran, slack_h=args.slack_h, forecast=args.forecast,
                         utilization=args.utilization or 0, gross_pct=round(gross, 3),
                         net_pct=round(net, 3), tr_per_job=round(d["tr"] / d["n"], 2),
                         lat_s_per_job=round(d["lat"] / d["n"], 1), n=int(d["n"])))
        print(f"{w:14} {gross:7.2f} {net:7.2f} {d['tr']/d['n']:7.2f} {d['lat']/d['n']:9.1f}")
    if args.out:
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
    ap.add_argument("--zone", default="DE")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--gran", type=int, default=60)
    ap.add_argument("--jobs", default=",".join(WORKLOADS))
    ap.add_argument("--policy", choices=["blind", "ca_costblind", "ca_costaware"],
                    default="ca_costblind")
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--slack-h", type=float, default=24)
    ap.add_argument("--forecast", default="oracle", help="oracle | noise:<MAPE%%>")
    ap.add_argument("--utilization", type=float, default=0.0, help="0 = infinite capacity")
    ap.add_argument("--arrive-every-h", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    simulate(args)


if __name__ == "__main__":
    main()
