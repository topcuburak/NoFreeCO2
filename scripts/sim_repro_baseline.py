#!/usr/bin/env python3
"""Reproduce the EuroSys'24 zero-overhead baselines by running the ARTIFACT'S OWN task()
(imported verbatim from umassos/decarbonization-potential) and summarizing savings.

Step 1 of the N2 pipeline: validates environment + their numbers before we inject the
measured mechanism cost (sim_n2.py). Runs anywhere with pandas; artifact path required.

  python scripts/sim_repro_baseline.py --artifact ~/decarbonization-potential \
      --zones DE,FR,GB,PL,US-CAL-CISO,US-MIDA-PJM --jobs 1,6,24 --slacks 24,168
Full-scale (all 123 zones) is cheap per (zone,job,slack): ~8760 iterations each.
Outputs a savings table: defer% (best contiguous start) and interrupt% (k-lowest slots),
both vs run-now, matching their postprocessing definition.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="path to decarbonization-potential clone")
    ap.add_argument("--zones", default="DE,FR,GB,PL,US-CAL-CISO,US-MIDA-PJM,SE,IN-WE",
                    help="comma zone codes, or 'all'")
    ap.add_argument("--jobs", default="1,6,24", help="job lengths (hours)")
    ap.add_argument("--slacks", default="24", help="absolute slack hours beyond job length")
    ap.add_argument("--out", default=None, help="csv output (default: print only)")
    args = ap.parse_args()

    import pandas as pd
    art = os.path.expanduser(args.artifact)
    os.chdir(os.path.join(art, "sim_temporal"))            # their relative paths are cwd-based
    sys.path.insert(0, ".")
    m = importlib.import_module("experiment_vary_job_slack")   # __main__ guard: no sweep runs
    zones = m.zone_code_list if args.zones == "all" else \
        [z for z in args.zones.split(",") if z in m.zone_code_list]
    jobs = [int(x) for x in args.jobs.split(",")]
    slacks = [int(x) for x in args.slacks.split(",")]
    print(f"[repro] artifact year={m.year} zones={len(zones)} jobs={jobs} slacks={slacks}")

    out = tempfile.mkdtemp()
    rows = []
    for slack in slacks:
        for job in jobs:
            for z in zones:
                m.task(z, job, slack, out)                 # THEIR code, verbatim
                df = pd.read_csv(f"{out}/{z}.csv")
                s0 = df.slack_0.sum()
                rows.append(dict(zone=z, job=job, slack=slack,
                                 defer_pct=100 * (s0 - df.non_interrupt.sum()) / s0,
                                 interrupt_pct=100 * (s0 - df.interrupt.sum()) / s0,
                                 n_arrivals=len(df)))
                print(f"  {z:14} job={job:3} slack={slack:4} "
                      f"defer={rows[-1]['defer_pct']:6.2f}% interrupt={rows[-1]['interrupt_pct']:6.2f}%")
    r = pd.DataFrame(rows)
    print("\n[repro] mean savings by (job, slack):")
    print(r.groupby(["job", "slack"])[["defer_pct", "interrupt_pct"]].mean().round(2).to_string())
    if args.out:
        r.to_csv(args.out, index=False); print(f"[repro] -> {args.out}")


if __name__ == "__main__":
    main()
