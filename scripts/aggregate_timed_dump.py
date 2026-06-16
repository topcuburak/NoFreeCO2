#!/usr/bin/env python3
"""Aggregate a tagged multi-cycle timed_dump run into per-leg mean ± std.

Reads data/timed_dump.jsonl, filters to one --tag (e.g. a1_nvme_10cyc), groups by
suspend/store/load/resume, and reports latency, GB/s, measured (GPU+CPU) energy, the
MODELED DRAM/drive terms, and the FULL total, plus the round-trip and a cost-drift dump.

    python scripts/aggregate_timed_dump.py --tag a1_nvme_10cyc
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st

PHASES = ["suspend", "store", "load", "resume"]


def ms(x):
    return (st.mean(x), st.pstdev(x)) if x else (0.0, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--path", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "timed_dump.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.path) if args.tag in l]
    rows = [r for r in rows if r.get("config", {}).get("tag") == args.tag]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["config"]["mark_min"]][r["config"]["phase"]] = r
    marks = sorted(by)
    lat = lambda r: r["t_end_mono"] - r["t_start_mono"]
    fp = [by[m]["suspend"]["extra"].get("gpu_freed_bytes", 0) / 1e9 for m in marks if "suspend" in by[m]]
    foot = ms(fp)[0]
    print(f"tag {args.tag}: {len(marks)} rounds {marks}")
    print(f"footprint {foot:.1f} ± {ms(fp)[1]:.2f} GB\n")

    print(f"{'leg':8} {'n':>2} {'latency_s':>15} {'GB/s':>6} {'meas_kJ':>14} "
          f"{'DRAM':>5} {'drive':>6} {'FULL_kJ':>14}")
    print("-" * 92)
    for ph in PHASES:
        rs = [by[m][ph] for m in marks if ph in by[m]]
        if not rs:
            continue
        la = ms([lat(r) for r in rs])
        rate = foot / la[0] if la[0] else 0.0
        meas = ms([r["extra"].get("measured_abs_j", 0) / 1000 for r in rs])
        dram = ms([r["extra"].get("dram_model_j", 0) / 1000 for r in rs])
        drive = ms([r["extra"].get("drive_model_j", 0) / 1000 for r in rs])
        full = ms([r["extra"].get("full_total_j", 0) / 1000 for r in rs])
        print(f"{ph:8} {len(rs):>2} {la[0]:7.2f} ± {la[1]:4.2f} {rate:6.2f} "
              f"{meas[0]:7.2f} ± {meas[1]:4.2f} {dram[0]:5.2f} {drive[0]:6.2f} "
              f"{full[0]:7.2f} ± {full[1]:4.2f}")
    rt_lat = [sum(lat(by[m][ph]) for ph in PHASES if ph in by[m]) for m in marks]
    rt_meas = [sum(by[m][ph]["extra"].get("measured_abs_j", 0) for ph in PHASES if ph in by[m]) / 1000 for m in marks]
    rt_full = [sum(by[m][ph]["extra"].get("full_total_j", 0) for ph in PHASES if ph in by[m]) / 1000 for m in marks]
    print("-" * 92)
    print(f"{'ROUNDTRIP':8}    {ms(rt_lat)[0]:7.2f} ± {ms(rt_lat)[1]:4.2f}        "
          f"{ms(rt_meas)[0]:7.2f} ± {ms(rt_meas)[1]:4.2f}              {ms(rt_full)[0]:7.2f} ± {ms(rt_full)[1]:4.2f}")
    print(f"\nFULL energy/byte: {ms(rt_full)[0]*1000/foot:.0f} J/GB  (measured-only {ms(rt_meas)[0]*1000/foot:.0f} J/GB)")
    print("\ncost drift (round-trip FULL kJ by step):")
    for m in marks:
        print(f"  step {m:5}: {sum(by[m][ph]['extra'].get('full_total_j',0) for ph in PHASES if ph in by[m])/1000:6.2f} kJ")


if __name__ == "__main__":
    main()
