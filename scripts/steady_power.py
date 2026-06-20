#!/usr/bin/env python3
"""Steady-state RUNNING power of a workload -- the denominator for the mechanism-overhead ratio.

Attach to an ALREADY-RUNNING workload at steady state and integrate full power over a window
(GPU board via NVML + CPU pkg via RAPL + modeled DRAM), using the SAME methodology as the
dump/restore energy so the two are directly comparable. Reports average watts per component and
total, plus energy over the window. Mechanism overhead = E_dump+E_restore (measured) expressed as
seconds-of-running = E_mech / P_run.

    # GPU workload (e.g. A1 on 4 GPUs): launch the workload, let it reach steady state, then:
    sudo -E python scripts/steady_power.py --seconds 60 --gpus 0,1,2,3 --dram-gb 200 --tag a1_run
    # CPU workload (criu targets, no GPU):
    sudo -E python scripts/steady_power.py --seconds 60 --gpus none --dram-gb 71 --tag a5_run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from harness import measure_operation                            # noqa: E402
from _common import build_telemetry, write_record               # noqa: E402

DRAM_W_PER_GB = 0.3


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0, help="integration window")
    ap.add_argument("--gpus", default="all", help="'all', 'none', or comma list e.g. 0,1,2,3")
    ap.add_argument("--dram-gb", type=float, default=0.0, help="host DRAM resident (GB) for the model")
    ap.add_argument("--dram-w-per-gb", type=float, default=DRAM_W_PER_GB)
    ap.add_argument("--tag", default="run_power")
    args = ap.parse_args()

    if args.gpus == "none":
        gpus = []
    elif args.gpus == "all":
        gpus = None                                              # let build_telemetry pick all
    else:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    tele = build_telemetry(nvml_gpus=gpus) if gpus is not None else build_telemetry()
    tele.start()
    print(f"[run-power] sampling {args.seconds:.0f}s (gpus={args.gpus}, dram={args.dram_gb}GB)...", flush=True)
    rec = measure_operation(tele, workload="steady_power", operation="idle_sample",
                            state_bytes=0, baseline_seconds=1.0,
                            op=lambda: time.sleep(args.seconds), config={"phase": "run"})
    tele.stop()

    sec = rec.latency_s
    gpu_j = sum(s.energy_abs_j or 0.0 for s in rec.sources if "nvml" in s.name)
    cpu_j = sum(s.energy_abs_j or 0.0 for s in rec.sources if "rapl" in s.name)
    dram_w = args.dram_w_per_gb * args.dram_gb
    dram_j = dram_w * sec
    tot_j = gpu_j + cpu_j + dram_j
    gpu_w, cpu_w, tot_w = gpu_j / sec, cpu_j / sec, tot_j / sec

    print(f"\n[run-power] {args.tag}: {sec:.1f}s window", flush=True)
    print(f"  GPU board (NVML)  {gpu_w:7.1f} W   ({gpu_j/1000:.2f} kJ)")
    print(f"  CPU pkg   (RAPL)  {cpu_w:7.1f} W   ({cpu_j/1000:.2f} kJ)")
    print(f"  DRAM (modeled)    {dram_w:7.1f} W   ({dram_j/1000:.2f} kJ)")
    print(f"  TOTAL running     {tot_w:7.1f} W   ({tot_j/1000:.2f} kJ over {sec:.0f}s)")

    rec.extra.update(gpu_w=round(gpu_w, 1), cpu_w=round(cpu_w, 1), dram_w=round(dram_w, 1),
                     total_w=round(tot_w, 1), window_s=round(sec, 1), dram_gb=args.dram_gb)
    rec.config.update(tag=args.tag, workload="steady_power")
    write_record(rec, "steady_power")
    print(f"\n[run-power] -> data/steady_power.jsonl tag {args.tag}", flush=True)


if __name__ == "__main__":
    main()
