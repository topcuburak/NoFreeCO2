#!/usr/bin/env python3
"""Dump-free baseline: wall-clock RUNTIME + full-system ENERGY for ONE fixed job of a workload.

Handshake: launch the workload, wait for its 'RUNJOB_READY' line (setup/model-load/table-build
done), then start integrating GPU(NVML)+CPU(RAPL)+DRAM power, trigger the job (touch a file the
workload waits on), and stop when the workload exits. So the measured window is the JOB ONLY --
setup excluded. Same energy methodology as the dump/restore measurements, so the mechanism
overhead E_mech / E_job and T_mech / T_job are directly comparable.

    sudo -E python scripts/job_energy.py --gpus none --dram-gb 100 --tag a8_job -- \
        python scripts/work_duck.py --gb 100 --threads 64 --job-queries 50
    # binaries with no handshake (GAPBS/gem5): add --no-handshake (whole run is the job)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from harness import measure_operation                            # noqa: E402
from _common import build_telemetry, write_record               # noqa: E402

DRAM_W_PER_GB = 0.3
TRIGGER = "/tmp/runjob.go"
READY = "RUNJOB_READY"


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="all", help="'all', 'none', or comma list e.g. 0,1,2,3")
    ap.add_argument("--dram-gb", type=float, default=0.0)
    ap.add_argument("--dram-w-per-gb", type=float, default=DRAM_W_PER_GB)
    ap.add_argument("--tag", default="job")
    ap.add_argument("--ready-timeout", type=float, default=1200.0, help="max wait for setup/READY")
    ap.add_argument("--no-handshake", action="store_true",
                    help="no READY/trigger: measure the WHOLE run (setup+job), for binaries")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the workload command")
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        raise SystemExit("[job] no command given (put it after --)")
    if cmd[0] in ("python", "python3"):                          # sudo strips PATH -> resolve the
        cmd[0] = sys.executable                                  # conda interpreter explicitly

    gpus = [] if args.gpus == "none" else (None if args.gpus == "all"
            else [int(x) for x in args.gpus.split(",") if x.strip()])
    try:
        os.remove(TRIGGER)
    except OSError:
        pass

    env = dict(os.environ)
    if not args.no_handshake:                                   # in no-handshake mode we never create
        env["RUNJOB_TRIGGER"] = TRIGGER                         # the trigger -> don't make handshake-
    else:                                                       # aware scripts (serve.py) wait for it
        env.pop("RUNJOB_TRIGGER", None)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)

    ready = threading.Event()

    def pump():
        for line in proc.stdout:
            sys.stdout.write(line)
            if READY in line:
                ready.set()
    threading.Thread(target=pump, daemon=True).start()

    if not args.no_handshake:
        print(f"[job] waiting for {READY} (setup)...", flush=True)
        dl = time.monotonic() + args.ready_timeout
        while not ready.is_set() and time.monotonic() < dl and proc.poll() is None:
            time.sleep(0.5)
        if proc.poll() is not None:
            raise SystemExit("[job] workload exited before READY")
        print("[job] READY -- measuring the job", flush=True)

    tele = build_telemetry(nvml_gpus=gpus) if gpus is not None else build_telemetry()
    tele.start()

    def run_job():
        if not args.no_handshake:
            open(TRIGGER, "w").close()                          # release the workload into the job
        proc.wait()                                             # job runs until the workload exits

    rec = measure_operation(tele, workload="job_energy", operation="job",
                            state_bytes=0, baseline_seconds=1.0, op=run_job,
                            config={"phase": "job"})
    tele.stop()

    sec = rec.latency_s
    gpu_j = sum(s.energy_abs_j or 0.0 for s in rec.sources if "nvml" in s.name)
    cpu_j = sum(s.energy_abs_j or 0.0 for s in rec.sources if "rapl" in s.name)
    dram_w = args.dram_w_per_gb * args.dram_gb
    dram_j = dram_w * sec
    tot_j = gpu_j + cpu_j + dram_j

    if sec > 180.0:                                             # RAPL pkg counter wraps ~every 224s
        print(f"[job] WARNING: window {sec:.0f}s exceeds ~180s -- the RAPL counter can wrap more than "
              f"once between starved samples under full load, UNDER-counting CPU energy. Prefer "
              f"shorter jobs (<180s) and extrapolate. Treat this CPU number with suspicion.", flush=True)
    print(f"\n[job] {args.tag}: runtime {sec:.1f}s | energy GPU {gpu_j/1000:.2f} + CPU {cpu_j/1000:.2f} "
          f"+ DRAM {dram_j/1000:.2f} = {tot_j/1000:.2f} kJ  ({tot_j/sec:.0f} W avg)", flush=True)
    rec.extra.update(runtime_s=round(sec, 2), gpu_j=round(gpu_j, 1), cpu_j=round(cpu_j, 1),
                     dram_j=round(dram_j, 1), total_j=round(tot_j, 1), avg_w=round(tot_j / sec, 1),
                     dram_gb=args.dram_gb)
    rec.config.update(tag=args.tag, workload="job_energy")
    write_record(rec, "job_energy")
    print(f"[job] -> data/job_energy.jsonl tag {args.tag}", flush=True)


if __name__ == "__main__":
    main()
