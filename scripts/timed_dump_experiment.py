#!/usr/bin/env python3
"""Timed transparent-dump experiment: while a vLLM serving run is live, perform a
suspend -> dump -> resume at fixed wall-clock marks (e.g. 10,20,30,40,50 min) and
measure the cost of each, tagged by mark.

Per mark, the cycle (serving continues between marks):
  1. cuda-checkpoint --toggle   running -> checkpointed   (HBM->host)   [accelerator]
  2. criu dump --leave-running  process image -> NVMe                   [host]
  3. cuda-checkpoint --toggle   checkpointed -> running   (host->HBM)   [accelerator]

Records 3 measured RunRecords per mark -> data/timed_dump.jsonl.

ATTACH MODEL (clean privilege split):
  terminal 1 (you):   launch serving as yourself, single checkpointable process:
      VLLM_ENABLE_V1_MULTIPROCESSING=0 python workloads/a2_vllm/serve.py \
          --model meta-llama/Llama-3.1-8B --tensor-parallel-size 1 \
          --dataset data/lbv2_40k_90k.jsonl --prompt-field prompt --repeat 12 ...
  terminal 2 (root):  once it's at steady state, start the clock:
      sudo -E $(which python) scripts/timed_dump_experiment.py \
          --marks-min 10,20,30,40,50 --out /mnt/md0/tdump

PIDs to checkpoint are auto-detected from NVML (processes using the GPU); override
with --pids. Marks are relative to THIS controller's start (t0). Images are deleted
after measuring unless --keep-images (resume uses cuda-checkpoint, not the image).

TP=4 is the hard case (NCCL + multiple procs); prove TP=1 first, then model TP=4.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)                                   # for transparent_dump
sys.path.insert(0, _REPO)                                   # for harness/_common

import transparent_dump as td                               # noqa: E402  (reuse ops)
from harness import measure_operation                       # noqa: E402
from _common import build_telemetry, write_record, print_record  # noqa: E402

try:
    import pynvml                                            # noqa: E402
    _HAVE_NVML = True
except Exception:
    _HAVE_NVML = False


def gpu_compute_pids() -> list[int]:
    """PIDs currently holding a CUDA context on any GPU (the ones to checkpoint)."""
    if not _HAVE_NVML:
        return []
    pids: set[int] = set()
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                pids.add(int(p.pid))
    except Exception:
        pass
    return sorted(pids)


def dump_and_resume(tele, cc_bin, criu_bin, pids, out_dir, mark_min, baseline, keep_images,
                    multiproc=False, criu_root=None, hold_seconds=0.0, skip_criu=False):
    """One suspend->[dump]->resume cycle, measured records tagged by mark.

    skip_criu: skip the host criu_dump phase (it's blocked by io_uring on stock
    criu) and measure only the cuda-checkpoint footprint + resume. The store cost
    then comes from the storage characterization, not a live criu image.
    """
    gpu_before = td.gpu_used_bytes()
    mark_dir = os.path.join(out_dir, f"mark_{mark_min}min")

    # phase 1: suspend (HBM -> host), pauses inference -- the footprint cuda-checkpoint dumps
    rec_s = measure_operation(
        tele, workload="timed_dump", operation="cuda_checkpoint",
        state_bytes=gpu_before, baseline_seconds=baseline,
        op=lambda: td.cuda_suspend(cc_bin, pids, multiproc),
        config={"mark_min": mark_min, "domain": "accelerator", "phase": "suspend",
                "multiproc": multiproc})
    gpu_after = td.gpu_used_bytes()
    rec_s.extra["gpu_freed_bytes"] = gpu_before - gpu_after

    # phase 2: persist (host -> NVMe) -- optional (criu blocked by io_uring on ford)
    rec_c = None
    if not skip_criu:
        rss = td.rss_bytes(pids)
        rec_c = measure_operation(
            tele, workload="timed_dump", operation="criu_dump",
            state_bytes=rss, baseline_seconds=baseline,
            op=lambda: td.op_criu_dump(criu_bin, pids, mark_dir, leave_running=True,
                                       criu_root=criu_root),
            config={"mark_min": mark_min, "domain": "host"})
        rec_c.extra["image_bytes"] = int(rec_c.extra.get("op_result") or 0)
        if not keep_images:
            shutil.rmtree(mark_dir, ignore_errors=True)

    # optional hold: observe the checkpointed state (nvidia-smi shows GPU freed;
    # image at mark_dir if --keep-images) and capture idle-holding power E_idle.
    if hold_seconds > 0:
        img_note = mark_dir if keep_images else "(image deleted; use --keep-images to inspect)"
        print(f"[timed] HOLDING checkpointed/suspended for {hold_seconds:.0f}s -- "
              f"observe `nvidia-smi` (GPU freed) and {img_note}")
        gpu_idle = td.gpu_used_bytes()
        rec_h = measure_operation(
            tele, workload="timed_dump", operation="idle_hold",
            state_bytes=gpu_idle, baseline_seconds=min(baseline, 2.0),
            op=lambda: time.sleep(hold_seconds),
            config={"mark_min": mark_min, "domain": "host", "phase": "hold",
                    "hold_seconds": hold_seconds})
        rec_h.extra["mark_min"] = mark_min
        write_record(rec_h, "timed_dump")

    # phase 3: resume (host -> HBM), resumes inference
    rec_r = measure_operation(
        tele, workload="timed_dump", operation="cuda_restore",
        state_bytes=gpu_before, baseline_seconds=baseline,
        op=lambda: td.cuda_resume(cc_bin, pids, multiproc),
        config={"mark_min": mark_min, "domain": "accelerator", "phase": "resume",
                "multiproc": multiproc})

    recs = [rec_s] + ([rec_c] if rec_c else []) + [rec_r]
    for r in recs:
        r.extra["mark_min"] = mark_min
        print_record(r)
        write_record(r, "timed_dump")
    footprint_gb = rec_s.extra.get("gpu_freed_bytes", 0) / 1e9
    dump_s = rec_c.latency_s if rec_c else 0.0
    paused = rec_s.latency_s + dump_s + rec_r.latency_s
    print(f"[timed] mark {mark_min}min: FOOTPRINT {footprint_gb:.1f} GB | "
          f"suspend {rec_s.latency_s:.2f}s + dump {dump_s:.2f}s + resume {rec_r.latency_s:.2f}s "
          f"| serving paused ~{paused:.1f}s")
    return rec_s, rec_c, rec_r


def main() -> None:
    ap = argparse.ArgumentParser(description="timed suspend/dump/resume during a live serving run")
    ap.add_argument("--marks-min", default="10,20,30,40,50",
                    help="comma-separated minute marks (relative to this start)")
    ap.add_argument("--out", default=os.path.join(_REPO, "dumps"),
                    help="CRIU image dir (default: <repo>/dumps; gitignored)")
    ap.add_argument("--pids", default=None, help="override GPU PIDs (else NVML auto-detect)")
    ap.add_argument("--baseline", type=float, default=5.0, help="telemetry baseline sec per phase")
    ap.add_argument("--keep-images", action="store_true", help="don't delete CRIU images")
    ap.add_argument("--criu-bin", default=None)
    ap.add_argument("--cc-bin", default=None)
    ap.add_argument("--hold-seconds", type=float, default=0.0,
                    help="hold the checkpointed/suspended state this long before resume "
                         "(observe nvidia-smi/dump; measures idle-holding power)")
    ap.add_argument("--multiproc", action="store_true",
                    help="TP>1: lock-all then checkpoint-all (and restore-all/unlock-all)")
    ap.add_argument("--criu-root", type=int, default=None,
                    help="TP>1: root PID for criu tree dump (parent of the workers)")
    ap.add_argument("--skip-criu", action="store_true",
                    help="skip the host criu_dump phase (blocked by io_uring on ford); "
                         "measure cuda-checkpoint footprint + resume only")
    args = ap.parse_args()

    pf = td.preflight(args)
    print(f"[timed] cuda-checkpoint={pf['cuda_checkpoint']} criu={pf['criu']} euid={pf['euid']}")
    if pf["problems"]:
        for p in pf["problems"]:
            print(f"[timed] BLOCKER: {p}")
        raise SystemExit(1)

    marks = sorted(int(m) for m in args.marks_min.split(",") if m.strip())
    pids = ([int(p) for p in args.pids.split(",")] if args.pids else gpu_compute_pids())
    if not pids:
        raise SystemExit("[timed] no GPU PIDs found -- is serving running and holding the GPU?")
    print(f"[timed] target GPU PIDs: {pids}")
    print(f"[timed] marks (min): {marks}")

    tele = build_telemetry()
    tele.start()
    t0 = time.monotonic()
    try:
        for m in marks:
            target = t0 + m * 60
            while time.monotonic() < target:
                time.sleep(min(5.0, target - time.monotonic()))
            # refresh PIDs in case they changed (auto-detect mode)
            cur = pids if args.pids else (gpu_compute_pids() or pids)
            print(f"\n[timed] === mark {m}min (t+{time.monotonic()-t0:.0f}s) pids={cur} ===")
            try:
                dump_and_resume(tele, pf["cuda_checkpoint"], pf["criu"], cur,
                                args.out, m, args.baseline, args.keep_images,
                                multiproc=args.multiproc, criu_root=args.criu_root,
                                hold_seconds=args.hold_seconds, skip_criu=args.skip_criu)
            except Exception as e:
                print(f"[timed] mark {m}min FAILED: {type(e).__name__}: {e}")
                print(f"[timed] (if criu errored, paste it -- likely needs extra flags)")
        print(f"\n[timed] done. {len(marks)} marks -> data/timed_dump.jsonl")
    finally:
        tele.stop()


if __name__ == "__main__":
    main()
