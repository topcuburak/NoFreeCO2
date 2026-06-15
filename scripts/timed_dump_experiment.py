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
from microbench.isolation import dd_write, dd_read          # noqa: E402
from _common import build_telemetry, write_record, print_record  # noqa: E402


def drop_caches() -> None:
    os.system("sync")
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except OSError:
        pass

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


def pid_gpu_indices(pids) -> list[int] | None:
    """NVML indices of the GPU(s) the given PIDs run on -- to scope power to just
    the GPU(s) involved (clean attribution for TP=1)."""
    if not _HAVE_NVML:
        return None
    want = {int(p) for p in pids}
    idx = set()
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if int(p.pid) in want:
                    idx.add(i)
    except Exception:
        return None
    return sorted(idx) or None


# ---- MODELED energy for the two domains ford has no hardware power telemetry for ----
# (GPU board incl. HBM is measured via NVML; CPU package via RAPL. DRAM DIMMs have no RAPL
# domain on this EPYC; the NVMe/SATA drive has only a byte counter.) Kept SEPARATE from the
# measured "TOTAL absolute" -- never folded into it -- and reported as a FULL total.
#  DRAM:  w_dram_per_gb * resident_GB * t_leg. The 150 GB host staging buffer is resident and
#         accessed every leg. 0.3 W/GB ~ active server DDR4 (refresh ~0.1 + access). Range 0.1-0.4.
#  DRIVE: P_drive * t_leg over the DISK legs only (store/load). From storage_size_sweep.md:
#         NVMe RAID-0 = 50 W (25 W x2), SATA = 3 W active. (suspend/resume touch no disk -> 0.)
DRAM_W_PER_GB_DEFAULT = 0.3
DRIVE_W_DEFAULT = 50.0          # NVMe RAID-0; pass --drive-w 3 for SATA


def model_aux_energy(phase, footprint_bytes, latency_s, dram_w_per_gb, drive_w):
    """MODELED (dram_j, drive_j) for one leg. DRAM on every leg (staging buffer resident/
    accessed); drive only on the disk legs."""
    s_gb = (footprint_bytes or 0) / 1e9
    dram_j = dram_w_per_gb * s_gb * latency_s
    drive_j = drive_w * latency_s if phase in ("store", "load") else 0.0
    return dram_j, drive_j


def dump_and_resume(tele, cc_bin, criu_bin, pids, out_dir, mark_min, baseline, keep_images,
                    multiproc=False, criu_root=None, hold_seconds=0.0, skip_criu=False,
                    store=False, store_out=None, tag=None,
                    dram_w_per_gb=DRAM_W_PER_GB_DEFAULT, drive_w=DRIVE_W_DEFAULT):
    """One full cycle, measured records tagged by mark:
    suspend (HBM->host) -> [store (host->NVMe)] -> [hold] -> [load (NVMe->host)] -> resume.

    skip_criu: skip the criu image write (blocked by io_uring on stock criu).
    store: add a footprint-sized O_DIRECT write (store) + cold read (load) to/from
    store_out as a device-rate proxy for the disk persist/restore leg.
    """
    gpu_before = td.gpu_used_bytes()
    mark_dir = os.path.join(out_dir, f"mark_{mark_min}min")
    out = {"suspend": None, "criu": None, "store": None, "load": None, "resume": None}

    def _emit(rec):                                  # finalize + write IMMEDIATELY (survive failures)
        rec.extra["mark_min"] = mark_min
        if tag:
            rec.config["tag"] = tag
        phase = rec.config.get("phase", "")
        dram_j, drive_j = model_aux_energy(phase, gpu_before, rec.latency_s, dram_w_per_gb, drive_w)
        meas_abs = sum(s.energy_abs_j or 0.0 for s in rec.sources)   # GPU+CPU measured
        rec.extra.update(measured_abs_j=round(meas_abs, 1), dram_model_j=round(dram_j, 1),
                         drive_model_j=round(drive_j, 1), full_total_j=round(meas_abs + dram_j + drive_j, 1),
                         dram_w_per_gb=dram_w_per_gb, drive_w=drive_w)
        print_record(rec)
        print(f"  modeled: DRAM {dram_j:.0f} J + drive {drive_j:.0f} J  |  "
              f"FULL (meas GPU+CPU + modeled DRAM+drive): {meas_abs + dram_j + drive_j:.0f} J", flush=True)
        write_record(rec, "timed_dump")

    # phase 1: suspend (HBM -> host). If THIS fails (e.g. cuda-checkpoint can't checkpoint
    # a live IPC handle), the suspend may be PARTIAL -> best-effort recover to running, then
    # re-raise so the caller records nothing and (GPU intact) can safely continue training.
    try:
        rec_s = measure_operation(
            tele, workload="timed_dump", operation="cuda_checkpoint",
            state_bytes=gpu_before, baseline_seconds=baseline,
            op=lambda: td.cuda_suspend(cc_bin, pids, multiproc),
            config={"mark_min": mark_min, "domain": "accelerator", "phase": "suspend",
                    "multiproc": multiproc})
    except Exception as e:
        print(f"[timed] mark {mark_min}: SUSPEND failed ({type(e).__name__}: {e}); "
              f"recovering processes to running", flush=True)
        td.cuda_recover(cc_bin, pids)
        raise
    gpu_after = td.gpu_used_bytes()
    rec_s.extra["gpu_freed_bytes"] = gpu_before - gpu_after
    out["suspend"] = rec_s
    _emit(rec_s)

    # Once suspended, the GPU is EVICTED. Guarantee cuda_resume runs no matter what happens
    # in store/load/hold (finally) -- else training would reinit onto freed memory and corrupt.
    try:
        # phase 2: persist (host -> NVMe) -- optional (criu blocked by io_uring on ford)
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
            out["criu"] = rec_c
            _emit(rec_c)

        # phase 2b: STORE (host DRAM -> NVMe) -- footprint-sized O_DIRECT proxy write
        spath = None
        if store:
            os.makedirs(store_out, exist_ok=True)
            spath = os.path.join(store_out, f"socc_store_{mark_min}min.bin")
            rec_store = measure_operation(
                tele, workload="timed_dump", operation="store",
                state_bytes=gpu_before, baseline_seconds=baseline,
                op=lambda: dd_write(spath, gpu_before),
                config={"mark_min": mark_min, "domain": "host", "phase": "store", "out": store_out})
            out["store"] = rec_store
            _emit(rec_store)

        # optional hold: observe checkpointed state; capture idle-holding power E_idle.
        if hold_seconds > 0:
            print(f"[timed] HOLDING checkpointed/suspended for {hold_seconds:.0f}s", flush=True)
            gpu_idle = td.gpu_used_bytes()
            rec_h = measure_operation(
                tele, workload="timed_dump", operation="idle_hold",
                state_bytes=gpu_idle, baseline_seconds=min(baseline, 2.0),
                op=lambda: time.sleep(hold_seconds),
                config={"mark_min": mark_min, "domain": "host", "phase": "hold",
                        "hold_seconds": hold_seconds})
            _emit(rec_h)

        # phase 3b: LOAD (NVMe -> host DRAM) -- cold read back before resume
        if store:
            drop_caches()
            rec_load = measure_operation(
                tele, workload="timed_dump", operation="load",
                state_bytes=gpu_before, baseline_seconds=baseline,
                op=lambda: dd_read(spath, gpu_before),
                config={"mark_min": mark_min, "domain": "host", "phase": "load"})
            try: os.remove(spath)
            except OSError: pass
            out["load"] = rec_load
            _emit(rec_load)
    except Exception as e:
        print(f"[timed] mark {mark_min}: store/load FAILED ({type(e).__name__}: {e}); "
              f"restoring GPU anyway", flush=True)
    finally:
        # clear page cache so the store/load proxy's 150 GB doesn't keep the cuda-checkpoint
        # host staging buffer paged out -> otherwise resume faults it all back in (the 84 s outlier)
        if store:
            drop_caches()
        # phase 3: resume (host -> HBM) -- ALWAYS, so the GPU is never left evicted
        rec_r = measure_operation(
            tele, workload="timed_dump", operation="cuda_restore",
            state_bytes=gpu_before, baseline_seconds=baseline,
            op=lambda: td.cuda_resume(cc_bin, pids, multiproc),
            config={"mark_min": mark_min, "domain": "accelerator", "phase": "resume",
                    "multiproc": multiproc})
        out["resume"] = rec_r
        _emit(rec_r)

    footprint_gb = rec_s.extra.get("gpu_freed_bytes", 0) / 1e9
    store_s = out["store"].latency_s if out["store"] else (out["criu"].latency_s if out["criu"] else 0.0)
    load_s = out["load"].latency_s if out["load"] else 0.0
    print(f"[timed] mark {mark_min}min: FOOTPRINT {footprint_gb:.1f} GB | "
          f"suspend {rec_s.latency_s:.2f}s + store {store_s:.2f}s + load {load_s:.2f}s + "
          f"resume {out['resume'].latency_s:.2f}s")
    return out


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
    ap.add_argument("--store", action="store_true",
                    help="add store(host->NVMe)+load(NVMe->host) of the footprint between "
                         "suspend and resume (device-rate O_DIRECT proxy for the disk persist)")
    ap.add_argument("--store-out", default="/var/data",
                    help="dir for the store proxy file (put on the tier you want, e.g. /var/data)")
    ap.add_argument("--tag", default=None,
                    help="label written into each record's config (e.g. a1_fsdp_nvme) to separate "
                         "this run from other timed_dump rows in data/timed_dump.jsonl")
    ap.add_argument("--dram-w-per-gb", type=float, default=DRAM_W_PER_GB_DEFAULT,
                    help="MODELED DRAM power per GB resident (no DRAM RAPL on this EPYC)")
    ap.add_argument("--drive-w", type=float, default=DRIVE_W_DEFAULT,
                    help="MODELED drive active power for the store/load legs (NVMe 50, SATA 3)")
    ap.add_argument("--verbose-cc", action="store_true",
                    help="log cuda-checkpoint state transitions (adds get-state subprocesses "
                         "INSIDE the timed op -> only for debugging, not clean measurement)")
    args = ap.parse_args()
    td.VERBOSE = args.verbose_cc

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

    tele = build_telemetry(nvml_gpus=pid_gpu_indices(pids))   # scope GPU power to vLLM's GPU(s)
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
                                hold_seconds=args.hold_seconds, skip_criu=args.skip_criu,
                                store=args.store, store_out=args.store_out, tag=args.tag,
                                dram_w_per_gb=args.dram_w_per_gb, drive_w=args.drive_w)
            except Exception as e:
                print(f"[timed] mark {m}min FAILED: {type(e).__name__}: {e}")
                print(f"[timed] (if criu errored, paste it -- likely needs extra flags)")
        print(f"\n[timed] done. {len(marks)} marks -> data/timed_dump.jsonl")
    finally:
        tele.stop()


if __name__ == "__main__":
    main()
