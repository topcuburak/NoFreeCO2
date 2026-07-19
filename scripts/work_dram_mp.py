#!/usr/bin/env python3
"""Synthetic multi-process / multi-thread resident-DRAM workload for the CPU criu
process/thread-overhead sweep. Total --gb is split across --procs processes; each
process runs --threads threads that continuously increment their slice so memory stays
resident, dirty and hot (like work_dram.py, but with a controllable process/thread
structure). criu then checkpoints the whole fork tree -> lets us isolate the per-process
and per-thread overhead of the mechanism at fixed footprint.

Plain anonymous memory + fork + threads only (no io_uring / GPU / sockets) so criu dumps
it cleanly. All tasks set comm "wdmp" so the sweep driver matches the tree, not python/sudo.

    python scripts/work_dram_mp.py --gb 32 --procs 8 --threads 1 --seconds 100000
Used as sweep_criu_dump.py --target work_dram_mp.py --comm wdmp \
    --target-extra "--procs 8 --threads 1"
"""
from __future__ import annotations

import argparse
import ctypes
import os
import threading
import time

COMM = "wdmp"


def set_comm(name=COMM):
    try:
        ctypes.CDLL("libc.so.6").prctl(15, name.encode()[:15], 0, 0, 0)   # PR_SET_NAME
    except Exception:
        pass


def churn(gb, seconds, nthreads, mode="hot"):
    """Allocate gb GiB float64 resident, and run nthreads threads for `seconds`.
    mode=hot : threads continuously increment their slice (dirty+active, worst-case freeze).
    mode=idle: threads sleep (array stays resident+held but NOT churned) -> isolates the
               cost of criu freezing IDLE vs ACTIVELY-RUNNING threads at checkpoint."""
    import numpy as np
    n = max(1, int(gb * (1024 ** 3) // 8))
    arr = np.ones(n, dtype=np.float64)                    # resident from the start (held below)
    deadline = time.time() + seconds
    if mode == "idle":                                    # threads exist but block -> fast to freeze
        def sleeper():
            while time.time() < deadline:
                time.sleep(0.5)
        ts = [threading.Thread(target=sleeper, daemon=True) for _ in range(max(1, nthreads))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert arr[0] >= 1.0                              # keep arr referenced (resident) to the end
        return
    if nthreads <= 1:
        while time.time() < deadline:
            arr += 1.0                                    # full read+write pass
        return
    bounds = [(i * n // nthreads, (i + 1) * n // nthreads) for i in range(nthreads)]

    def work(lo, hi):
        while time.time() < deadline:
            arr[lo:hi] += 1.0
    ts = [threading.Thread(target=work, args=b, daemon=True) for b in bounds]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=8.0, help="TOTAL GiB across all processes")
    ap.add_argument("--procs", type=int, default=1, help="number of processes (fork tree)")
    ap.add_argument("--threads", type=int, default=1, help="threads per process")
    ap.add_argument("--thread-mode", default="hot", choices=["hot", "idle"],
                    help="hot=threads churn (dirty+active); idle=threads sleep (resident, blocked)")
    ap.add_argument("--seconds", type=int, default=100000)
    a = ap.parse_args()
    per = a.gb / a.procs                                   # per-process footprint
    set_comm()
    for _ in range(a.procs - 1):                          # fork procs-1 children; parent is the procs-th
        if os.fork() == 0:
            set_comm()
            churn(per, a.seconds, a.threads, a.thread_mode)
            os._exit(0)
    print(f"[wdmp] PID={os.getpid()} procs={a.procs} threads={a.threads} mode={a.thread_mode} "
          f"total={a.gb:.1f}GiB per_proc={per:.2f}GiB", flush=True)
    churn(per, a.seconds, a.threads, a.thread_mode)       # parent holds its own share too


if __name__ == "__main__":
    main()
