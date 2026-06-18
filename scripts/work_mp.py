#!/usr/bin/env python3
"""A7 -- multi-PROCESS CPU workload at a configurable TOTAL DRAM footprint, for criu sweeps.

A parent forks P-1 children; every process (parent + children) holds gb/P resident and loops a
read+write pass over it. The footprint is spread across the tree, so the criu driver sums RSS
over the whole group for readiness and dumps the tree root (criu -t <root> dumps all descendants).
Every process sets comm 'a7mp' so the driver (--comm a7mp) matches the tree, not python / sudo.
This is the separate-address-space (process-tree) criu target, the counterpart to A8's threads.

    python scripts/work_mp.py --gb 64 --procs 16 --seconds 100000
"""
from __future__ import annotations

import argparse
import ctypes
import os
import time


def _setcomm(name: str) -> None:
    try:
        ctypes.CDLL("libc.so.6").prctl(15, name.encode(), 0, 0, 0)   # PR_SET_NAME
    except Exception:
        pass


def hold(gb: float, seconds: int) -> None:
    import numpy as np
    n = int(gb * 1e9 / 8)
    arr = np.ones(n, dtype=np.float64)               # gb GB resident, touched on init
    stop = time.time() + seconds
    while time.time() < stop:
        arr += 1.0                                   # read+write pass (single-threaded per process)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=64.0, help="TOTAL resident DRAM across all procs (GB)")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seconds", type=int, default=100000)
    a = ap.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # keep each proc single-threaded (pure MP)
    _setcomm("a7mp")

    P = max(1, a.procs)
    per = a.gb / P
    print(f"[a7-mp] PID={os.getpid()} {a.gb} GB total over {P} procs ({per:.2f} GB each)", flush=True)
    for _ in range(P - 1):
        if os.fork() == 0:                           # child
            _setcomm("a7mp")
            hold(per, a.seconds)
            os._exit(0)
    hold(per, a.seconds)                             # parent holds its share + is the tree root


if __name__ == "__main__":
    main()
