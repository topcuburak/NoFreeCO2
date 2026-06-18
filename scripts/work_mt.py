#!/usr/bin/env python3
"""A8 -- multi-THREAD CPU workload at a configurable DRAM footprint, for criu sweeps.

One process, T threads sharing a single large resident array; each thread loops a read+write
pass over its own slice. numpy ufuncs release the GIL on big arrays, so the threads run on
distinct cores -- a genuinely multi-core, single-address-space criu target (the thread-state
case, the counterpart to A7's process tree). Sets its comm to 'a8mt' so the criu driver
(--comm a8mt) locks onto this process and never the python driver / sudo.

    python scripts/work_mt.py --gb 64 --threads 64 --seconds 100000
"""
from __future__ import annotations

import argparse
import ctypes
import os
import threading
import time


def _setcomm(name: str) -> None:
    try:
        ctypes.CDLL("libc.so.6").prctl(15, name.encode(), 0, 0, 0)   # PR_SET_NAME
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=64.0, help="total resident DRAM footprint (GB)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seconds", type=int, default=100000)
    a = ap.parse_args()
    _setcomm("a8mt")

    import numpy as np
    n = int(a.gb * 1e9 / 8)                          # float64 elements -> gb GB
    arr = np.ones(n, dtype=np.float64)               # touch-on-init: full RSS immediately
    T = max(1, a.threads)
    chunk = (n + T - 1) // T
    print(f"[a8-mt] PID={os.getpid()} {a.gb} GB, {T} threads", flush=True)

    stop = time.time() + a.seconds

    def worker(i: int) -> None:
        lo = i * chunk
        hi = min(n, lo + chunk)
        while time.time() < stop:
            arr[lo:hi] += 1.0                        # read+write pass (GIL released in the ufunc)

    ths = [threading.Thread(target=worker, args=(i,)) for i in range(T)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


if __name__ == "__main__":
    main()
