#!/usr/bin/env python3
"""Configurable-footprint ACTIVE DRAM workload for criu testing.

Allocates --gb GiB and continuously operates on ALL of it (a full-array increment each
iteration), so the memory stays resident, dirty, and hot -- criu then checkpoints a LIVE
compute process, not an idle one (the realistic case, the CPU analogue of the GPU workloads).

The running counter `arr[0]` (+1 per iteration) is a CORRECTNESS CHECK: after a criu restore
it must keep counting up from the snapshot value -- proving criu preserved the live in-memory
state. Each iteration also reads+writes the whole array, so the reported mem_bw shows it's
genuinely memory-bound (not paged out).

    python scripts/work_dram.py --gb 8 --report-every 5
    # then (root) criu dump/restore its PID -- arr[0] keeps counting across the cycle.
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="hold N GiB resident DRAM and actively work on it")
    ap.add_argument("--gb", type=float, default=8.0, help="GiB of resident DRAM to allocate + work on")
    ap.add_argument("--seconds", type=int, default=100000)
    ap.add_argument("--report-every", type=float, default=5.0, help="seconds between progress prints")
    args = ap.parse_args()

    import numpy as np
    n = max(1, int(args.gb * (1024 ** 3) // 8))             # float64 elements
    arr = np.ones(n, dtype=np.float64)                      # resident from the start
    print(f"[work_dram] PID={os.getpid()} allocated {arr.nbytes/1e9:.2f} GB float64; "
          f"working (full-array increment/iter)...", flush=True)

    it = 0
    t0 = time.time(); last = t0; touched = 0
    try:
        while time.time() - t0 < args.seconds:
            arr += 1.0                                      # full read+write pass over all GB
            it += 1
            touched += arr.nbytes
            now = time.time()
            if now - last >= args.report_every:
                bw = touched / (now - t0) / 1e9             # effective r+w memory bandwidth
                print(f"[work_dram] iter={it} arr[0]={arr[0]:.0f} mem_bw={bw:.1f} GB/s "
                      f"resident={arr.nbytes/1e9:.1f}GB", flush=True)
                last = now
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
