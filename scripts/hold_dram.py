#!/usr/bin/env python3
"""CPU/DRAM-resident validation target: allocate N GiB in host DRAM, make it RESIDENT
(touch every page), and hold. The CPU analogue of hold_gpu.py, for measuring the criu
checkpoint (DRAM->disk) cost vs footprint without standing up a full simulator.

No io_uring / GPU / sockets -> a plain anonymous-memory process that criu can dump
(unlike vLLM, whose io_uring fd blocks criu). Use it to probe the host/DRAM-domain
mechanism cost, the analogue of the GPU/HBM sweep.

    python scripts/hold_dram.py --gb 8 &
    # note the printed PID, then criu dump / restore it (run as root)

A real simulator (gem5, GPGPU-Sim, etc.) gives the same mechanism cost at the same RSS;
this just makes the footprint exactly controllable for the sweep.
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="hold N GiB of resident DRAM and sleep")
    ap.add_argument("--gb", type=float, default=8.0, help="GiB of resident anonymous DRAM")
    ap.add_argument("--seconds", type=int, default=100000)
    args = ap.parse_args()

    nbytes = int(args.gb * (1024 ** 3))
    # Force the pages RESIDENT (RSS), not lazily-zeroed COW, so criu actually dumps them.
    try:
        import numpy as np                              # fast C-level fill
        buf = np.ones(nbytes, dtype=np.uint8)           # noqa: F841  (keep alive)
        resident = buf.nbytes
    except Exception:
        buf = bytearray(nbytes)                         # fallback: touch every 4 KB page
        for i in range(0, nbytes, 4096):
            buf[i] = 1
        resident = len(buf)

    print(f"[hold_dram] PID={os.getpid()} holding {resident/1e9:.2f} GB resident DRAM "
          f"for {args.seconds}s", flush=True)
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
