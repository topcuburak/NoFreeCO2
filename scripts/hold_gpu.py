#!/usr/bin/env python3
"""Trivial validation target for transparent_dump.py: allocate N GiB on a GPU and
hold (sleep), printing the PID. Lets you exercise the cuda-checkpoint + criu +
telemetry loop on a single, simple process before tackling vLLM / TP=4 / NCCL.

    python scripts/hold_gpu.py --gb 20 --gpu 0 &
    # note the printed PID, then run transparent_dump.py against it
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser(description="hold N GiB of HBM and sleep")
    ap.add_argument("--gb", type=float, default=10.0, help="GiB to allocate on the GPU")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=3600)
    args = ap.parse_args()

    import torch  # lazy: only needed on the GPU box
    dev = torch.device(f"cuda:{args.gpu}")
    n = int(args.gb * (1024 ** 3) // 2)        # fp16 elements (2 bytes each)
    x = torch.empty(n, dtype=torch.float16, device=dev)
    x.fill_(1.0)
    torch.cuda.synchronize(dev)
    print(f"[hold_gpu] PID={os.getpid()} holding {args.gb} GiB on cuda:{args.gpu} "
          f"for {args.seconds}s", flush=True)
    try:
        time.sleep(args.seconds)
    finally:
        del x


if __name__ == "__main__":
    main()
