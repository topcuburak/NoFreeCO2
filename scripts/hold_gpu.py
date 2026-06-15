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
    ap.add_argument("--chunks", type=int, default=1,
                    help="split the footprint into this many separate allocations. "
                         "cuda-checkpoint has per-allocation overhead, so this probes "
                         "how allocation STRUCTURE (not just size) drives the checkpoint "
                         "cost -- a real workload (e.g. vLLM's many KV tensors) has many "
                         "allocations, unlike one big tensor (--chunks 1).")
    args = ap.parse_args()

    import torch  # lazy: only needed on the GPU box
    dev = torch.device(f"cuda:{args.gpu}")
    n_total = int(args.gb * (1024 ** 3) // 2)  # fp16 elements (2 bytes each)
    k = max(1, args.chunks)
    per = max(1, n_total // k)
    xs = [torch.empty(per, dtype=torch.float16, device=dev).fill_(1.0) for _ in range(k)]
    torch.cuda.synchronize(dev)
    print(f"[hold_gpu] PID={os.getpid()} holding {args.gb} GiB on cuda:{args.gpu} "
          f"in {k} allocation(s) for {args.seconds}s", flush=True)
    try:
        time.sleep(args.seconds)
    finally:
        del xs


if __name__ == "__main__":
    main()
