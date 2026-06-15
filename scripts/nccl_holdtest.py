#!/usr/bin/env python3
"""Minimal multi-GPU NCCL target to test the
    pause -> DESTROY NCCL -> cuda-checkpoint suspend/resume -> RE-INIT NCCL
sequence. Question: does destroying the process group make a multi-GPU process
checkpointable by cuda-checkpoint? (The live-NCCL case fails with CUDA_ERROR_OPERATING_SYSTEM.)

Each rank: init NCCL -> prove all-reduce works -> destroy_process_group -> hold a
footprint and poll a resume flag (CPU-only, survives cuda-checkpoint) -> on the flag,
init_process_group again and TIME the re-init.

Run (4 GPUs):
    torchrun --nproc_per_node=4 scripts/nccl_holdtest.py --gb 8

Once all ranks print 'NCCL DESTROYED', in another (root) terminal cuda-checkpoint them:
    sudo -E $(which python) scripts/timed_dump_experiment.py --multiproc --skip-criu --marks-min 0
Then re-init NCCL on the workers and read the timing:
    touch /tmp/nccl_resume
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=8.0, help="GiB footprint per GPU (to checkpoint)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--resume-flag", default="/tmp/nccl_resume")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    if rank == 0:                                   # clean stale flags
        for f in [args.resume_flag] + glob.glob("/tmp/nccl_destroyed.*"):
            try:
                os.remove(f)
            except OSError:
                pass

    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    dist.init_process_group("nccl")

    # footprint to checkpoint + a tensor to all-reduce
    n = int(args.gb * (1024 ** 3) // 2)
    hold = torch.ones(n, dtype=torch.float16, device=dev)   # noqa: F841 (keep resident)
    x = torch.ones(1 << 22, dtype=torch.float32, device=dev)

    for _ in range(args.iters):                     # prove NCCL works
        dist.all_reduce(x)
        torch.cuda.synchronize(dev)
    dist.barrier()
    if rank == 0:
        print(f"[rank0] NCCL OK: {args.iters} all-reduces succeeded. "
              f"Destroying the process group on all ranks...", flush=True)

    # ===== DESTROY NCCL (releases communicators / CUDA-IPC / proxy threads) =====
    dist.destroy_process_group()
    torch.cuda.synchronize(dev)
    open(f"/tmp/nccl_destroyed.{rank}", "w").close()
    print(f"[rank{rank} PID={os.getpid()}] NCCL DESTROYED -- holding {args.gb} GiB on cuda:{local}, "
          f"ready for cuda-checkpoint. (touch {args.resume_flag} to re-init)", flush=True)

    # wait for resume -- a pure file poll, no CUDA, so it survives cuda-checkpoint lock/suspend
    while not os.path.exists(args.resume_flag):
        time.sleep(1)

    # ===== RE-INIT NCCL and time it =====
    t0 = time.time()
    dist.init_process_group("nccl")
    for _ in range(5):
        dist.all_reduce(x)
        torch.cuda.synchronize(dev)
    dist.barrier()
    dt = time.time() - t0
    print(f"[rank{rank} PID={os.getpid()}] NCCL RE-INIT + all-reduce OK in {dt:.2f}s "
          f"(post-checkpoint)", flush=True)
    time.sleep(120)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
