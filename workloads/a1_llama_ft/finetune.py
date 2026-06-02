#!/usr/bin/env python3
"""A1 -- Llama-3-8B fine-tuning (FSDP across 4 GPUs). State ~70 GB.

Standalone runner: drives a fine-tuning loop to a mid-epoch *checkpointable* state
(sharded params + optimizer + grads ~= 70 GB), the state the dump/restore harness
measures. Compute-bound on GPU. All knobs are CLI-configurable.

    torchrun --nproc_per_node=4 finetune.py \
        --model meta-llama/Meta-Llama-3-8B \
        --dataset tatsu-lab/alpaca --seq-len 2048 --batch-size 4 \
        --max-steps 200 --checkpoint-dir /mnt/md0/a1ckpt

Runs on ford (needs 4x A100 + HF_TOKEN for gated weights). torch/transformers
are imported inside main() so the file loads off-testbed.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A1 Llama-3-8B FSDP fine-tuning runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--dataset", default="tatsu-lab/alpaca")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4, help="per-GPU micro-batch")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--sharding", choices=["full", "hybrid", "grad_op"], default="full",
                   help="FSDP sharding strategy (FULL_SHARD / HYBRID / SHARD_GRAD_OP)")
    p.add_argument("--checkpoint-dir", default="/mnt/md0/a1ckpt")
    p.add_argument("--steady-step", type=int, default=50,
                   help="reach this step, then hold for dump/restore measurement")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    # import torch, transformers, torch.distributed.fsdp here
    # 1. init process group (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK)
    # 2. load model + tokenizer; wrap in FSDP(sharding=args.sharding, mp=args.precision)
    # 3. build AdamW optimizer (this is what makes state ~70 GB: params+grads+moments)
    # 4. stream args.dataset, run the train loop to args.steady_step, then HOLD
    # 5. expose sharded state via FSDP.state_dict(...) for the dump callable
    raise NotImplementedError(
        "A1 finetune loop: implement on ford with FSDP + the configured args above")


if __name__ == "__main__":
    main()
