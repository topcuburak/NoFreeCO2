#!/usr/bin/env python3
"""A3 -- ViT-Large pretraining on ImageNet-1K. State ~3-5 GB.

Standalone runner: drives ViT-Large training to a checkpointable state (model +
optimizer ~= 3-5 GB), the "small-state, compute-bound GPU" point in the workload
set. CLI-configurable.

    torchrun --nproc_per_node=4 pretrain.py \
        --data-dir /mnt/md0/imagenet --batch-size 256 --precision bf16 \
        --max-steps 500 --checkpoint-dir /mnt/md0/a3ckpt

Runs on ford (needs GPUs + ImageNet-1K). torch/timm imported inside main().
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A3 ViT-Large ImageNet pretraining runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="vit_large_patch16_224", help="timm model name")
    p.add_argument("--data-dir", default="/mnt/md0/imagenet",
                   help="ImageNet-1K root (train/ val/ ImageFolder layout)")
    p.add_argument("--batch-size", type=int, default=256, help="per-GPU batch")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--steady-step", type=int, default=100,
                   help="reach this step, then hold for dump/restore measurement")
    p.add_argument("--checkpoint-dir", default="/mnt/md0/a3ckpt")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    # import torch, timm, torchvision here
    # 1. init DDP (torchrun); build timm.create_model(args.model)
    # 2. ImageFolder(args.data-dir/train) + DataLoader(args.batch_size, args.num_workers)
    # 3. AdamW + autocast(args.precision); train loop to args.steady_step, then HOLD
    # 4. expose model+optimizer state_dict for the dump callable (state ~3-5 GB)
    raise NotImplementedError(
        "A3 ViT pretrain loop: implement on ford with timm + the configured args above")


if __name__ == "__main__":
    main()
