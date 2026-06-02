#!/usr/bin/env python3
"""A4 -- DLRMv2 training (MLPerf, reduced embedding budget). State ~20-30 GB.

Standalone runner: drives DLRMv2 training to a checkpointable state dominated by
the embedding tables (~20-30 GB) -- the "memory-bound GPU, big sparse state" point.
The dump path here is DRAM<->NIC or PCIe rather than pure HBM. CLI-configurable.

    torchrun --nproc_per_node=4 train.py \
        --embedding-budget-gb 24 --batch-size 4096 \
        --max-steps 300 --checkpoint-dir /mnt/md0/a4ckpt

Runs on ford (needs GPUs + Criteo data or synthetic). torch/torchrec inside main().
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A4 DLRMv2 training runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="criteo_1tb",
                   help="criteo_1tb | criteo_kaggle | synthetic")
    p.add_argument("--data-dir", default="/mnt/md0/criteo")
    p.add_argument("--embedding-budget-gb", type=float, default=24.0,
                   help="reduced embedding-table budget (sets the dominant state size)")
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4096, help="global batch")
    p.add_argument("--lr", type=float, default=15.0)
    p.add_argument("--sharding", choices=["table_wise", "row_wise", "column_wise"],
                   default="row_wise", help="TorchRec embedding sharding")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--steady-step", type=int, default=80,
                   help="reach this step, then hold for dump/restore measurement")
    p.add_argument("--checkpoint-dir", default="/mnt/md0/a4ckpt")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    # import torch, torchrec here
    # 1. init DDP (torchrun); build DLRMv2 (dense MLP + EmbeddingBagCollection)
    # 2. size embeddings to args.embedding_budget_gb; shard with args.sharding
    # 3. stream args.dataset; train loop to args.steady_step, then HOLD
    # 4. expose embedding+dense state for the dump callable (state ~20-30 GB)
    raise NotImplementedError(
        "A4 DLRMv2 train loop: implement on ford with TorchRec + the configured args")


if __name__ == "__main__":
    main()
