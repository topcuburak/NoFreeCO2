#!/usr/bin/env python3
"""A1 -- FSDP state-dict checkpoint cost (the REAL training temporal mechanism).

In-place cuda-checkpoint of FSDP fails to RESUME (FSDP holds a stale process-group
handle after destroy/reinit -> "NCCL communicator was aborted"; see fsdp_train.py). So
real training jobs suspend/resume via a state-dict checkpoint to disk + restart. This
measures that path: FSDP-train a few steps, then save the sharded {model, optimizer}
state to disk with torch.distributed.checkpoint (DCP) and load it back, timing each and
reporting the on-disk size.

    torchrun --nproc_per_node=4 workloads/a1_fsdp/fsdp_checkpoint.py \
        --model meta-llama/Llama-3.1-8B --batch 1 --seq-len 512 --warmup 4 \
        --ckpt-dir /var/data/a1_ckpt

Reports per-rank save/load latency + total checkpoint size. Energy derives from the
storage coefficients (results/storage_size_sweep.md). Set HF_TOKEN. NVMe vs SATA: point
--ckpt-dir at /var/data vs /home/test.
"""
from __future__ import annotations

import argparse
import functools
import os
import shutil
import time


def log(rank, msg):
    print(f"[A1-ckpt rank{rank}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=4, help="train steps before checkpoint")
    ap.add_argument("--ckpt-dir", default="/var/data/a1_ckpt")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (get_state_dict, set_state_dict)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); dev = torch.device(f"cuda:{local}")
    dist.init_process_group("nccl")
    if rank == 0:                                     # keep the (user-owned) dir; clear contents
        os.makedirs(args.ckpt_dir, exist_ok=True)
        for name in os.listdir(args.ckpt_dir):
            p = os.path.join(args.ckpt_dir, name)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    dist.barrier()

    log(rank, f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    model = FSDP(model, auto_wrap_policy=wrap, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
                 device_id=local)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    vocab = model.config.vocab_size

    for step in range(args.warmup):                              # build real optimizer state
        ids = torch.randint(0, vocab, (args.batch, args.seq_len), device=dev)
        out = model(input_ids=ids, labels=ids); out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if rank == 0:
            log(0, f"warmup step {step} loss {out.loss.item():.3f} "
                   f"gpu_alloc {torch.cuda.memory_allocated(dev)/1e9:.1f} GB")
    dist.barrier()

    # ===== SAVE (suspend): sharded {model, optimizer} -> disk =====
    msd, osd = get_state_dict(model, opt)
    state = {"model": msd, "optim": osd}
    torch.cuda.synchronize(dev); dist.barrier(); t0 = time.time()
    dcp.save(state, checkpoint_id=args.ckpt_dir)
    dist.barrier(); save_s = time.time() - t0
    size_gb = 0.0
    if rank == 0:
        for root, _, files in os.walk(args.ckpt_dir):
            for f in files:
                try: size_gb += os.path.getsize(os.path.join(root, f))
                except OSError: pass
        size_gb /= 1e9
        log(0, f"SAVE (state-dict -> {args.ckpt_dir}): {save_s:.2f}s, on-disk {size_gb:.1f} GB "
               f"-> {size_gb/save_s:.2f} GB/s")

    # ===== LOAD (restore): disk -> sharded state =====
    dist.barrier(); t0 = time.time()
    dcp.load(state, checkpoint_id=args.ckpt_dir)
    set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    dist.barrier(); load_s = time.time() - t0
    if rank == 0:
        log(0, f"LOAD (state-dict <- disk): {load_s:.2f}s -> {size_gb/load_s:.2f} GB/s")
        log(0, f"=== A1 state-dict checkpoint: save {save_s:.2f}s / load {load_s:.2f}s / "
               f"{size_gb:.1f} GB on {args.ckpt_dir} ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
