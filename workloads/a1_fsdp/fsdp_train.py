#!/usr/bin/env python3
"""A1 -- Llama-3-8B FSDP fine-tuning (4 GPUs), real-world temporal-suspend test.

Runs genuine FSDP training (model + AdamW optimizer sharded across GPUs), then at a
step boundary performs the app-aware multi-GPU suspend we validated on nccl_holdtest:

    quiesce (barrier) -> dist.destroy_process_group()  [release NCCL/IPC]
      -> [external: cuda-checkpoint suspend/store/resume the 4 ranks]
      -> init_process_group(fresh TCPStore) -> CONTINUE training (validates correctness)

This measures the temporal mechanism cost on a real FSDP training footprint (~100+ GB
across 4 GPUs) and tests whether training resumes after an in-place cuda-checkpoint.

    torchrun --nproc_per_node=4 workloads/a1_fsdp/fsdp_train.py \
        --model meta-llama/Llama-3.1-8B --batch 1 --seq-len 512 --steps 12 --suspend-step 6

Then (root), once all ranks print 'PG DESTROYED':
    sudo -E $(which python) scripts/timed_dump_experiment.py --multiproc --skip-criu \
        --store --store-out /var/data --marks-min 0 --hold-seconds 8
    touch /tmp/a1_resume
Gated model: set HF_TOKEN. Needs host RAM >= total GPU footprint (cuda-checkpoint stages HBM in DRAM).
"""
from __future__ import annotations

import argparse
import datetime
import functools
import glob
import os
import time


def log(rank, msg):
    print(f"[A1 rank{rank}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--suspend-step", type=int, default=6, help="step after which to suspend (-1 = never)")
    ap.add_argument("--resume-flag", default="/tmp/a1_resume")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    rank = int(os.environ["RANK"]); local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    base_port = int(os.environ.get("MASTER_PORT", "29500"))
    if rank == 0:
        for f in [args.resume_flag] + glob.glob("/tmp/a1_destroyed.*"):
            try: os.remove(f)
            except OSError: pass

    torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}")
    dist.init_process_group("nccl")

    log(rank, f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    model = FSDP(model, auto_wrap_policy=wrap,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
                 device_id=local)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    vocab = model.config.vocab_size if hasattr(model, "config") else 128256

    def batch():
        ids = torch.randint(0, vocab, (args.batch, args.seq_len), device=dev)
        return ids

    def train_step(step):
        ids = batch()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        if rank == 0:
            used = torch.cuda.memory_allocated(dev) / 1e9
            log(0, f"step {step:3d}  loss {out.loss.item():.3f}  gpu_alloc {used:.1f} GB")

    for step in range(args.suspend_step if args.suspend_step >= 0 else args.steps):
        train_step(step)

    if 0 <= args.suspend_step < args.steps:
        dist.barrier()                                       # quiesce -- no collective in flight
        log(rank, "reached suspend point; destroying process group (releasing NCCL)...")
        dist.destroy_process_group()
        torch.cuda.synchronize(dev)
        open(f"/tmp/a1_destroyed.{rank}", "w").close()
        log(rank, f"PG DESTROYED -- PID={os.getpid()} holding FSDP state on cuda:{local}, "
                  f"ready for cuda-checkpoint. (touch {args.resume_flag} to reinit+continue)")
        while not os.path.exists(args.resume_flag):
            time.sleep(1)
        t0 = time.time()
        store = dist.TCPStore(master_addr, base_port + 1, world, is_master=(rank == 0),
                              timeout=datetime.timedelta(seconds=180))
        dist.init_process_group("nccl", store=store, rank=rank, world_size=world)
        dist.barrier()
        log(rank, f"PG RE-INIT (fresh store) in {time.time()-t0:.2f}s -- continuing training")
        for step in range(args.suspend_step, args.steps):   # CONTINUE -- validates correctness
            train_step(step)

    dist.barrier()
    log(rank, "done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
