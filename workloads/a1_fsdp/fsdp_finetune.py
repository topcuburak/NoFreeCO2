#!/usr/bin/env python3
"""A1 -- REAL FSDP fine-tuning (4 GPUs) with an in-place temporal suspend.

Unlike fsdp_train.py (a footprint generator on random data), this runs a genuine
instruction-tuning job: a real dataset (Alpaca by default), proper prompt masking,
AdamW + cosine LR schedule + grad accumulation. The loss actually descends. At a
chosen optimizer-step boundary it performs the validated in-place suspend:

    quiesce (barrier) -> dist.destroy_process_group()        [release NCCL/IPC]
      -> [external: cuda-checkpoint suspend/store/resume the 4 ranks]
      -> init_process_group(fresh TCPStore) -> rebind FSDP -> CONTINUE training

Because cuda-checkpoint evicts only GPU state (we --skip-criu), the Python process
stays alive across the hold, so the dataloader/step/scheduler resume seamlessly and
the loss curve is CONTINUOUS across the suspend -- the correctness proof for a real
fine-tune. The same rebind_fsdp_pg used by fsdp_train.py re-points FSDP at the fresh
process group so training continues after destroy/reinit.

    torchrun --nproc_per_node=4 workloads/a1_fsdp/fsdp_finetune.py \
        --model meta-llama/Llama-3.1-8B --dataset tatsu-lab/alpaca \
        --max-len 512 --batch 1 --grad-accum 4 --lr 2e-5 \
        --steps 200 --suspend-step 100

Then (root), once all ranks print 'PG DESTROYED':
    sudo -E $(which python) scripts/timed_dump_experiment.py --multiproc --skip-criu \
        --store --store-out /var/data --marks-min 0 --hold-seconds 8 --tag a1_finetune_nvme
    touch /tmp/a1_resume
Set HF_TOKEN for the gated Llama model. Alpaca is public (no token).
"""
from __future__ import annotations

import argparse
import datetime
import functools
import glob
import os
import time


def log(rank, msg):
    print(f"[A1-ft rank{rank}] {msg}", flush=True)


def rebind_fsdp_pg(model, new_pg, rank):
    """After destroy/reinit, re-point every cached process_group reference (module,
    _fsdp_state, flat-param handle) at the fresh default group so the next collective
    uses the live comm instead of the aborted one. Mirrors fsdp_train.py."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    n_mod = n_handle = 0
    for m in FSDP.fsdp_modules(model):
        for obj in (m, getattr(m, "_fsdp_state", None)):
            if obj is not None and hasattr(obj, "process_group"):
                obj.process_group = new_pg; n_mod += 1
        handles = []
        for attr in ("_handle", "_handles"):
            h = getattr(m, attr, None) or (getattr(m, "_fsdp_state", None)
                                           and getattr(m._fsdp_state, attr, None))
            if h is None:
                continue
            handles.extend(h if isinstance(h, (list, tuple)) else [h])
        for h in handles:
            if hasattr(h, "process_group"):
                h.process_group = new_pg; n_handle += 1
            if hasattr(h, "_process_group"):
                h._process_group = new_pg
    if rank == 0:
        log(0, f"rebind_fsdp_pg: re-pointed {n_mod} module(s) + {n_handle} handle(s)")


# ---- dataset: real instruction tuning (Alpaca-format), prompt tokens masked ----
PROMPT_INPUT = ("Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n### Response:\n")
PROMPT_NO_INPUT = ("Below is an instruction that describes a task. Write a response that "
                   "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n"
                   "### Response:\n")


def build_dataset(name, tokenizer, max_len):
    import torch
    from datasets import load_dataset
    ds = load_dataset(name, split="train")

    def encode(ex):
        instr = ex.get("instruction", "") or ""
        inp = ex.get("input", "") or ""
        out = ex.get("output", "") or ""
        prompt = (PROMPT_INPUT.format(instruction=instr, input=inp) if inp.strip()
                  else PROMPT_NO_INPUT.format(instruction=instr))
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(out + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        ids = ([tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []) + p_ids + r_ids
        labels = ([-100] * (1 + len(p_ids))) + r_ids[:]             # mask prompt (+bos)
        ids, labels = ids[:max_len], labels[:max_len]
        pad = max_len - len(ids)
        attn = [1] * len(ids) + [0] * pad
        ids = ids + [tokenizer.pad_token_id] * pad
        labels = labels + [-100] * pad
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}

    ds = ds.map(encode, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--dataset", default="tatsu-lab/alpaca")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1, help="micro-batch per GPU")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--steps", type=int, default=200, help="optimizer steps")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--suspend-step", type=int, default=100, help="single suspend step (-1 = never)")
    ap.add_argument("--suspend-steps", default=None,
                    help="comma-separated optimizer steps to suspend at, e.g. 40,80,120,160,200 "
                         "(overrides --suspend-step). Each is a full destroy/dump/reinit/rebind cycle.")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--job-steps", type=int, default=0,
                    help="dump-free baseline: warmup, rank-0 RUNJOB_READY, all ranks wait on trigger, "
                         "run N optimizer steps, exit (cross-rank handshake)")
    ap.add_argument("--resume-flag", default="/tmp/a1_resume")
    ap.add_argument("--destroyed-prefix", default="/tmp/a1_destroyed.",
                    help="per-rank held-marker prefix; the driver deletes <prefix><rank> to resume")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              get_cosine_schedule_with_warmup)
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

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if rank == 0:
        log(0, f"building dataset {args.dataset} (max_len {args.max_len}) ...")
    data = build_dataset(args.dataset, tok, args.max_len)
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(data, batch_size=args.batch, sampler=sampler, num_workers=0,
                        drop_last=True, pin_memory=True)

    log(rank, f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    model = FSDP(model, auto_wrap_policy=wrap,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16),
                 device_id=local)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.steps)

    def suspend_resume(step, port):
        # Marker-based handshake: this rank writes its OWN private marker and spins until the
        # DRIVER deletes it (driver = sole remover -> no shared-flag TOCTOU race). Manual fallback:
        # touching --resume-flag also releases. The driver must NOT proceed until the dump is done.
        marker = f"{args.destroyed_prefix}{rank}"
        dist.barrier()
        log(rank, f"[step {step}] reached suspend point; destroying process group (releasing NCCL)...")
        dist.destroy_process_group()
        torch.cuda.synchronize(dev)
        open(marker, "w").close()
        log(rank, f"PG DESTROYED -- PID={os.getpid()} holding FSDP state on cuda:{local}, "
                  f"ready for cuda-checkpoint. (driver clears {marker}, or touch {args.resume_flag})")
        while os.path.exists(marker) and not os.path.exists(args.resume_flag):
            time.sleep(0.5)
        try: os.remove(marker)                               # idempotent (driver may have removed it)
        except OSError: pass
        t0 = time.time()
        store = dist.TCPStore(master_addr, port, world, is_master=(rank == 0),  # fresh port per round
                              timeout=datetime.timedelta(seconds=180))
        dist.init_process_group("nccl", store=store, rank=rank, world_size=world)
        dist.barrier()
        rebind_fsdp_pg(model, dist.group.WORLD, rank)
        if rank == 0 and os.path.exists(args.resume_flag):   # manual-mode cleanup only
            try: os.remove(args.resume_flag)
            except OSError: pass
        log(rank, f"PG RE-INIT (fresh store) in {time.time()-t0:.2f}s -- continuing fine-tuning")

    if args.suspend_steps:
        suspend_set = sorted(int(s) for s in args.suspend_steps.split(",") if s.strip())
    elif args.suspend_step is not None and args.suspend_step >= 0:
        suspend_set = [args.suspend_step]
    else:
        suspend_set = []
    suspend_port = {s: base_port + 1 + i for i, s in enumerate(suspend_set)}  # unique store port each
    if rank == 0 and suspend_set:
        log(0, f"will suspend/dump/resume at optimizer steps {suspend_set}")

    model.train()

    if args.job_steps and args.job_steps > 0:            # dump-free baseline: fixed job, then exit
        import itertools
        batches = itertools.cycle(loader)
        def opt_step():
            for _ in range(args.grad_accum):
                b = next(batches)
                out = model(input_ids=b["input_ids"].to(dev, non_blocking=True),
                            attention_mask=b["attention_mask"].to(dev, non_blocking=True),
                            labels=b["labels"].to(dev, non_blocking=True))
                (out.loss / args.grad_accum).backward()
            model.clip_grad_norm_(1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        for _ in range(3):                                # warmup (clocks, cudnn, NCCL)
            opt_step()
        torch.cuda.synchronize(dev); dist.barrier()
        if rank == 0:
            print("RUNJOB_READY", flush=True)
        trig = os.environ.get("RUNJOB_TRIGGER")
        while trig and not os.path.exists(trig):          # all ranks wait for the job trigger
            time.sleep(0.05)
        dist.barrier()
        for _ in range(args.job_steps):                   # the fixed job: N optimizer steps
            opt_step()
        torch.cuda.synchronize(dev); dist.barrier()
        if rank == 0:
            print(f"RUNJOB_DONE steps={args.job_steps}", flush=True)
        dist.destroy_process_group()
        return

    step = 0                                              # optimizer steps taken
    micro = 0
    accum_loss = 0.0
    t_start = time.time(); tok_seen = 0
    epoch = 0
    done = False
    while not done:
        sampler.set_epoch(epoch)
        for batch in loader:
            ids = batch["input_ids"].to(dev, non_blocking=True)
            lab = batch["labels"].to(dev, non_blocking=True)
            att = batch["attention_mask"].to(dev, non_blocking=True)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += out.loss.item()
            tok_seen += int(att.sum().item())
            micro += 1
            if micro % args.grad_accum == 0:
                model.clip_grad_norm_(1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and step % args.log_every == 0:
                    avg = accum_loss / (args.grad_accum * args.log_every)
                    tps = tok_seen / (time.time() - t_start)
                    used = torch.cuda.memory_allocated(dev) / 1e9
                    log(0, f"step {step:4d}/{args.steps}  loss {avg:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                           f"{tps:.0f} tok/s  gpu_alloc {used:.1f} GB")
                    accum_loss = 0.0
                if step in suspend_port:
                    suspend_resume(step, suspend_port[step])
                if step >= args.steps:
                    done = True; break
        epoch += 1

    dist.barrier()
    if rank == 0:
        log(0, f"done. {step} optimizer steps, {tok_seen} tokens, {time.time()-t_start:.0f}s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
