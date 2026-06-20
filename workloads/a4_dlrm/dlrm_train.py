#!/usr/bin/env python3
"""A4 -- DLRM training (single GPU), the memory-EXTREME workload: the footprint is the
embedding table(s), dialed to a target via --emb-gb. Real DLRM structure (sparse embedding
lookups + bottom/top MLP + feature interaction), synthetic clicklog-shaped batches. SGD (no
optimizer states) so the footprint ≈ the embedding table -- a clean, large, fixed knob.

Single process, no NCCL -> transparent cuda-checkpoint suspends it in place. Launch, let it
reach steady footprint, then attach the dump harness:

    python workloads/a4_dlrm/dlrm_train.py --emb-gb 25 &
    # then (root): scripts/timed_dump_experiment.py --marks-min 1,2,3,4,5 --store \
    #   --store-out /var/data --skip-criu --tag a4_dlrm_nvme

--emb-gb sets the embedding-table footprint (the whole point of A4 — embeddings dominate; MLP <1 GB).
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-gb", type=float, default=25.0, help="embedding-table footprint (GiB)")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--dense-dim", type=int, default=13)
    ap.add_argument("--num-sparse", type=int, default=26, help="sparse feature lookups/sample")
    ap.add_argument("--seconds", type=int, default=100000)
    ap.add_argument("--report-every", type=float, default=10.0)
    ap.add_argument("--job-steps", type=int, default=0,
                    help="dump-free baseline: warmup, then run N training steps and exit (handshake)")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    rows = max(1, int(args.emb_gb * (1024 ** 3) / (args.emb_dim * 4)))   # fp32 embedding
    emb = nn.Embedding(rows, args.emb_dim, sparse=True).to(dev)   # sparse grad: only touched rows
    bot = nn.Sequential(nn.Linear(args.dense_dim, 256), nn.ReLU(),
                        nn.Linear(256, args.emb_dim)).to(dev)
    top = nn.Sequential(nn.Linear(args.emb_dim * 2, 256), nn.ReLU(),
                        nn.Linear(256, 1)).to(dev)
    params = list(emb.parameters()) + list(bot.parameters()) + list(top.parameters())
    # plain SGD (momentum=0): no optimizer states AND handles the embedding's sparse grad ->
    # backward never materializes a full-table dense gradient, so footprint stays ≈ emb table.
    opt = torch.optim.SGD(params, lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    emb_gb = rows * args.emb_dim * 4 / 1e9
    print(f"[a4-dlrm] PID={os.getpid()} embedding {rows/1e6:.0f}M rows x {args.emb_dim} "
          f"= {emb_gb:.1f} GB, batch={args.batch}", flush=True)

    dense = torch.randn(args.batch, args.dense_dim, device=dev)
    idx = torch.randint(0, rows, (args.batch, args.num_sparse), device=dev)
    label = torch.randint(0, 2, (args.batch, 1), device=dev).float()

    def step():
        e = emb(idx).mean(dim=1)                            # sparse lookups -> [B, emb_dim]
        out = top(torch.cat([e, bot(dense)], dim=1))
        loss = loss_fn(out, label)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        return loss

    if args.job_steps > 0:                                  # dump-free baseline: fixed job, then exit
        for _ in range(5):
            step()
        torch.cuda.synchronize(dev)
        print("RUNJOB_READY", flush=True)
        trig = os.environ.get("RUNJOB_TRIGGER")
        while trig and not os.path.exists(trig):
            time.sleep(0.05)
        for _ in range(args.job_steps):
            step()
        torch.cuda.synchronize(dev)
        print(f"RUNJOB_DONE steps={args.job_steps}", flush=True)
        return

    it = 0
    t0 = time.time(); last = t0
    while time.time() - t0 < args.seconds:
        e = emb(idx).mean(dim=1)                            # sparse lookups -> [B, emb_dim]
        d = bot(dense)
        out = top(torch.cat([e, d], dim=1))
        loss = loss_fn(out, label)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        it += 1
        now = time.time()
        if now - last >= args.report_every:
            used = torch.cuda.memory_allocated(dev) / 1e9
            print(f"[a4-dlrm] iter={it} loss={loss.item():.4f} gpu_alloc={used:.1f}GB "
                  f"{it/(now-t0):.1f} it/s", flush=True)
            last = now


if __name__ == "__main__":
    main()
