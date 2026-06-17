#!/usr/bin/env python3
"""A3 -- ViT-Large training (single GPU), a real GPU compute workload for the suspend/restore
measurement. Active training loop (forward/backward/AdamW step) on synthetic ImageNet-shaped
batches -- the HBM footprint (weights + AdamW moments + activations) is identical to real
ImageNet data, which only changes pixel values, not the byte count the mechanism dumps.

Single process, one CUDA context, no NCCL -> transparent cuda-checkpoint suspends it in place
(like A2 TP=1). Launch it, let it reach steady-state footprint, then attach the dump harness:

    python workloads/a3_vit/vit_train.py --batch 32 &
    # then (root): sudo -E python scripts/timed_dump_experiment.py --marks-min 1,2,3,4,5 \
    #   --store --store-out /var/data --skip-criu --tag a3_vit_nvme

Footprint knob: --batch (activations) and the model is ViT-L/16 (~300M params). ~3-5 GB total.
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_large_patch16_224",
                    help="timm model name: vit_large_patch16_224 / vit_huge_patch14_224 / "
                         "vit_giant_patch14_224 (bigger weights+optimizer). Scales the FIXED footprint.")
    ap.add_argument("--batch", type=int, default=32, help="scales activation footprint ~linearly")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seconds", type=int, default=100000)
    ap.add_argument("--report-every", type=float, default=10.0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)

    # timm if available (any ViT size); else torchvision for the two it ships.
    try:
        import timm
        model = timm.create_model(args.model, pretrained=False, num_classes=1000,
                                  img_size=args.img_size)
        src = f"timm {args.model}"
    except Exception:
        import torchvision
        tv = {"vit_huge_patch14_224": "vit_h_14"}.get(args.model, "vit_l_16")
        model = getattr(torchvision.models, tv)(weights=None, image_size=args.img_size)
        src = f"torchvision {tv}"
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[a3-vit] PID={os.getpid()} {src} ({nparams/1e6:.0f}M params) batch={args.batch}", flush=True)

    x = torch.randn(args.batch, 3, args.img_size, args.img_size, device=dev)   # synthetic batch
    y = torch.randint(0, 1000, (args.batch,), device=dev)

    model.train()
    it = 0
    t0 = time.time(); last = t0
    while time.time() - t0 < args.seconds:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            out = model(x)
            loss = loss_fn(out, y)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        it += 1
        now = time.time()
        if now - last >= args.report_every:
            used = torch.cuda.memory_allocated(dev) / 1e9
            print(f"[a3-vit] iter={it} loss={loss.item():.3f} gpu_alloc={used:.2f}GB "
                  f"{it/(now-t0):.1f} it/s", flush=True)
            last = now


if __name__ == "__main__":
    main()
