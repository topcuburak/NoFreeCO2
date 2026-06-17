#!/usr/bin/env python3
"""A2 TP=4 LOSSLESS suspend (real vLLM V1): swap KV HBM<->host + store/load host<->disk,
with MEASURED energy (telemetry), multi-cycle, per tier. The V1 analogue of TCC bench_storage.

Per cycle, 4 legs, each wrapped in the telemetry harness (latency + measured GPU+CPU + modeled
DRAM/drive), driven on all 4 workers via collective_rpc:
  swap_out : KV GPU->host   (copy each kv_caches tensor to a host buffer; held on the worker)
  store    : host->disk      (write the host KV shard to --store-out tier, per rank)
  load     : disk->host      (cold read back)
  swap_in  : host->GPU       (copy host buffer back into the kv_caches tensors)

Self-probes the kv_caches location first and PRINTS it, so a wrong attribute is diagnosable.
Run with sudo -E (RAPL + the root-owned data file). Records -> data/timed_dump.jsonl.

    sudo -E $(which python) workloads/a2_vllm/tp4_kv_suspend.py --tp 4 --gpu-mem-util 0.6 \
        --cycles 5 --store-out /var/data --tag a2_tp4_kv_nvme
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
sys.path.insert(0, _REPO)

import transparent_dump as td                               # noqa: E402
from harness import measure_operation                       # noqa: E402
from _common import build_telemetry, write_record, print_record  # noqa: E402

DRAM_W_PER_GB = 0.3


# ---------- worker-side functions (run on every rank via collective_rpc) ----------
def _find_kv(worker):
    import torch
    for hname in ("model_runner", "model_runner_"):
        mr = getattr(worker, hname, None)
        if mr is None:
            continue
        kv = getattr(mr, "kv_caches", None)
        if isinstance(kv, (list, tuple)) and kv and isinstance(kv[0], torch.Tensor):
            return list(kv)
    kv = getattr(worker, "kv_caches", None)
    if isinstance(kv, (list, tuple)) and kv and isinstance(kv[0], torch.Tensor):
        return list(kv)
    return None


def w_probe(worker):
    import os as _os
    import torch
    kv = _find_kv(worker)
    out = {"pid": _os.getpid(), "rank": getattr(worker, "rank", None), "found": kv is not None}
    if kv is not None:
        out["n_tensors"] = len(kv)
        out["bytes_GB"] = round(sum(t.numel() * t.element_size() for t in kv) / 1e9, 2)
        out["elem0"] = {"shape": list(kv[0].shape), "dtype": str(kv[0].dtype), "dev": str(kv[0].device)}
    else:
        mr = getattr(worker, "model_runner", None)
        out["model_runner_attrs"] = [a for a in dir(mr) if "kv" in a.lower() or "cache" in a.lower()] if mr else None
    return out


def w_swap_out(worker):
    import torch
    kv = _find_kv(worker)
    if not getattr(worker, "_kv_host", None):
        worker._kv_host = []
        for t in kv:
            try:
                h = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
            except Exception:
                h = torch.empty(t.shape, dtype=t.dtype, device="cpu")
            worker._kv_host.append(h)
    for h, t in zip(worker._kv_host, kv):
        h.copy_(t, non_blocking=True)
    torch.cuda.synchronize(getattr(worker, "device", None))
    return sum(t.numel() * t.element_size() for t in kv)


def w_swap_in(worker):
    import torch
    kv = _find_kv(worker)
    for t, h in zip(kv, worker._kv_host):
        t.copy_(h, non_blocking=True)
    torch.cuda.synchronize(getattr(worker, "device", None))
    return 1


def w_store(worker, store_dir):
    import os as _os
    import torch
    path = _os.path.join(store_dir, f"a2kv_rank{getattr(worker,'rank',0)}.bin")
    worker._kv_path = path
    tot = 0
    with open(path, "wb", buffering=0) as f:
        for h in worker._kv_host:                       # bf16 -> flat uint8 (numpy has no bf16)
            arr = h.contiguous().reshape(-1).view(torch.uint8).numpy()
            arr.tofile(f); tot += arr.nbytes
    _os.sync()
    return tot


def w_drop(worker):
    import os as _os                                        # UNTIMED cold-cache setup (sync flushes
    _os.system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")  # store's dirty pages, not load)
    return 1


def w_load(worker):
    import os as _os
    import torch
    fd = _os.open(worker._kv_path, _os.O_RDONLY)
    try:
        try: _os.posix_fadvise(fd, 0, 0, _os.POSIX_FADV_SEQUENTIAL)   # aggressive readahead
        except Exception: pass
        f = _os.fdopen(fd, "rb", buffering=0, closefd=False)
        for h in worker._kv_host:
            mv = memoryview(h.contiguous().reshape(-1).view(torch.uint8).numpy())  # into pinned buf
            got = 0
            while got < len(mv):
                r = f.readinto(mv[got:])
                if not r:
                    break
                got += r
    finally:
        _os.close(fd)
    return 1


def w_cleanup(worker):
    import os as _os
    p = getattr(worker, "_kv_path", None)
    if p:
        try: _os.remove(p)
        except OSError: pass
    return 1


def main() -> None:
    import torch  # noqa
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.6)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prefill-isl", type=int, default=4096, help="tokens to prefill (populate KV)")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--store-out", default="/var/data")
    ap.add_argument("--tag", default="a2_tp4_kv_nvme")
    args = ap.parse_args()
    drive_w = 3.0 if "home" in args.store_out else 50.0

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
              enable_prefix_caching=False)
    sp = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    n = min(args.prefill_isl, args.max_model_len - 1)
    llm.generate({"prompt_token_ids": [(i % 1000) + 5 for i in range(n)]}, sp, use_tqdm=False)
    print(f"[kv-suspend] prefilled {n} tokens", flush=True)

    print("[kv-suspend] === KV probe ===", flush=True)
    probe = llm.collective_rpc(w_probe)
    for r in probe:
        print(f"   {r}", flush=True)
    if not all(r.get("found") for r in probe):
        print("[kv-suspend] BLOCKER: kv_caches not found on some worker -- see attrs above", flush=True)
        return
    kv_total_gb = sum(r["bytes_GB"] for r in probe)
    print(f"[kv-suspend] KV pool total across {args.tp} ranks: {kv_total_gb:.1f} GB", flush=True)

    os.makedirs(args.store_out, exist_ok=True)
    tele = build_telemetry(nvml_gpus=list(range(args.tp)))
    tele.start()

    def emit(rec, phase, c):
        foot = kv_total_gb
        dram = DRAM_W_PER_GB * foot * rec.latency_s
        drive = drive_w * rec.latency_s if phase in ("store", "load") else 0.0
        meas = sum(s.energy_abs_j or 0.0 for s in rec.sources)
        rec.extra.update(mark_min=c, measured_abs_j=round(meas, 1), dram_model_j=round(dram, 1),
                         drive_model_j=round(drive, 1), full_total_j=round(meas + dram + drive, 1),
                         kv_total_gb=round(foot, 2))
        rec.config.update(tag=args.tag, phase=phase, mechanism="vllm_kv_swap", tp=args.tp)
        print_record(rec)
        print(f"  modeled DRAM {dram:.0f} + drive {drive:.0f} | FULL {meas+dram+drive:.0f} J", flush=True)
        write_record(rec, "timed_dump")

    legs = [("swap_out", w_swap_out, ()), ("store", w_store, (args.store_out,)),
            ("load", w_load, ()), ("swap_in", w_swap_in, ())]
    try:
        for c in range(args.cycles):
            for phase, fn, fargs in legs:
                if phase == "load":
                    llm.collective_rpc(w_drop)               # untimed: drop caches for a cold read
                rec = measure_operation(
                    tele, workload="timed_dump", operation=f"kv_{phase}",
                    state_bytes=int(kv_total_gb * 1e9), baseline_seconds=args.baseline,
                    op=(lambda fn=fn, fargs=fargs: llm.collective_rpc(fn, args=fargs)),
                    config={"phase": phase})
                emit(rec, phase, c)
            print(f"[kv-suspend] cycle {c} done", flush=True)
        llm.collective_rpc(w_cleanup)
    finally:
        tele.stop()
    print(f"[kv-suspend] done. {args.cycles} cycles, {kv_total_gb:.1f} GB -> tag {args.tag}", flush=True)


if __name__ == "__main__":
    main()
