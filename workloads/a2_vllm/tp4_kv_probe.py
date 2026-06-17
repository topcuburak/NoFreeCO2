#!/usr/bin/env python3
"""A2 TP=4 KV-swap probe: find the KV-cache tensors in each vLLM worker so we can checkpoint
them to host+disk (the LOSSLESS suspend that preserves KV, without a cuda-checkpoint/NCCL
teardown). Introspection only -- no swap yet. Reports, per worker, the attribute holding the
KV tensors + their shape/dtype/device/total bytes, so the full harness can copy them.

    python workloads/a2_vllm/tp4_kv_probe.py --model meta-llama/Llama-3.1-8B --tp 4

Needs 4 free GPUs + HF_TOKEN.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def kv_introspect(worker):
    import os as _os
    import torch
    info = {"pid": _os.getpid(), "rank": getattr(worker, "rank", None)}

    def summarize(obj, depth=0):
        if isinstance(obj, torch.Tensor):
            return {"tensor": list(obj.shape), "dtype": str(obj.dtype),
                    "dev": str(obj.device), "bytes": obj.numel() * obj.element_size()}
        if isinstance(obj, (list, tuple)) and obj and depth < 2:
            return {"seq_len": len(obj), "elem0": summarize(obj[0], depth + 1)}
        if isinstance(obj, dict) and obj and depth < 2:
            k = next(iter(obj))
            return {"dict_len": len(obj), "key0": str(k), "val0": summarize(obj[k], depth + 1)}
        return {"type": type(obj).__name__}

    # find the model_runner and any attribute that looks like KV cache
    mr = getattr(worker, "model_runner", None) or getattr(worker, "model_runner_", None)
    info["has_model_runner"] = mr is not None
    found = {}
    holders = [("worker", worker)] + ([("model_runner", mr)] if mr is not None else [])
    for hname, h in holders:
        for a in dir(h):
            if a.startswith("__"):
                continue
            if "kv_cache" in a.lower() or a.lower() in ("kv_caches", "kv_cache"):
                try:
                    val = getattr(h, a)
                    if val is not None and not callable(val):
                        found[f"{hname}.{a}"] = summarize(val)
                except Exception as e:
                    found[f"{hname}.{a}"] = {"err": repr(e)}
    info["kv_candidates"] = found
    # total bytes of the best candidate (a list/tuple of tensors)
    best = None
    for hname, h in holders:
        v = getattr(h, "kv_caches", None)
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
            best = v; info["kv_attr"] = f"{hname}.kv_caches"; break
    if best is not None:
        info["kv_n_tensors"] = len(best)
        info["kv_total_GB"] = round(sum(t.numel() * t.element_size() for t in best) / 1e9, 2)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    out = llm.generate(["The capital of France is"], sp)   # populate some KV
    print(f"[kv-probe] WARMUP OK: {out[0].outputs[0].text!r}", flush=True)

    print("[kv-probe] === KV introspection per worker ===", flush=True)
    import json
    for r in llm.collective_rpc(kv_introspect):
        print(json.dumps(r, indent=2), flush=True)


if __name__ == "__main__":
    main()
