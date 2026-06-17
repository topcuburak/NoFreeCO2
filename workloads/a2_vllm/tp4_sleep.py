#!/usr/bin/env python3
"""A2 TP=4 temporal suspend via vLLM-native sleep()/wake_up() -- the SUPPORTED TP>1 serving
suspend (app-native, analogue of A1's DCP). Unlike the transparent cuda-checkpoint path, this
doesn't tear down NCCL under a live engine -- vLLM's sleep() quiesces the engine internally and
frees GPU memory (offload weights HBM->host, drop KV pool); wake_up() reloads.

Measures, per cycle: sleep (HBM->host) + wake (host->HBM) latency + FULL energy (measured GPU
board x4 + CPU pkg, modeled DRAM for the offloaded weights resident in host). No disk leg --
sleep offloads to host RAM (for a migration dump you'd add a host->disk write of the offload).

    python workloads/a2_vllm/tp4_sleep.py --model meta-llama/Llama-3.1-8B --tp 4 --cycles 5

Needs 4 free GPUs + HF_TOKEN. Records -> data/timed_dump.jsonl tag a2_tp4_sleep.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # JIT needs nvcc (absent) -> off
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
sys.path.insert(0, _REPO)

import transparent_dump as td                              # noqa: E402
from harness import measure_operation                      # noqa: E402
from _common import build_telemetry, write_record, print_record  # noqa: E402

DRAM_W_PER_GB = 0.3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--sleep-level", type=int, default=1, help="1=offload weights to host; 2=discard")
    ap.add_argument("--tag", default="a2_tp4_sleep")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enforce_eager=True,
              enable_sleep_mode=True, gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    prompt = "The capital of France is"

    if not (hasattr(llm, "sleep") and hasattr(llm, "wake_up")):
        print("[tp4-sleep] BLOCKER: this vLLM build has no llm.sleep/wake_up -> fall back to compose")
        return
    print(f"[tp4-sleep] warmup: {llm.generate([prompt], sp)[0].outputs[0].text!r}", flush=True)

    tele = build_telemetry(nvml_gpus=list(range(args.tp)))   # all TP GPUs
    tele.start()

    def emit(rec, foot_gb):
        dram = DRAM_W_PER_GB * foot_gb * rec.latency_s
        meas = sum(s.energy_abs_j or 0.0 for s in rec.sources)
        rec.extra.update(measured_abs_j=round(meas, 1), dram_model_j=round(dram, 1),
                         full_total_j=round(meas + dram, 1))
        rec.config["tag"] = args.tag
        print_record(rec)
        print(f"  modeled DRAM {dram:.0f} J | FULL (meas GPU+CPU + DRAM): {meas + dram:.0f} J", flush=True)
        write_record(rec, "timed_dump")

    try:
        for c in range(args.cycles):
            before = td.gpu_used_bytes()
            rec_s = measure_operation(
                tele, workload="timed_dump", operation="vllm_sleep", state_bytes=before,
                baseline_seconds=args.baseline, op=lambda: llm.sleep(level=args.sleep_level),
                config={"mark_min": c, "domain": "accelerator", "phase": "suspend",
                        "mechanism": "vllm_sleep", "tp": args.tp})
            after = td.gpu_used_bytes()
            rec_s.extra["gpu_freed_bytes"] = before - after
            emit(rec_s, before / 1e9)
            print(f"[tp4-sleep] cycle {c}: SLEEP freed {(before-after)/1e9:.1f} GB "
                  f"(now {after/1e9:.1f} GB) in {rec_s.latency_s:.2f}s", flush=True)

            if args.hold > 0:
                time.sleep(args.hold)

            rec_w = measure_operation(
                tele, workload="timed_dump", operation="vllm_wake", state_bytes=before,
                baseline_seconds=args.baseline, op=lambda: llm.wake_up(),
                config={"mark_min": c, "domain": "accelerator", "phase": "resume",
                        "mechanism": "vllm_sleep", "tp": args.tp})
            emit(rec_w, before / 1e9)
            print(f"[tp4-sleep] cycle {c}: WAKE in {rec_w.latency_s:.2f}s "
                  f"(GPU {td.gpu_used_bytes()/1e9:.1f} GB)", flush=True)

            out = llm.generate([prompt], sp)
            print(f"[tp4-sleep] cycle {c}: RESUMED gen={out[0].outputs[0].text!r}", flush=True)
    finally:
        tele.stop()
    print(f"[tp4-sleep] done. {args.cycles} cycles -> data/timed_dump.jsonl tag {args.tag}", flush=True)


if __name__ == "__main__":
    main()
