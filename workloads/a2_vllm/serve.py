#!/usr/bin/env python3
"""A2 -- Llama-3-8B throughput serving (vLLM, offline batch mode).

Standalone, self-contained runner: downloads the model + dataset and runs a vLLM
batch-throughput generation. ALL vLLM engine arguments are configurable on the
CLI (auto-exposed via vllm's EngineArgs.add_cli_args), plus dataset and sampling
flags. No classes -- run it directly:

    python serve.py --model meta-llama/Meta-Llama-3-8B \
        --dataset anon8231489123/ShareGPT_Vicuna_unfiltered \
        --num-prompts 1000 --max-tokens 128 \
        --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-model-len 8192

List every available vLLM knob with:  python serve.py --help

Gated models (Llama-3): set HF_TOKEN in the environment first.
Runs on ford (needs GPUs); vLLM/torch are imported inside main().
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import time


# --------------------------------------------------------------------------- #
# Dataset loading: model is fetched by vLLM/HF automatically; here we fetch and
# normalize prompts into a flat list[str].
# --------------------------------------------------------------------------- #
def load_prompts(args: argparse.Namespace) -> list[str]:
    """Return a list of prompt strings from a ShareGPT json, a local jsonl, an
    HF dataset id, or (if no --dataset) a synthetic fallback."""
    if not args.dataset:
        # Synthetic fallback so the script runs with zero external data.
        base = ("Explain the concept of carbon-aware computing and how workload "
                "shifting between regions can reduce operational emissions.")
        return [f"[{i}] {base}" for i in range(args.num_prompts)]

    prompts: list[str] = []
    src = args.dataset

    if src.endswith(".json") or os.path.basename(src).lower().startswith("sharegpt"):
        prompts = _load_sharegpt(_ensure_local(src), args.prompt_field)
    elif src.endswith(".jsonl"):
        prompts = _load_jsonl(_ensure_local(src), args.prompt_field)
    else:
        prompts = _load_hf_dataset(src, args.dataset_split, args.prompt_field)

    prompts = [p.strip() for p in prompts if p and p.strip()]
    if not prompts:
        raise RuntimeError(f"no usable prompts found in dataset {src!r}")

    rng = random.Random(args.seed)
    if args.shuffle:
        rng.shuffle(prompts)
    return prompts[: args.num_prompts]


def _ensure_local(src: str) -> str:
    """If src is an hf 'repo::file' ref or a bare HF file, download it; else return as-is."""
    if os.path.exists(src):
        return src
    # Try the HF hub (e.g. a ShareGPT json shipped in a dataset repo).
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as e:
        raise RuntimeError(f"{src} not local and huggingface_hub unavailable: {e}")
    repo, _, filename = src.partition("::")
    if not filename:
        raise RuntimeError(
            f"to pull a single file from the hub use 'repo_id::path/in/repo.json', got {src!r}")
    return hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")


def _load_sharegpt(path: str, field: str) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    out: list[str] = []
    for conv in data:
        turns = conv.get("conversations") or conv.get("conversation") or []
        for t in turns:
            if t.get("from") in ("human", "user"):
                out.append(t.get("value", ""))
                break
    return out


def _load_jsonl(path: str, field: str) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line).get(field, ""))
    return out


def _load_hf_dataset(repo: str, split: str, field: str) -> list[str]:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(repo, split=split)
    return [row[field] for row in ds if field in row]


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def collect_latencies(outputs) -> dict:
    """Per-request latency from vLLM RequestOutput.metrics, if populated.

    ttft = first_token_time - arrival_time      (time to first token)
    e2e  = finished_time   - arrival_time       (end-to-end, incl. queue/wave wait)
    tpot = (finished - first_token) / (out_tok-1)  (time per output token, decode)

    In offline batch mode arrival_time is ~shared, so ttft/e2e include the wait for
    a sequence's wave to be scheduled -- i.e. realistic latency under this batch.
    """
    ttft, e2e, tpot = [], [], []
    for o in outputs:
        m = getattr(o, "metrics", None)
        if not m:
            continue
        arr = getattr(m, "arrival_time", None)
        ftt = getattr(m, "first_token_time", None)
        fin = getattr(m, "finished_time", None)
        n_out = len(o.outputs[0].token_ids)
        if arr is not None and ftt is not None:
            ttft.append(ftt - arr)
        if arr is not None and fin is not None:
            e2e.append(fin - arr)
        if ftt is not None and fin is not None and n_out > 1:
            tpot.append((fin - ftt) / (n_out - 1))
    return {"ttft_s": ttft, "e2e_s": e2e, "tpot_s": tpot}


def _summarize(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 4),
        "p50": round(_pct(xs, 50), 4),
        "p90": round(_pct(xs, 90), 4),
        "p99": round(_pct(xs, 99), 4),
        "max": round(max(xs), 4),
    }


def build_token_prompts(llm, n: int, input_len: int, seed: int):
    """Exact-length synthetic prompts for controlled long-context batching:
    n sequences, each exactly input_len random token ids. Returns vLLM inputs."""
    tok = llm.get_tokenizer()
    vocab = getattr(tok, "vocab_size", None) or 32000
    lo, hi = 10, max(20, vocab - 100)          # avoid special tokens at the edges
    rng = random.Random(seed)
    id_lists = [[rng.randint(lo, hi) for _ in range(input_len)] for _ in range(n)]
    try:
        from vllm import TokensPrompt  # type: ignore
        return [TokensPrompt(prompt_token_ids=ids) for ids in id_lists]
    except Exception:                          # older vLLM: dict form
        return [{"prompt_token_ids": ids} for ids in id_lists]


# --------------------------------------------------------------------------- #
# CLI: vLLM engine args (auto) + dataset/sampling/output args (ours)
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    from vllm.engine.arg_utils import EngineArgs  # type: ignore

    p = argparse.ArgumentParser(
        description="A2 vLLM batch-throughput runner (all vLLM engine args configurable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    EngineArgs.add_cli_args(p)               # every vLLM engine knob
    p.set_defaults(model="meta-llama/Meta-Llama-3-8B")
    # Default prefix caching OFF: with --repeat, identical prompts would hit the
    # prefix cache and skip prefill (no real duration increase). Re-enable with
    # --enable-prefix-caching if you want it.
    p.set_defaults(enable_prefix_caching=False)

    g = p.add_argument_group("dataset")
    g.add_argument("--dataset", default=None,
                   help="HF dataset id, local .json/.jsonl, or 'repo_id::file.json'. "
                        "Omit for a synthetic prompt set.")
    g.add_argument("--dataset-split", default="train")
    g.add_argument("--prompt-field", default="conversations",
                   help="jsonl/HF field holding the prompt text")
    g.add_argument("--num-prompts", type=int, default=1000,
                   help="number of sequences (= batch size in --input-len mode)")
    g.add_argument("--shuffle", action="store_true")
    g.add_argument("--repeat", type=int, default=1,
                   help="replay the loaded prompt set this many times for a linear "
                        "duration increase (pair with prefix caching OFF, the default)")

    lc = p.add_argument_group("long-context / synthetic")
    lc.add_argument("--input-len", type=int, default=None,
                    help="if set, ignore --dataset and submit --num-prompts synthetic "
                         "prompts of EXACTLY this many input tokens (controlled long-context "
                         "batching). KV size = num_prompts * (input_len + max_tokens) * "
                         "per-token-KV-bytes. Needs a model whose --max-model-len >= "
                         "input_len + max_tokens (e.g. Llama-3.1-8B for >8k).")

    s = p.add_argument_group("sampling")
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--top-p", type=float, default=1.0)

    o = p.add_argument_group("output")
    o.add_argument("--output-json", default=None, help="write throughput metrics here")
    o.add_argument("--hold-idle", type=float, default=0.0,
                   help="after generation, keep the process alive (holding the GPU + KV "
                        "pool) IDLE for this many seconds, issuing NO GPU work. The KV "
                        "pool is pre-allocated at init so the footprint is unchanged; "
                        "idling just means no in-flight collectives. Required for "
                        "checkpointing TP>1: cuda-checkpoint lock-all DEADLOCKS on an "
                        "in-flight all-reduce, so the dump must hit a collective-free point.")
    return p


def main() -> None:
    from vllm import LLM, SamplingParams              # type: ignore
    from vllm.engine.arg_utils import EngineArgs      # type: ignore

    args = build_parser().parse_args()

    engine_args = EngineArgs.from_cli_args(args)
    print(f"[a2] starting vLLM: model={engine_args.model} "
          f"tp={engine_args.tensor_parallel_size} "
          f"gpu_mem_util={engine_args.gpu_memory_utilization} "
          f"max_len={engine_args.max_model_len}")
    # Shallow extract -- NOT dataclasses.asdict(), which recursively converts
    # nested config objects (e.g. CompilationConfig) into dicts with None fields
    # that fail pydantic validation when LLM reconstructs them.
    engine_kwargs = {f.name: getattr(engine_args, f.name)
                     for f in dataclasses.fields(engine_args)}
    llm = LLM(**engine_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
    )

    if args.input_len:
        inputs = build_token_prompts(llm, args.num_prompts, args.input_len, args.seed)
        print(f"[a2] synthetic long-context batch: {len(inputs)} seqs x "
              f"{args.input_len} input tok (+{args.max_tokens} out)")
    else:
        print(f"[a2] loading prompts from {args.dataset or 'synthetic'} ...")
        inputs = load_prompts(args)
        print(f"[a2] {len(inputs)} prompts ready")

    unique = len(inputs)
    if args.repeat > 1:
        inputs = list(inputs) * args.repeat
    pc = getattr(engine_args, "enable_prefix_caching", None)
    print(f"[a2] prefix_caching={pc}  repeat=x{args.repeat}  "
          f"unique={unique}  total_requests={len(inputs)}")

    print(f"[a2] generating ({len(inputs)} seqs, max_tokens={args.max_tokens}) ...")
    t0 = time.perf_counter()
    outputs = llm.generate(inputs, sampling)
    elapsed = time.perf_counter() - t0

    in_tok = sum(len(o.prompt_token_ids) for o in outputs)
    out_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
    metrics = {
        "model": engine_args.model,
        "num_prompts": len(outputs),
        "unique_prompts": unique,
        "repeat": args.repeat,
        "prefix_caching": getattr(engine_args, "enable_prefix_caching", None),
        "elapsed_s": round(elapsed, 3),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "output_tok_per_s": round(out_tok / elapsed, 1) if elapsed else None,
        "total_tok_per_s": round((in_tok + out_tok) / elapsed, 1) if elapsed else None,
        "requests_per_s": round(len(outputs) / elapsed, 2) if elapsed else None,
    }
    print("[a2] throughput:")
    for k, v in metrics.items():
        print(f"    {k:18} {v}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[a2] metrics -> {args.output_json}")

    # Idle-hold: keep the engine alive holding the GPU + pre-allocated KV pool, but
    # issue NO further generate() -> no forward pass -> no all-reduce in flight. This
    # is the collective-free point at which cuda-checkpoint lock-all can succeed for
    # TP>1 (a busy server is almost never simultaneously outside a collective).
    if args.hold_idle > 0:
        used_gb = None
        try:
            import pynvml
            pynvml.nvmlInit()
            used_gb = sum(pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(i)).used
                for i in range(pynvml.nvmlDeviceGetCount())) / 1e9
        except Exception:
            pass
        foot = f"{used_gb:.1f} GB across GPUs" if used_gb else "(GPU pool retained)"
        print(f"[a2] HOLD-IDLE {args.hold_idle:.0f}s -- engine quiescent, {foot}; "
              f"NO collectives in flight. Safe to checkpoint NOW (run timed_dump). "
              f"engine pid {os.getpid()}")
        held = 0.0
        try:
            while held < args.hold_idle:
                time.sleep(5.0)
                held += 5.0
        except KeyboardInterrupt:
            print("[a2] hold-idle interrupted; exiting")
        print("[a2] hold-idle done; exiting")


if __name__ == "__main__":
    main()
