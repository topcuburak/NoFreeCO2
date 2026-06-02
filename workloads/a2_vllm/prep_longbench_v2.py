#!/usr/bin/env python3
"""Pick LongBench-v2 samples in a target INPUT-TOKEN window for A2 long-context runs.

Scans THUDM/LongBench-v2, builds the prompt (context + question [+ choices]),
tokenizes it with the *serving model's* tokenizer (so token counts match what vLLM
will see), keeps samples whose input length falls in [--min-tokens, --max-tokens],
and writes a jsonl that serve.py consumes:

    python scripts/prep_longbench_v2.py \
        --tokenizer meta-llama/Llama-3.1-8B \
        --min-tokens 70000 --max-tokens 90000 \
        --num-samples 32 --output data/lbv2_70k_90k.jsonl

    # then feed it to the runner:
    python workloads/a2_vllm/serve.py \
        --model meta-llama/Llama-3.1-8B --tensor-parallel-size 4 \
        --max-model-len 98304 --max-tokens 256 \
        --dataset data/lbv2_70k_90k.jsonl --prompt-field prompt --num-prompts 32

Needs `datasets` + `transformers`. Gated tokenizers (Llama) need HF_TOKEN set.
Use the SAME --tokenizer here as the --model you serve, or token counts drift.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics

_CHOICES = ("A", "B", "C", "D")


def build_prompt(row: dict, include_choices: bool) -> str:
    """Assemble a LongBench-v2 row into one prompt string.

    LongBench-v2 fields: context (the long doc), question, choice_A..choice_D, answer.
    """
    ctx = row.get("context", "") or ""
    q = row.get("question", "") or ""
    parts = [ctx, "", f"Question: {q}"]
    if include_choices:
        for c in _CHOICES:
            parts.append(f"{c}. {row.get('choice_' + c, '')}")
    parts.append("Answer:")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter LongBench-v2 by input-token length for long-context batching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.1-8B",
                    help="MUST match the model you will serve")
    ap.add_argument("--min-tokens", type=int, default=70000)
    ap.add_argument("--max-tokens", type=int, default=90000)
    ap.add_argument("--num-samples", type=int, default=32,
                    help="how many in-window samples to keep (0 = all matches)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--no-choices", action="store_true",
                    help="prompt = context+question only (no A-D options)")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomize scan order so picks aren't all from the front")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/lbv2_70k_90k.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset           # type: ignore
    from transformers import AutoTokenizer      # type: ignore

    print(f"[lbv2] loading tokenizer {args.tokenizer} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print("[lbv2] loading THUDM/LongBench-v2 ...")
    ds = load_dataset("THUDM/LongBench-v2", split=args.split)

    order = list(range(len(ds)))
    if args.shuffle:
        random.Random(args.seed).shuffle(order)

    picked: list[dict] = []
    scanned = 0
    want = args.num_samples or float("inf")
    for idx in order:
        if len(picked) >= want:
            break
        row = ds[idx]
        prompt = build_prompt(row, not args.no_choices)
        n = len(tok(prompt, add_special_tokens=False).input_ids)
        scanned += 1
        if args.min_tokens <= n <= args.max_tokens:
            picked.append({
                "prompt": prompt,
                "input_tokens": n,
                "_id": row.get("_id"),
                "domain": row.get("domain"),
                "sub_domain": row.get("sub_domain"),
                "difficulty": row.get("difficulty"),
                "length": row.get("length"),
                "answer": row.get("answer"),
            })
            print(f"  + match {len(picked):>3}: {n} tok  "
                  f"[{row.get('domain')}/{row.get('difficulty')}]")

    if not picked:
        raise SystemExit(
            f"[lbv2] no samples in [{args.min_tokens}, {args.max_tokens}] tok "
            f"after scanning {scanned}/{len(ds)}. Widen the window or check the tokenizer.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for rec in picked:
            f.write(json.dumps(rec) + "\n")

    lens = [r["input_tokens"] for r in picked]
    print(f"\n[lbv2] kept {len(picked)} / scanned {scanned} / total {len(ds)}")
    print(f"[lbv2] input tokens  min={min(lens)}  "
          f"median={int(statistics.median(lens))}  max={max(lens)}")
    print(f"[lbv2] -> {args.output}")
    print(f"[lbv2] feed to serve.py with: --dataset {args.output} "
          f"--prompt-field prompt --num-prompts {len(picked)}")


if __name__ == "__main__":
    main()
