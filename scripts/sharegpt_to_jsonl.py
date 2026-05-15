#!/usr/bin/env python3
"""Convert ShareGPT V3 JSON to xk6-llm JSONL.

Mirrors the prompt selection of `vllm bench serve --dataset-name sharegpt`:
keep only conversations whose first turn is a human prompt followed by a gpt
response, take the first human message as the request, and tag the original
gpt response length as `reference_output_len` so consumers can replicate vLLM's
fixed-output-length behavior.

Usage:
  python scripts/sharegpt_to_jsonl.py \\
      --input ShareGPT_V3_unfiltered_cleaned_split.json \\
      --output data/sharegpt-1k.jsonl \\
      --num-prompts 1000 \\
      --seed 42 \\
      --max-tokens 256

Source dataset (community mirror; ~600MB):
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load(p: Path) -> list[dict]:
    with p.open() as f:
        return json.load(f)


def filter_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    """Return (human_prompt, gpt_response) pairs from valid two-turn openings."""
    pairs: list[tuple[str, str]] = []
    for row in rows:
        conv = row.get("conversations") or []
        if len(conv) < 2:
            continue
        a, b = conv[0], conv[1]
        if a.get("from") != "human" or b.get("from") != "gpt":
            continue
        prompt = (a.get("value") or "").strip()
        response = (b.get("value") or "").strip()
        if not prompt or not response:
            continue
        pairs.append((prompt, response))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Sets max_tokens on every emitted request (matches vLLM bench's "
        "--sharegpt-output-len when used with --ignore-eos).",
    )
    args = ap.parse_args()

    pairs = filter_pairs(load(args.input))
    if len(pairs) < args.num_prompts:
        print(
            f"warning: only {len(pairs)} valid pairs after filtering, requested "
            f"{args.num_prompts}",
            file=sys.stderr,
        )

    random.seed(args.seed)
    sampled = random.sample(pairs, k=min(args.num_prompts, len(pairs)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for prompt, response in sampled:
            rec = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "reference_output_len": len(response),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(sampled)} prompts -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
