"""Token-length distribution for the derivation dataset. CPU only, no weights.

    python -m src.finetuning.derivation.token_stats --model_name unsloth/gemma-4-E2B-it

Reports min / median / p90 / max for prompt, completion and total, and counts how
many games exceed 8192 and 12288 - the sequence limits used for E2B/E4B and 31B.

Nothing is dropped, truncated or filtered. The previous build silently lost games
to over-length filtering; this script only measures, and exits non-zero if any
game exceeds the --limit you pass so that a bad max_seq_length cannot pass unnoticed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.derivation.build_dataset import build_all  # noqa: E402


def dist(values):
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Token lengths for the SFT dataset.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--limit", type=int, default=4096,
                        help="Intended max_seq_length. Non-zero exit if any game exceeds it.")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    examples = build_all()
    prompt_lens, completion_lens, total_lens = [], [], []
    for example in examples:
        # Total is measured on the rendered chat template, so it includes the
        # control tokens the trainer will actually see.
        rendered = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)
        prompt_lens.append(len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]))
        completion_lens.append(
            len(tokenizer(example["completion"], add_special_tokens=False)["input_ids"]))
        total_lens.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))

    stats = {
        "model_name": args.model_name,
        "num_games": len(examples),
        "prompt": dist(prompt_lens),
        "completion": dist(completion_lens),
        "total_rendered": dist(total_lens),
        "over_8192": sum(1 for t in total_lens if t > 8192),
        "over_12288": sum(1 for t in total_lens if t > 12288),
        "over_limit": sum(1 for t in total_lens if t > args.limit),
        "limit": args.limit,
    }

    print("=" * 68)
    print("TOKEN LENGTHS - %s  (%d games)" % (args.model_name, len(examples)))
    print("=" * 68)
    print("%-16s %8s %8s %8s %8s" % ("field", "min", "median", "p90", "max"))
    for name in ("prompt", "completion", "total_rendered"):
        d = stats[name]
        print("%-16s %8d %8d %8d %8d" % (name, d["min"], d["median"], d["p90"], d["max"]))
    print()
    print("games over  8192 : %d" % stats["over_8192"])
    print("games over 12288 : %d" % stats["over_12288"])
    print("games over %5d : %d   <-- your --limit" % (args.limit, stats["over_limit"]))
    print()
    if stats["over_limit"]:
        print("*** %d game(s) exceed max_seq_length=%d. RAISE IT - do not filter them out."
              % (stats["over_limit"], args.limit))
    else:
        print("All %d games fit in max_seq_length=%d with %d tokens to spare."
              % (len(examples), args.limit, args.limit - stats["total_rendered"]["max"]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print("wrote %s" % args.out)
    return 1 if stats["over_limit"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
