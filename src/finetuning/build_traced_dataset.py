from __future__ import annotations

"""
Build the traced SFT dataset: base-model reasoning as context, gold answer supervised.

Each assistant turn becomes, per Google's Gemma 4 thought format:

    <|channel>thought
    {base-model reasoning}
    <channel|>{"roles": {...gold...}}

The trainer then anchors train_on_responses_only at `<channel|>` rather than
`<|turn>model\\n`, so the thought is context and the loss covers only the gold answer.
Two consequences, both deliberate:

- Thinking never leaves the training distribution, so the model cannot learn to skip
  the thought channel. That is the failure this dataset exists to fix.
- No gradient touches the reasoning, so the reasoning measured at evaluation time is
  not something the finetuning taught.

The train/val split is inherited verbatim from the answer-only dataset - same episodes,
same order - so the two variants are directly comparable.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Optional

from src.utils.io_utils import find_repo_root


# Gemma 4 thought delimiters. The 12B/26B/31B checkpoints use these; the small
# E2B/E4B checkpoints use a different pair, so both are CLI-overridable. Discover a
# model's pair with the trainer's --dry_run generation-prompt probe: rendering with
# enable_thinking=False pre-fills an empty thought block, which shows both markers.
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"

# Fallback only. Measured against a real trace this UNDER-counts by ~16% overall and
# ~23% on the reasoning text itself: chain-of-thought is bullets, newlines, repeated
# player names and short function words, closer to 3.2 chars/token than 4. Under-
# counting is the dangerous direction here - the gold answer sits at the end of the
# sequence, so an example that slips past the filter gets truncated in training and
# loses its entire supervised span.
FALLBACK_CHARS_PER_TOKEN = 3.2


def make_token_counter(model_name: Optional[str]) -> tuple[Callable[[str], int], str]:
    """
    Return (counter, description). Uses the real tokenizer when available.
    """
    if not model_name:
        return (
            lambda text: int(len(text) / FALLBACK_CHARS_PER_TOKEN),
            f"heuristic (chars/{FALLBACK_CHARS_PER_TOKEN})",
        )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        print(
            f"  WARNING: could not load tokenizer for {model_name} "
            f"({type(exc).__name__}); falling back to the heuristic.",
            flush=True,
        )
        return (
            lambda text: int(len(text) / FALLBACK_CHARS_PER_TOKEN),
            f"heuristic (chars/{FALLBACK_CHARS_PER_TOKEN})",
        )

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count, f"tokenizer ({model_name})"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_assistant_content(
    trace: str,
    gold_completion: str,
    thought_open: str = THOUGHT_OPEN,
    thought_close: str = THOUGHT_CLOSE,
) -> str:
    """
    Assemble the thought block plus gold answer.

    The trace is stripped and re-wrapped rather than reused verbatim, because
    parse_reasoning_response already removed the channel markers when extracting it.
    """
    return f"{thought_open}\n{trace.strip()}\n{thought_close}{gold_completion}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the traced SFT dataset from generated reasoning traces."
    )
    parser.add_argument(
        "--traces_path",
        type=str,
        default="data/processed/jin2024_onuw/traces/role_inference_traces.jsonl",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="data/processed/jin2024_onuw/sft_role_inference",
        help="Answer-only dataset whose split and prompts are reused.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/jin2024_onuw/sft_role_inference_traced",
    )
    parser.add_argument(
        "--thought_open",
        type=str,
        default=THOUGHT_OPEN,
        help="Opening thought delimiter for this checkpoint (31B default).",
    )
    parser.add_argument(
        "--thought_close",
        type=str,
        default=THOUGHT_CLOSE,
        help=(
            "Closing thought delimiter. MUST equal the trainer's --response_part, "
            "or the loss mask lands in the wrong place."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
        help=(
            "Tokenizer used to measure sequence length exactly. Pass an empty string "
            "to fall back to the char heuristic (not recommended - it under-counts)."
        ),
    )
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        default=12288,
        help=(
            "Drop examples whose prompt+thought+answer exceeds this, measured with the "
            "real tokenizer. Set it a few hundred BELOW the trainer's --max_seq_length "
            "to leave room for chat-template markers, which are not counted here. The "
            "gold answer sits at the END of the sequence, so truncation deletes the "
            "entire supervised span and training silently learns nothing."
        ),
    )
    parser.add_argument(
        "--require_agreement",
        action="store_true",
        help=(
            "Keep only episodes where the trace's own answer matched the gold. Removes "
            "the decoupling risk entirely, at the cost of dropping episodes and biasing "
            "the set toward games the base model already solved."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    traces_path = repo_root / args.traces_path
    if not traces_path.exists():
        raise FileNotFoundError(
            f"No traces at {traces_path}. Run src.finetuning.generate_traces first."
        )

    count_tokens, counter_desc = make_token_counter(args.model_name)
    print(f"Length measured with: {counter_desc}")

    traces = {row["session_name"]: row for row in load_jsonl(traces_path)}
    source_dir = repo_root / args.source_dir
    output_dir = repo_root / args.output_dir

    stats: dict[str, Any] = {
        "traces_path": args.traces_path,
        "source_dir": args.source_dir,
        "require_agreement": args.require_agreement,
        "thought_open": args.thought_open,
        "thought_close": args.thought_close,
        "length_counter": counter_desc,
        "max_total_tokens": args.max_total_tokens,
        "splits": {},
    }

    for split in ("train", "val"):
        source_path = source_dir / f"{split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing {source_path}")

        rows = load_jsonl(source_path)
        built: list[dict[str, Any]] = []
        dropped: dict[str, int] = {
            "no_trace": 0,
            "incomplete": 0,
            "disagreed": 0,
            "marker_in_trace": 0,
            "too_long": 0,
        }

        for row in rows:
            trace_row = traces.get(row["session_name"])
            if trace_row is None:
                dropped["no_trace"] += 1
                continue
            if not trace_row.get("trace_complete") or not trace_row.get("trace"):
                dropped["incomplete"] += 1
                continue
            if args.require_agreement and not trace_row.get("agrees_with_gold"):
                dropped["disagreed"] += 1
                continue

            # train_on_responses_only anchors the mask on the FIRST occurrence of
            # THOUGHT_CLOSE. A trace containing that literal string would move the
            # anchor into the middle of the reasoning, silently supervising the wrong
            # span. Drop rather than risk it.
            trace_text = trace_row["trace"]
            if args.thought_close in trace_text or args.thought_open in trace_text:
                dropped["marker_in_trace"] += 1
                continue

            assistant = build_assistant_content(
                trace_text, row["completion"], args.thought_open, args.thought_close
            )
            if assistant.count(args.thought_close) != 1:
                dropped["marker_in_trace"] += 1
                continue

            total_tokens = count_tokens(row["prompt"]) + count_tokens(assistant)
            if total_tokens > args.max_total_tokens:
                dropped["too_long"] += 1
                continue
            built.append(
                {
                    **{k: v for k, v in row.items() if k not in {"completion", "messages"}},
                    "completion": assistant,
                    "gold_answer": row["completion"],
                    "trace_agrees_with_gold": trace_row.get("agrees_with_gold"),
                    "messages": [
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": assistant},
                    ],
                    "completion_chars": len(assistant),
                }
            )

        write_jsonl(output_dir / f"{split}.jsonl", built)

        lengths = sorted(count_tokens(r["prompt"]) + count_tokens(r["completion"]) for r in built)
        agreed = sum(1 for r in built if r["trace_agrees_with_gold"])
        stats["splits"][split] = {
            "source_rows": len(rows),
            "built_rows": len(built),
            "dropped": dropped,
            "trace_agrees_with_gold": agreed,
            "agreement_rate": (agreed / len(built)) if built else None,
            "total_tokens": {
                "min": lengths[0] if lengths else None,
                "median": lengths[len(lengths) // 2] if lengths else None,
                "max": lengths[-1] if lengths else None,
            },
        }

        print(f"{split}: {len(built)}/{len(rows)} kept   dropped={dropped}")
        if built:
            print(f"  trace answer agreed with gold: {agreed}/{len(built)} = {agreed / len(built):.1%}")
            print(f"  total tokens:  min {lengths[0]} / median {lengths[len(lengths)//2]} / max {lengths[-1]}"
                  f"   (trainer --max_seq_length must exceed the max)")

    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nWritten to {output_dir}")
    print(
        "\nTrain with:\n"
        "  --train_path <output_dir>/train.jsonl --val_path <output_dir>/val.jsonl \\\n"
        f"  --response_part '{args.thought_close}' --enable_thinking_in_prompt"
    )


if __name__ == "__main__":
    main()
