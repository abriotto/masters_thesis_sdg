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
from typing import Any

from src.utils.io_utils import find_repo_root


THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_assistant_content(trace: str, gold_completion: str) -> str:
    """
    Assemble the thought block plus gold answer.

    The trace is stripped and re-wrapped rather than reused verbatim, because
    parse_reasoning_response already removed the channel markers when extracting it.
    """
    return f"{THOUGHT_OPEN}\n{trace.strip()}\n{THOUGHT_CLOSE}{gold_completion}"


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
        "--max_total_tokens",
        type=int,
        default=12288,
        help=(
            "Drop examples whose prompt+thought+answer exceeds this, estimated at 4 "
            "chars/token. Must match the trainer's --max_seq_length: the gold answer "
            "sits at the END of the sequence, so truncation deletes the entire "
            "supervised span and training silently learns nothing."
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

    traces = {row["session_name"]: row for row in load_jsonl(traces_path)}
    source_dir = repo_root / args.source_dir
    output_dir = repo_root / args.output_dir

    stats: dict[str, Any] = {
        "traces_path": args.traces_path,
        "source_dir": args.source_dir,
        "require_agreement": args.require_agreement,
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
            if THOUGHT_CLOSE in trace_text or THOUGHT_OPEN in trace_text:
                dropped["marker_in_trace"] += 1
                continue

            assistant = build_assistant_content(trace_text, row["completion"])
            if assistant.count(THOUGHT_CLOSE) != 1:
                dropped["marker_in_trace"] += 1
                continue

            approx_tokens = (len(row["prompt"]) + len(assistant)) // 4
            if approx_tokens > args.max_total_tokens:
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

        lengths = sorted((len(r["prompt"]) + len(r["completion"])) // 4 for r in built)
        agreed = sum(1 for r in built if r["trace_agrees_with_gold"])
        stats["splits"][split] = {
            "source_rows": len(rows),
            "built_rows": len(built),
            "dropped": dropped,
            "trace_agrees_with_gold": agreed,
            "agreement_rate": (agreed / len(built)) if built else None,
            "approx_tokens": {
                "min": lengths[0] if lengths else None,
                "median": lengths[len(lengths) // 2] if lengths else None,
                "max": lengths[-1] if lengths else None,
            },
        }

        print(f"{split}: {len(built)}/{len(rows)} kept   dropped={dropped}")
        if built:
            print(f"  trace answer agreed with gold: {agreed}/{len(built)} = {agreed / len(built):.1%}")
            print(f"  approx tokens: min {lengths[0]} / median {lengths[len(lengths)//2]} / max {lengths[-1]}"
                  f"   (trainer --max_seq_length must exceed the max)")

    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nWritten to {output_dir}")
    print(
        "\nTrain with:\n"
        "  --train_path <output_dir>/train.jsonl --val_path <output_dir>/val.jsonl \\\n"
        f"  --response_part '{THOUGHT_CLOSE}' --enable_thinking_in_prompt"
    )


if __name__ == "__main__":
    main()
