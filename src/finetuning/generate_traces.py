from __future__ import annotations

"""
Generate reasoning traces with the BASE model for the role-inference episodes.

Why this exists
---------------
Answer-only finetuning eliminated the thought channel: the finetuned model dropped
from ~6,360 output tokens with a full thought block to ~130 tokens with none, on the
voting prompt, with thinking explicitly enabled. Every training sequence went
`<|turn>model` -> `{`, so the model learned to skip reasoning.

The fix is to put a thought block in every training sequence and mask the loss over
it, supervising only the gold answer (see build_traced_dataset.py). Thinking then
never leaves the training distribution, while no gradient shapes the reasoning - so
the reasoning measured later stays emergent rather than trained.

Traces come from the BASE model, so no foreign reasoning style is imported. Unsloth's
own guidance for Gemma 4 is to keep at least 75% reasoning-style examples when
finetuning these checkpoints if reasoning ability is to be preserved.

Gemma 4 31B thought format (Google's docs):
    <|turn>model <|channel>thought [reasoning] <channel|> [answer] <turn|>

Output
------
JSONL, one row per episode, with the trace, the model's own answer, the gold answer,
and whether they agree. The agreement rate matters: training a gold answer after a
trace that concluded differently teaches the model that its answer need not follow
from its reasoning, so the number should be reported, not assumed.
"""

import unsloth  # noqa: F401  - must precede transformers imports

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from src.utils.experiment_utils import (
    add_common_soft_warnings,
    get_internal_thoughts,
    prepare_response_for_json,
)
from src.utils.io_utils import find_repo_root
from src.utils.json_utils import parse_model_json
from src.utils.model_utils import call_local_model, load_local_model


THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"

# Substring that survives special-token stripping, for detecting a thought block that
# was emitted but never closed (and therefore could not be parsed).
THOUGHT_MARKER_FRAGMENT = "channel>thought"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_sessions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row.get("session_name") for row in load_jsonl(path)}


def roles_agree(predicted_text: str, gold_completion: str) -> Optional[bool]:
    """
    Whether the model's own answer matches the gold role assignment.

    Goes through prepare_response_for_json first, exactly as run_llm_votes does: the
    base model wraps its answer in ```json fences (present in all 573 baseline voting
    games), so parsing the raw text would fail and undercount agreement.
    """
    gold = json.loads(gold_completion).get("roles")
    cleaned = prepare_response_for_json(predicted_text)
    parsed = parse_model_json(cleaned) if isinstance(cleaned, str) else None
    if not isinstance(parsed, dict):
        return None
    predicted = parsed.get("roles")
    if not isinstance(predicted, dict):
        return None
    return predicted == gold


def generate_trace(
    model: Any,
    model_io: Any,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    text, debug_info = call_local_model(
        model=model,
        model_io=model_io,
        prompt=prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        gemma_enable_thinking=True,
        return_debug_info=True,
        # Same decoding as the voting evaluation.
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    debug_info = debug_info or {}
    thoughts = get_internal_thoughts(debug_info, save_internal_thoughts=True)
    out_tokens = debug_info.get("output_token_count")

    response_for_parsing = prepare_response_for_json(text)
    soft_warnings = add_common_soft_warnings(
        raw_response=text,
        response_for_parsing=response_for_parsing,
        parsed_output=parse_model_json(response_for_parsing)
        if isinstance(response_for_parsing, str)
        else None,
        debug_info=debug_info,
        max_new_tokens=max_new_tokens,
    )

    # A trace that runs past max_new_tokens never emits its closing marker, so the
    # parser cannot split it. Such traces are unusable here: the loss mask is anchored
    # on THOUGHT_CLOSE, and without it the mask would land in the wrong place.
    complete = bool(thoughts and str(thoughts).strip())
    truncated = bool(out_tokens and out_tokens >= max_new_tokens)
    emitted = complete or (THOUGHT_MARKER_FRAGMENT in (text or ""))

    return {
        "trace": str(thoughts) if complete else None,
        "model_answer": text,
        "trace_complete": complete,
        "truncated": truncated,
        "thought_emitted": complete or emitted,
        "output_token_count": out_tokens,
        "generation_time_sec": debug_info.get("generation_time_sec"),
        "soft_warnings": soft_warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate base-model reasoning traces for role-inference episodes."
    )
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-4-31B-it-unsloth-bnb-4bit")
    parser.add_argument(
        "--input_paths",
        type=str,
        nargs="+",
        default=[
            "data/processed/jin2024_onuw/sft_role_inference/train.jsonl",
            "data/processed/jin2024_onuw/sft_role_inference/val.jsonl",
        ],
        help="SFT files to generate traces for. Prompts are reused verbatim.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/jin2024_onuw/traces/role_inference_traces_31B.jsonl",
    )
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8000,
        help=(
            "Must be generous: a truncated trace never closes its thought channel and "
            "is unusable. Base traces on the voting prompt reached ~6,400 tokens."
        ),
    )
    parser.add_argument("--limit", type=int, default=-1, help="Cap episodes, for testing.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate episodes already present in the output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    output_path = repo_root / args.output_path

    rows: list[dict[str, Any]] = []
    for rel in args.input_paths:
        for row in load_jsonl(repo_root / rel):
            row["_source_file"] = rel
            rows.append(row)

    if args.limit > 0:
        rows = rows[: args.limit]

    if args.overwrite and output_path.exists():
        output_path.unlink()

    already = done_sessions(output_path)
    todo = [r for r in rows if r["session_name"] not in already]

    print(f"Model:        {args.model_name}")
    print(f"Episodes:     {len(rows)} total, {len(already)} already done, {len(todo)} to generate")
    print(f"Output:       {output_path}")
    print(f"max_new_tokens: {args.max_new_tokens}")

    if not todo:
        print("Nothing to do.")
        return

    model_io, model = load_local_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        adapter_path=None,  # traces must come from the UNMODIFIED base model
    )

    complete = 0
    agreed = 0
    scored = 0

    for i, row in enumerate(todo, start=1):
        try:
            result = generate_trace(
                model=model,
                model_io=model_io,
                model_name=args.model_name,
                prompt=row["prompt"],
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {row['session_name']} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue

        agreement = (
            roles_agree(result["model_answer"], row["completion"])
            if result["trace_complete"]
            else None
        )
        if agreement is not None:
            scored += 1
            agreed += int(agreement)
        complete += int(result["trace_complete"])

        append_jsonl(
            output_path,
            {
                "session_name": row["session_name"],
                "source_file": row["_source_file"],
                "player_names": row["player_names"],
                "gold_completion": row["completion"],
                "agrees_with_gold": agreement,
                **result,
            },
        )

        elapsed = result.get("generation_time_sec")
        print(
            f"[{i}/{len(todo)}] {row['session_name']}  complete={result['trace_complete']}  "
            f"tokens={result['output_token_count']}  agrees={agreement}  "
            f"{elapsed:.0f}s" if elapsed else
            f"[{i}/{len(todo)}] {row['session_name']}  complete={result['trace_complete']}",
            flush=True,
        )

    print(f"\ncomplete traces: {complete}/{len(todo)}")
    if scored:
        print(f"trace answer agrees with gold: {agreed}/{scored} = {agreed / scored:.1%}")
        print(
            "  (This is the decoupling risk: on the disagreeing episodes, training the\n"
            "   gold answer after this trace teaches that the answer need not follow the\n"
            "   reasoning. Report this number.)"
        )


if __name__ == "__main__":
    main()
