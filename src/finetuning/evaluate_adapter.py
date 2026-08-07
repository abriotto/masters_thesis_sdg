from __future__ import annotations

"""
Diagnose a finetuned adapter before spending GPU hours on the voting reruns.

Answers three questions, for the base model or any checkpoint:

1. Does the model still emit a thought channel on the VOTING prompt?
   The training smoke test probes the role-inference prompt, i.e. the training
   distribution, where the finetuned model trivially reproduces the answer-only
   format. The voting prompt is a different instruction with a different output
   schema that the adapter never saw. Suppression there would confound the whole
   experiment; suppression only on the training prompt would not.

2. Did role inference actually improve, and on which roles?
   Reported per role, because parts of the label are not recoverable from a public
   transcript (Villager vs Insomniac frequently is not), so an aggregate hides
   whether the model learned mechanics or just fitted the marginal role distribution.

3. Does the voting output still parse into the expected JSON schema?

Run once per condition (base, checkpoint-25, checkpoint-50, final_adapter) and
compare. Decoding matches the voting evaluation throughout: t=1.0, top_p=0.95,
top_k=64, thinking enabled.
"""

import unsloth  # noqa: F401  - must precede transformers imports

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from src.utils.io_utils import find_repo_root, load_json, load_text
from src.utils.json_utils import parse_model_json
from src.utils.model_utils import call_local_model, load_local_model
from src.utils.prompt_utils import build_full_prompt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def thought_present(text: str, debug_info: dict[str, Any]) -> tuple[bool, bool]:
    """
    Return (emitted, parsed_cleanly).

    A thought that exceeds max_new_tokens never closes its channel, so the parser
    returns nothing even though the model clearly reasoned. Checking the opening
    marker separately keeps truncation from being mistaken for suppression.
    """
    thoughts = debug_info.get("internal_thoughts")
    parsed = bool(thoughts and str(thoughts).strip())
    emitted = parsed or ("channel>thought" in (text or ""))
    return emitted, parsed


def generate(
    model: Any,
    model_io: Any,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    text, debug_info = call_local_model(
        model=model,
        model_io=model_io,
        prompt=prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        gemma_enable_thinking=True,
        return_debug_info=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
    )
    return text, (debug_info or {})


def score_roles(
    predicted: Optional[dict[str, Any]],
    player_names: list[str],
    gold_roles: list[str],
) -> tuple[int, int, Counter, Counter]:
    """Return (correct, total, correct_by_role, seen_by_role)."""
    correct_by_role: Counter = Counter()
    seen_by_role: Counter = Counter()
    correct = 0

    roles = (predicted or {}).get("roles") if isinstance(predicted, dict) else None
    if not isinstance(roles, dict):
        roles = {}

    for name, gold in zip(player_names, gold_roles):
        seen_by_role[gold] += 1
        if str(roles.get(name, "")).strip() == gold:
            correct += 1
            correct_by_role[gold] += 1

    return correct, len(player_names), correct_by_role, seen_by_role


def run_role_inference(
    model: Any,
    model_io: Any,
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, Any]:
    rows = load_jsonl(repo_root / args.val_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    print("\n" + "=" * 78)
    print(f"ROLE INFERENCE - {len(rows)} validation episodes")
    print("=" * 78)

    total_correct = 0
    total_slots = 0
    correct_by_role: Counter = Counter()
    seen_by_role: Counter = Counter()
    emitted = 0
    parse_failures = 0
    out_tokens: list[int] = []

    for i, row in enumerate(rows, start=1):
        text, debug_info = generate(
            model, model_io, args.model_name, row["prompt"], args.max_new_tokens
        )
        has_thought, _ = thought_present(text, debug_info)
        emitted += int(has_thought)
        if debug_info.get("output_token_count"):
            out_tokens.append(int(debug_info["output_token_count"]))

        predicted = parse_model_json(text)
        if not isinstance(predicted, dict) or "roles" not in predicted:
            parse_failures += 1

        gold_roles = json.loads(row["completion"])["roles"]
        c, t, cbr, sbr = score_roles(
            predicted, row["player_names"], [gold_roles[n] for n in row["player_names"]]
        )
        total_correct += c
        total_slots += t
        correct_by_role.update(cbr)
        seen_by_role.update(sbr)

        print(
            f"[{i}/{len(rows)}] {row['session_name']}  {c}/{t} roles  "
            f"thought={has_thought}  tokens={debug_info.get('output_token_count')}"
        )

    accuracy = total_correct / total_slots if total_slots else 0.0
    per_role = {
        role: {
            "correct": correct_by_role.get(role, 0),
            "total": seen_by_role[role],
            "accuracy": correct_by_role.get(role, 0) / seen_by_role[role],
        }
        for role in sorted(seen_by_role)
    }

    print(f"\nrole accuracy:        {total_correct}/{total_slots} = {accuracy:.1%}")
    print(f"thought channel:      {emitted}/{len(rows)}")
    print(f"JSON parse failures:  {parse_failures}/{len(rows)}")
    if out_tokens:
        print(
            f"output tokens:        min {min(out_tokens)} / "
            f"median {sorted(out_tokens)[len(out_tokens) // 2]} / max {max(out_tokens)}"
        )
    print("\nper role:")
    for role, stats in sorted(per_role.items(), key=lambda kv: -kv[1]["total"]):
        print(f"  {role:14s} {stats['correct']:3d}/{stats['total']:<3d} = {stats['accuracy']:.1%}")

    return {
        "num_episodes": len(rows),
        "role_accuracy": accuracy,
        "correct": total_correct,
        "total": total_slots,
        "per_role": per_role,
        "thought_channel_emitted": emitted,
        "json_parse_failures": parse_failures,
        "output_tokens": out_tokens,
    }


def run_voting_probe(
    model: Any,
    model_io: Any,
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, Any]:
    """
    The decisive check: does the adapter still reason on the ACTUAL evaluation prompt?

    Uses the same prompt assembly as run_llm_votes, so the only thing being varied is
    the adapter.
    """
    index = load_json(repo_root / args.index_path)
    base_prompt = load_text(repo_root / args.voting_prompt_path)
    rules_text = load_text(repo_root / args.rules_path)
    rows = index[: args.voting_probe]

    print("\n" + "=" * 78)
    print(f"VOTING PROMPT PROBE - {len(rows)} games (the condition that actually matters)")
    print("=" * 78)

    emitted = 0
    valid_json = 0
    out_tokens: list[int] = []

    for i, row in enumerate(rows, start=1):
        transcript = load_text(repo_root / row["processed_txt_path"])
        prompt = build_full_prompt(
            base_prompt=base_prompt,
            rules_text=rules_text,
            player_names=row["player_names"],
            transcript_text=transcript,
        )
        text, debug_info = generate(
            model, model_io, args.model_name, prompt, args.max_new_tokens
        )
        has_thought, parsed_clean = thought_present(text, debug_info)
        emitted += int(has_thought)
        if debug_info.get("output_token_count"):
            out_tokens.append(int(debug_info["output_token_count"]))

        parsed = parse_model_json(text)
        ok = isinstance(parsed, dict) and "chosen_vote" in parsed and "justification" in parsed
        valid_json += int(ok)

        justification = (parsed or {}).get("justification") if isinstance(parsed, dict) else None
        print(
            f"[{i}/{len(rows)}] {row['session_name']}/{row['game_key']}  "
            f"thought={has_thought} (parsed={parsed_clean})  "
            f"tokens={debug_info.get('output_token_count')}  schema_ok={ok}"
        )
        if justification:
            print(f"      justification ({len(str(justification))} chars): {str(justification)[:160]!r}")

    print(f"\nthought channel:  {emitted}/{len(rows)}   (baseline: 573/573)")
    print(f"valid vote JSON:  {valid_json}/{len(rows)}")
    if out_tokens:
        print(
            f"output tokens:    min {min(out_tokens)} / "
            f"median {sorted(out_tokens)[len(out_tokens) // 2]} / max {max(out_tokens)}"
            "   (baseline median: 2884)"
        )

    return {
        "num_games": len(rows),
        "thought_channel_emitted": emitted,
        "valid_vote_json": valid_json,
        "output_tokens": out_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure role accuracy and thought-channel survival for an adapter."
    )
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-4-31B-it-unsloth-bnb-4bit")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA directory. Omit to measure the un-finetuned base model.",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/processed/jin2024_onuw/sft_role_inference/val.jsonl",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="data/processed/lai2023/onuw_transcripts_ready/index_cleaned.json",
    )
    parser.add_argument("--voting_prompt_path", type=str, default="src/prompts/voting_prompt_v4.txt")
    parser.add_argument("--rules_path", type=str, default="src/prompts/onuw_rules_v2.txt")
    parser.add_argument("--max_seq_length", type=int, default=12000)
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=-1, help="Cap role-inference episodes.")
    parser.add_argument(
        "--voting_probe",
        type=int,
        default=5,
        help="Games to run through the voting prompt. 0 disables.",
    )
    parser.add_argument(
        "--skip_role_inference",
        action="store_true",
        help="Only run the voting probe (much faster).",
    )
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    label = args.adapter_path or "BASE (no adapter)"
    print(f"Model:   {args.model_name}")
    print(f"Adapter: {label}")

    model_io, model = load_local_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        adapter_path=args.adapter_path,
    )

    results: dict[str, Any] = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "max_new_tokens": args.max_new_tokens,
    }

    if not args.skip_role_inference:
        results["role_inference"] = run_role_inference(model, model_io, args, repo_root)

    if args.voting_probe > 0:
        results["voting_probe"] = run_voting_probe(model, model_io, args, repo_root)

    if args.output_path:
        out = repo_root / args.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
