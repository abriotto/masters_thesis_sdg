from __future__ import annotations

"""
Diagnose a finetuned adapter before spending GPU hours on the voting reruns.

Answers three questions, for the base model or any checkpoint:

1. Does the model still emit a thought channel on the VOTING prompt?
   Measured by running src.voting.run_llm_votes itself - with --adapter_path and a
   small --max_games - and summarising its output here via
   --summarize_voting_results. Generation is NOT reimplemented in this file: a copy
   of the runner would only tell you about the copy.

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

from src.utils.io_utils import find_repo_root, load_json
from src.utils.json_utils import parse_model_json
from src.utils.model_utils import call_local_model, load_local_model


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
    exact_games = 0
    correct_by_role: Counter = Counter()
    seen_by_role: Counter = Counter()
    emitted = 0
    parse_failures = 0
    out_tokens: list[int] = []
    episodes: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        text, debug_info = generate(
            model, model_io, args.model_name, row["prompt"], args.max_new_tokens
        )
        has_thought, _ = thought_present(text, debug_info)
        emitted += int(has_thought)
        if debug_info.get("output_token_count"):
            out_tokens.append(int(debug_info["output_token_count"]))

        predicted = parse_model_json(text)
        parse_ok = isinstance(predicted, dict) and "roles" in predicted
        if not parse_ok:
            parse_failures += 1

        gold_roles = json.loads(row["completion"])["roles"]
        c, t, cbr, sbr = score_roles(
            predicted, row["player_names"], [gold_roles[n] for n in row["player_names"]]
        )
        total_correct += c
        total_slots += t
        exact_games += int(c == t)
        correct_by_role.update(cbr)
        seen_by_role.update(sbr)

        # Per-episode record. Two reasons this is kept: a parse failure is
        # otherwise unexaminable, since the raw text is discarded; and the
        # aggregate alone cannot support a paired base-vs-finetuned bootstrap,
        # which needs the per-game values. Raw text is stored only when parsing
        # failed, and clipped, because a runaway generation can reach 80k chars.
        rec = {
            "session_name": row["session_name"],
            "correct": c,
            "total": t,
            "exact_game": bool(c == t),
            "thought_emitted": bool(has_thought),
            "output_token_count": debug_info.get("output_token_count"),
            "hit_max_new_tokens": bool(
                debug_info.get("output_token_count") == args.max_new_tokens
            ),
            "parse_ok": bool(parse_ok),
        }
        if not parse_ok:
            rec["raw_head"] = text[:2000]
            rec["raw_tail"] = text[-2000:]
            rec["raw_char_count"] = len(text)
        episodes.append(rec)

        print(
            f"[{i}/{len(rows)}] {row['session_name']}  {c}/{t} roles  "
            f"thought={has_thought}  parse_ok={parse_ok}  "
            f"tokens={debug_info.get('output_token_count')}"
        )

    accuracy = total_correct / total_slots if total_slots else 0.0
    exact_game_accuracy = exact_games / len(rows) if rows else 0.0
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
        "exact_game_matches": exact_games,
        "episodes": episodes,
        "exact_game_accuracy": exact_game_accuracy,
        "per_role": per_role,
        "thought_channel_emitted": emitted,
        "json_parse_failures": parse_failures,
        "output_tokens": out_tokens,
    }


def summarize_voting_results(repo_root: Path, results_dir: str) -> dict[str, Any]:
    """
    Summarise voting results produced by src.voting.run_llm_votes.

    Deliberately does NOT generate anything. Re-implementing the voting loop here
    would mean measuring a copy of the runner rather than the runner itself; instead
    run_llm_votes writes its normal result JSONs (with --adapter_path and a small
    --max_games) and this reads them. Same code path as the real evaluation.
    """
    root = repo_root / results_dir
    if not root.exists():
        raise FileNotFoundError(f"No results under {root}")

    files = sorted(root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No result JSONs under {root}")

    print("\n" + "=" * 78)
    print(f"VOTING RESULTS SUMMARY - {results_dir}")
    print("=" * 78)

    emitted = 0
    valid = 0
    out_tokens: list[int] = []
    just_chars: list[int] = []
    thought_chars: list[int] = []
    errors = 0

    for path in files:
        record = load_json(path)
        if record.get("error"):
            errors += 1
            continue

        thoughts = record.get("internal_thoughts") or ""
        if str(thoughts).strip():
            emitted += 1
            thought_chars.append(len(str(thoughts)))

        if (record.get("validation") or {}).get("is_valid"):
            valid += 1

        parsed = record.get("parsed_output") or {}
        justification = parsed.get("justification")
        if isinstance(justification, str):
            just_chars.append(len(justification))

        tokens = (record.get("debug_info") or {}).get("output_token_count")
        if tokens:
            out_tokens.append(int(tokens))

    n = len(files)

    def _spread(values: list[int], label: str, reference: str = "") -> None:
        if not values:
            print(f"  {label:22s} (none)")
            return
        ordered = sorted(values)
        print(
            f"  {label:22s} min {ordered[0]:6d}  median {ordered[len(ordered) // 2]:6d}  "
            f"max {ordered[-1]:6d}   {reference}"
        )

    print(f"  games:                 {n}   (errors: {errors})")
    print(f"  thought channel:       {emitted}/{n}          baseline: 573/573")
    print(f"  valid vote JSON:       {valid}/{n}")
    _spread(out_tokens, "output tokens", "baseline median: 2884")
    _spread(just_chars, "justification chars", "baseline median: 452")
    _spread(thought_chars, "thought chars", "baseline median: 9893")

    return {
        "results_dir": results_dir,
        "num_games": n,
        "num_errors": errors,
        "thought_channel_emitted": emitted,
        "valid_vote_json": valid,
        "output_tokens": out_tokens,
        "justification_chars": just_chars,
        "thought_chars": thought_chars,
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
    parser.add_argument("--max_seq_length", type=int, default=12000)
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=-1, help="Cap role-inference episodes.")
    parser.add_argument(
        "--summarize_voting_results",
        type=str,
        default=None,
        help=(
            "Path to a results directory written by src.voting.run_llm_votes, relative "
            "to the repo root. Summarised without generating anything - run the real "
            "runner with --adapter_path and --max_games, then point this at its output."
        ),
    )
    parser.add_argument(
        "--skip_role_inference",
        action="store_true",
        help="Only summarise voting results; load no model.",
    )
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    results: dict[str, Any] = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "max_new_tokens": args.max_new_tokens,
    }

    if args.summarize_voting_results:
        results["voting_summary"] = summarize_voting_results(
            repo_root, args.summarize_voting_results
        )

    if not args.skip_role_inference:
        label = args.adapter_path or "BASE (no adapter)"
        print(f"Model:   {args.model_name}")
        print(f"Adapter: {label}")

        model_io, model = load_local_model(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            adapter_path=args.adapter_path,
        )
        results["role_inference"] = run_role_inference(model, model_io, args, repo_root)

    if args.output_path:
        out = repo_root / args.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
