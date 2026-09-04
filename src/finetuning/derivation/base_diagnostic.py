"""Base-model diagnostic: can the base model already do this task?

Runs one variant over the validation games in two prompt conditions:

- `night`      the full prompt, private Moderator night messages included. This is
               the finetuning condition. The derivation is deducible from it.
- `discussion` the same prompt with the private Moderator night messages removed,
               i.e. what an external observer sees. The derivation is NOT deducible;
               the final configuration can only be inferred from what players say.

The gap between the two is the diagnostic. If the base model already scores well in
`night`, the finetuning is teaching bookkeeping the model can do, and the headline
comparison should be read accordingly. If it scores near zero in `discussion`, that
is the honest baseline for the voting task, which never shows night actions.

One generation per game, decoding identical to the voting experiments
(temperature 1.0, top_p 0.95, top_k 64), so numbers are comparable.

    python -m src.finetuning.derivation.base_diagnostic \
        --model_name unsloth/gemma-4-E2B-it-unsloth-bnb-4bit \
        --split_path data/processed/jin2024_onuw/sft_derivation_v1/split.json \
        --output_path results/finetuning/base_diagnostic_E2B.jsonl
"""
from __future__ import annotations

import unsloth  # noqa: F401  - must precede transformers imports

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.derivation.build_dataset import build_example  # noqa: E402
from src.utils.model_utils import call_local_model, load_local_model  # noqa: E402

PRIVATE_LINE = re.compile(r"^\[\d+\] Moderator \(to player[\d, player]*\): ")

FINAL_RE = re.compile(r"^-\s*(player\d+)\s*:\s*(\w+)\s*$", re.M)


def strip_night_actions(prompt: str) -> str:
    """Remove the private Moderator night messages from an assembled prompt.

    Everything else - instruction, rules, player list, public transcript - is
    left byte-identical, so the two conditions differ in exactly one thing.
    """
    kept = [line for line in prompt.split("\n") if not PRIVATE_LINE.match(line)]
    return "\n".join(kept)


def parse_final_configuration(text: str) -> dict:
    """Pull the Final configuration block out of a generation.

    Deliberately lenient about surrounding prose and strict about the block: the
    base model has not been trained on this format, and a diagnostic that scored
    zero purely on formatting would measure the wrong thing.
    """
    if not text:
        return {}
    tail = text
    if "Final configuration:" in text:
        tail = text.rsplit("Final configuration:", 1)[1]
    return {m.group(1): m.group(2) for m in FINAL_RE.finditer(tail)}


def score(predicted: dict, gold: dict) -> dict:
    correct = sum(1 for p, r in gold.items() if predicted.get(p) == r)
    return {
        "players_correct": correct,
        "players_total": len(gold),
        "exact_match": correct == len(gold),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Base-model derivation diagnostic.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--split_path", type=str, required=True,
                        help="JSON with a 'val' list of game ids.")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--conditions", type=str, default="night,discussion")
    args = parser.parse_args()

    split = json.loads(Path(args.split_path).read_text(encoding="utf-8"))
    game_ids = split["val"]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print("model      : %s" % args.model_name)
    print("val games  : %d" % len(game_ids))
    print("conditions : %s" % conditions)
    print("decoding   : temperature=1.0 top_p=0.95 top_k=64  (as the voting runs)")

    model_io, model = load_local_model(
        model_name=args.model_name, max_seq_length=args.max_seq_length)

    rows = []
    for condition in conditions:
        for game_id in game_ids:
            example = build_example(game_id)
            prompt = example["prompt"]
            if condition == "discussion":
                prompt = strip_night_actions(prompt)

            text, debug_info = call_local_model(
                model=model,
                model_io=model_io,
                prompt=prompt,
                model_name=args.model_name,
                max_new_tokens=args.max_new_tokens,
                gemma_enable_thinking=True,
                return_debug_info=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
            debug_info = debug_info or {}
            predicted = parse_final_configuration(text)
            result = score(predicted, example["end_roles"])
            out_tokens = debug_info.get("output_token_count")

            # A generation that hit the token cap never finished, and one that
            # finished without a parseable Final configuration block said nothing
            # about the roles. Neither is a wrong answer, and scoring them as 0/5
            # would understate accuracy while hiding a decoding-budget problem.
            # They are recorded, excluded from the accuracy denominator, and
            # reported on their own line.
            non_terminating = bool(out_tokens and out_tokens >= args.max_new_tokens)
            parsed_ok = bool(predicted)
            scored = parsed_ok and not non_terminating

            rows.append({
                "game_id": game_id,
                "condition": condition,
                "model_name": args.model_name,
                "raw_response": text,
                "predicted": predicted,
                "gold": example["end_roles"],
                "parsed_ok": parsed_ok,
                "non_terminating": non_terminating,
                "scored": scored,
                "output_token_count": out_tokens,
                "max_new_tokens": args.max_new_tokens,
                **result,
            })
            if scored:
                note = ""
            elif non_terminating:
                note = "  [NON-TERMINATING, not scored]"
            else:
                note = "  [UNPARSEABLE, not scored]"
            print("  %-11s %s  %d/%d%s"
                  % (condition, game_id, result["players_correct"],
                     result["players_total"], note))

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print("=" * 72)
    print("Accuracy is over SCORED generations only. Non-terminating and")
    print("unparseable rows are reported separately, not counted as wrong.")
    print("=" * 72)
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        if not subset:
            continue
        scored = [r for r in subset if r["scored"]]
        nonterm = [r for r in subset if r["non_terminating"]]
        unparsed = [r for r in subset if not r["parsed_ok"] and not r["non_terminating"]]
        players = sum(r["players_correct"] for r in scored)
        total = sum(r["players_total"] for r in scored)
        exact = sum(1 for r in scored if r["exact_match"])

        print()
        print("%s  (%d generations)" % (condition, len(subset)))
        print("   scored          : %d" % len(scored))
        print("   non-terminating : %d   (hit max_new_tokens=%d)"
              % (len(nonterm), args.max_new_tokens))
        print("   unparseable     : %d   (finished, no Final configuration block)"
              % len(unparsed))
        if scored:
            print("   per-player      : %d/%d = %.1f%%"
                  % (players, total, 100.0 * players / total))
            print("   exact-match     : %d/%d = %.1f%%"
                  % (exact, len(scored), 100.0 * exact / len(scored)))
        else:
            print("   per-player      : n/a - nothing was scorable")
        if nonterm:
            print("   *** raise --max_new_tokens: %d generation(s) never terminated ***"
                  % len(nonterm))
    print()
    print("=" * 72)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
