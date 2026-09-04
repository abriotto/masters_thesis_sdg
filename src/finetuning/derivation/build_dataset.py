"""Assemble prompts and emit the derivation-trace SFT dataset.

Spec step 6.

    python -m src.finetuning.derivation.build_dataset --print_example episode_002

Prompt contents
---------------
instruction (role_inference_prompt_v3.txt) + ONUW rules + player list + the full
game transcript INCLUDING the private Moderator night messages, each tagged with
its recipient.

The private Moderator messages are the point of the design. The derivation names
who was dealt what and which cards moved; those facts are only present in the
input because the private confirmations are there. Reusing preprocess_jin.py,
which drops every private message, would have produced a target that asserts
facts the prompt does not contain - the failure mode this rebuild exists to fix.

Excluded from every prompt, without exception:
- `thought`, `belief`, `strategy`: never read by this module. The corpus JSON is
  the only source and only `agent_name`, `content`, `visible_to` and `turn` are
  touched, so the exclusion is structural rather than a filter that could be
  mis-specified.
- `index.json` and its `strategies` field: not opened. Everything is derived from
  the raw episodes.
- The final "Game over ..." line: it is public but names the eliminated player and
  their role, i.e. it is the label.
- Private PLAYER messages: the players' own night-action requests and vote
  declarations. Those are model-generated statements of intent, inconsistently
  phrased, and the spec takes choices from the Moderator's confirmation instead.

Completion and masking
----------------------
completion = derivation + "\\n\\n" + answer. `derivation` is the Night actions
section; `answer` is the Final configuration section. NOTHING IS MASKED WITHIN THE
COMPLETION: the loss covers all of it, starting at "Night actions, in call order:".
Only the prompt is masked. That is the change from the previous run, where the loss
covered the ~30-token answer alone.

The Dealt cards block is deliberately NOT in the target. It is given in the prompt,
so supervising it would train verbatim copying and dilute the signal across the
tokens that actually carry the derivation. The full three-section trace is kept in
`full_trace` for display and for the thesis appendix.

The trainer's loss anchor must therefore be the assistant turn marker, not the
thought-channel delimiter:

    --response_part '<|turn>model\\n'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.derivation.gate import game_ids, run_gate  # noqa: E402
from src.finetuning.derivation.records import (  # noqa: E402
    GAME_OVER_MARKER,
    build_record,
    load_game,
    recipients,
)
from src.finetuning.derivation.render import render_derivation  # noqa: E402

INSTRUCTION_PATH = REPO_ROOT / "src" / "prompts" / "role_inference_prompt_v3.txt"
RULES_PATH = REPO_ROOT / "src" / "prompts" / "onuw_rules_v2.txt"
DEFAULT_OUT = REPO_ROOT / "data" / "processed" / "jin2024_onuw" / "sft_derivation_v1"

ANSWER_MARKER = "\n\nFinal configuration:\n"
NIGHT_MARKER = "\n\nNight actions, in call order:\n"


def build_derivation_prompt(instruction, rules, players, deal_block, transcript):
    """Assemble the prompt.

    This mirrors src.utils.prompt_utils.build_full_prompt but adds the "## Initial
    deal" section. That helper is NOT extended, because it is shared with the
    voting prompt and the base arm has already been run against it; changing it
    would invalidate those results. The two layouts are otherwise identical.
    """
    return """{instruction}

Here are the game rules:

{rules}

## Player list

{players}

## Initial deal

{deal}

## Transcript

{transcript}
""".format(instruction=instruction, rules=rules, players=", ".join(players),
           deal=deal_block, transcript=transcript).strip()


def render_deal_block(record) -> str:
    """The cards as dealt, plus the centre.

    Present so that the derivation's "Dealt cards:" section is entailed by the
    prompt rather than inferred. Villagers never wake, so without this the model
    would have to read a player's dealt Villager card off the ABSENCE of a
    Moderator message, and tell that apart from the same role sitting in the
    centre. Absence of evidence is not entailment, and entailment is the point.

    Rendered in the same order and format as the derivation's own Dealt cards
    block, so the correspondence is exact.
    """
    lines = ["- %s: %s" % (p, record.dealt[p]) for p in record.players]
    lines.append("- Centre: %s" % ", ".join(record.centre))
    return "\n".join(lines)


def render_transcript(game: dict) -> str:
    """The full game transcript, public plus private Moderator night messages.

    Only `agent_name`, `content`, `visible_to` and `turn` are read.
    """
    lines = []
    for index, message in enumerate(game["messages"], start=1):
        speaker = message["agent_name"]
        content = message["content"]
        visible = message["visible_to"]

        if content.startswith(GAME_OVER_MARKER):
            continue

        if visible == "all":
            lines.append("[%d] %s: %s" % (index, speaker, content))
            continue

        # Private. Keep the Moderator's night confirmations, drop players' own
        # private statements of intent and their private vote declarations.
        if speaker != "Moderator":
            continue
        to = ", ".join(recipients(visible))
        lines.append("[%d] %s (to %s): %s" % (index, speaker, to, content))
    return "\n".join(lines)


def build_example(game_id: str) -> dict:
    game = load_game(game_id)
    record = build_record(game, game_id)

    # The full three-section trace is kept for display and for the thesis
    # appendix. The TARGET is only the last two sections.
    full_trace = render_derivation(record)
    if ANSWER_MARKER not in full_trace or NIGHT_MARKER not in full_trace:
        raise ValueError("%s: rendered trace is missing a section" % game_id)

    head, tail = full_trace.split(ANSWER_MARKER, 1)
    answer = "Final configuration:\n" + tail

    # Drop the Dealt cards block from the completion. It is verbatim prompt text,
    # so supervising it trains copying rather than derivation and dilutes the
    # signal across the tokens that do carry it.
    dealt_section, night_body = head.split(NIGHT_MARKER, 1)
    derivation = "Night actions, in call order:\n" + night_body
    completion = derivation + "\n\n" + answer

    if dealt_section + "\n\n" + completion != full_trace:
        raise ValueError("%s: completion does not reconstruct the rendered trace"
                         % game_id)

    instruction = INSTRUCTION_PATH.read_text(encoding="utf-8").strip()
    rules = RULES_PATH.read_text(encoding="utf-8").strip()
    deal_block = render_deal_block(record)
    prompt = build_derivation_prompt(
        instruction=instruction,
        rules=rules,
        players=record.players,
        deal_block=deal_block,
        transcript=render_transcript(game),
    )

    # The deal must be in the prompt, so the night actions are entailed rather
    # than inferred, and must NOT be in the target, so nothing is supervised that
    # is copyable from the input. Assert both rather than trust them.
    if deal_block not in prompt:
        raise ValueError("%s: the deal block is missing from the prompt" % game_id)
    if deal_block in completion:
        raise ValueError("%s: the deal block leaked into the supervised completion"
                         % game_id)

    end_roles = {p: record.final[p] for p in record.players}
    return {
        "game_id": game_id,
        "source": "jin2024",
        "player_names": list(record.players),
        "dealt": dict(record.dealt),
        "centre": list(record.centre),
        "end_roles": end_roles,
        "num_werewolves_end": sum(1 for r in end_roles.values() if r == "Werewolf"),
        "prompt": prompt,
        "derivation": derivation,
        "answer": answer,
        "full_trace": full_trace,
        "completion": completion,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "prompt_chars": len(prompt),
        "completion_chars": len(completion),
    }


def build_all(ids=None) -> list:
    return [build_example(gid) for gid in (ids or game_ids())]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the derivation SFT dataset.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--print_example", type=str, default=None,
                        help="Print one example verbatim and exit without writing.")
    parser.add_argument("--write", action="store_true",
                        help="Write all.jsonl. The split is step 7, not this script.")
    args = parser.parse_args()

    gate = run_gate()
    if gate["failed"]:
        print("REFUSING TO BUILD: %d game(s) fail the correctness gate: %s"
              % (len(gate["failed"]), gate["failed"]))
        return 1

    if args.print_example:
        example = build_example(args.print_example)
        print("=" * 78)
        print("PROMPT  (%d chars)" % example["prompt_chars"])
        print("=" * 78)
        print(example["prompt"])
        print()
        print("=" * 78)
        print("COMPLETION  (%d chars)  - loss covers ALL of this" % example["completion_chars"])
        print("=" * 78)
        print(example["completion"])
        return 0

    examples = build_all()
    print("built %d examples (all %d games passed the gate)"
          % (len(examples), len(gate["passed"])))

    if args.write:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "all.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
        print("wrote %s" % path)
    return 0


__all__ = ["build_all", "build_example", "render_transcript"]


if __name__ == "__main__":
    raise SystemExit(main())
