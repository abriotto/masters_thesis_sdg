"""Show exactly which tokens the loss is computed over for one training example.

CPU only - loads the tokenizer, never the weights.

    python -m src.finetuning.derivation.inspect_loss_span \
        --model_name unsloth/gemma-4-E2B-it --game_id episode_002

The point of this rebuild is that the loss covers the WHOLE completion, starting at
"Night actions, in call order:", not just the final configuration. The previous run
supervised ~30 answer tokens after an unsupervised thought block. So the check to
make here is: does the supervised span begin at the first token of the derivation?

The Dealt cards block is NOT in the completion. It is given in the prompt, and
supervising a verbatim copy of prompt text would train copying rather than
derivation.

With `--response_part '<|turn>model\\n'` (the trainer's default RESPONSE_PART) the
answer is yes: train_on_responses_only masks everything up to and including the
assistant turn marker, and supervises from there to the end of the turn. With the
previous run's `--response_part '<channel|>'` the answer would be no.

Without a tokenizer this prints the marker-level span instead, which shows the
same boundary but cannot show token ids or counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.derivation.build_dataset import build_example  # noqa: E402

# Must match train_role_inference.py.
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"


def marker_level(example, response_part):
    """Boundary illustration that needs no tokenizer."""
    prompt, completion = example["prompt"], example["completion"]
    print("=" * 78)
    print("LOSS SPAN (marker level - no tokenizer available)")
    print("=" * 78)
    print("response_part (loss anchor): %r" % response_part)
    print()
    print("--- MASKED: prompt, %d chars, labels = -100 -----------------------" % len(prompt))
    print("    ...first 200 chars...")
    print("    " + prompt[:200].replace("\n", "\n    "))
    print("    ...")
    print("    ...last 200 chars...")
    print("    " + prompt[-200:].replace("\n", "\n    "))
    print()
    print("--- %s   <-- loss starts immediately after this marker" % response_part.strip())
    print()
    print("--- SUPERVISED: completion, %d chars, labels = token ids ----------"
          % len(completion))
    print("    " + completion.replace("\n", "\n    "))
    print()
    print("FIRST SUPERVISED TEXT: %r" % completion[:40])
    print("LAST  SUPERVISED TEXT: %r" % completion[-40:])
    starts_at_derivation = completion.lstrip().startswith("Night actions, in call order:")
    print()
    print("loss begins at the derivation, not the answer: %s" % starts_at_derivation)
    if not starts_at_derivation:
        print("  *** WRONG: the supervised span does not start at 'Night actions, in call order:'")
    return 0 if starts_at_derivation else 1


def token_level(example, model_name, response_part):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    anchor = tokenizer(response_part, add_special_tokens=False)["input_ids"]
    start = None
    for i in range(len(ids) - len(anchor) + 1):
        if ids[i:i + len(anchor)] == anchor:
            start = i + len(anchor)
    if start is None:
        print("response_part %r not found in the rendered template." % response_part)
        print("Rendered tail: %r" % text[-400:])
        return 1

    labels = [-100] * start + ids[start:]
    supervised = [t for t, lab in zip(ids, labels) if lab != -100]

    print("=" * 78)
    print("LOSS SPAN (token level) - %s" % model_name)
    print("=" * 78)
    print("total tokens      : %d" % len(ids))
    print("masked (prompt)   : %d" % start)
    print("supervised        : %d" % len(supervised))
    print("response_part     : %r" % response_part)
    print()
    print("--- last 12 MASKED tokens (end of prompt) ---")
    for i in range(max(0, start - 12), start):
        print("  %5d  label=-100   %r" % (ids[i], tokenizer.decode([ids[i]])))
    print()
    print("--- first 24 SUPERVISED tokens (start of the loss span) ---")
    for i in range(start, min(len(ids), start + 24)):
        print("  %5d  label=%-6d %r" % (ids[i], ids[i], tokenizer.decode([ids[i]])))
    print()
    decoded = tokenizer.decode(supervised)
    print("decoded supervised span, first 120 chars:")
    print("  %r" % decoded[:120])
    print("decoded supervised span, last 120 chars:")
    print("  %r" % decoded[-120:])
    print()
    ok = decoded.lstrip().startswith("Night actions, in call order:")
    print("loss begins at the derivation, not the answer: %s" % ok)
    if not ok:
        print("  *** WRONG: the supervised span does not start at 'Night actions, in call order:'")
        print("  *** With --response_part '<channel|>' this is what you would see.")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the supervised token span.")
    parser.add_argument("--game_id", type=str, default="episode_002")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Tokenizer to load. Omit for the marker-level view.")
    parser.add_argument("--response_part", type=str, default=RESPONSE_PART)
    args = parser.parse_args()

    example = build_example(args.game_id)
    if args.model_name:
        return token_level(example, args.model_name, args.response_part)
    return marker_level(example, args.response_part)


if __name__ == "__main__":
    raise SystemExit(main())
