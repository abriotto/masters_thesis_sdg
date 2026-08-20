"""
Runner for annotating LLM vote justifications with the epistemic-basis scheme
defined in src/prompts/justification_annotation.txt, using DeepSeek V4 Pro --
the same model, decoding settings and call conventions used for the Lai corpus
annotations, so that the two annotation layers are comparable.

The prompt is the authority for the scheme. JUSTIFICATION_CODEBOOK.md is an
earlier draft and is out of date; the category set here (seven categories
including Payoff, plus a per-sentence rule_mentioned flag) follows the prompt.

Reads the pilot sample produced by sample_justification_pilot.py (one JSON
object per line: vote plus pre-split sentences) and calls DeepSeek once per
justification. Output is one JSONL file, one line per justification, with the
model output plus validation flags.

Two things differ from accusation_annotation_deepseek.py:

  * The prompt has no {{TRANSCRIPT}} placeholder. It defines an input schema
    instead, so the instructions go in as the system message and the
    {vote, sentences} object as the user message.
  * Validation is much stricter, because this prompt has never been run.
    Sentence coverage, verbatim text preservation, the category and use
    vocabularies, and evidence_span substring-exactness are all checked. The
    pilot's job is to surface where the prompt fails, so failures are
    recorded rather than silently repaired.

Usage:
    set DEEPSEEK_API_KEY=...
    python src/pt_annotation/justification_annotation_deepseek.py --dry-run
    python src/pt_annotation/justification_annotation_deepseek.py
    python src/pt_annotation/justification_annotation_deepseek.py --resume
    python src/pt_annotation/justification_annotation_deepseek.py --max-items 3
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# Configuration
# ============================================================

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_GENERATION_SEED = 42

# Deliberately generous, and deliberately starting at the ceiling rather than
# ramping up to it. High-effort thinking tokens count toward max_tokens, and
# a length-truncated response is a wasted call at full price -- the accusation
# run learned this the expensive way. A justification is 3-5 sentences and the
# output JSON is small, so the headroom costs nothing when it is not used:
# billing is on tokens produced, not on the cap requested.
DEFAULT_MAX_TOKENS = 32768
DEFAULT_MAX_TOKENS_CAP = 32768

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 20.0

ALLOWED_CATEGORIES = {
    "Deduction", "Consistency", "Payoff",
    "Testimony", "Social", "Behavioral", "Other",
}
ALLOWED_USES = {"used", "discounted", "mentioned"}

DEFAULT_PILOT_DIR = REPO_ROOT / "data" / "processed" / "justification_annotations" / "pilot_v1"
DEFAULT_INPUT_PATH = DEFAULT_PILOT_DIR / "pilot_sample.jsonl"
DEFAULT_OUTPUT_PATH = DEFAULT_PILOT_DIR / "pilot_annotations.jsonl"
DEFAULT_PROMPT_PATH = REPO_ROOT / "src" / "prompts" / "justification_annotation.txt"


# ============================================================
# Small helpers
# ============================================================

def parse_json_response(text):
    """Strip markdown code fences if present, then parse JSON."""
    if text is None:
        raise ValueError("Model response content is None")
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalise_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


# ============================================================
# Validation
#
# Nothing here edits the model output. Flags are recorded so the pilot can
# report how often the prompt is followed, which is the point of running it.
# ============================================================

def validate_annotation(parsed, item):
    flags = []

    expected_sentences = {s["sentence_id"]: s["text"] for s in item["sentences"]}

    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]

    if normalise_whitespace(parsed.get("vote", "")) != normalise_whitespace(item["vote"]):
        flags.append(
            f"vote not preserved: expected {item['vote']!r}, got {parsed.get('vote')!r}"
        )

    sentences = parsed.get("sentences")
    if not isinstance(sentences, list):
        return flags + ["sentences is not a list"]

    returned_ids = []

    for index, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            flags.append(f"sentences[{index}] is not an object")
            continue

        sentence_id = sentence.get("sentence_id")
        returned_ids.append(sentence_id)
        label = f"sentence {sentence_id}"

        expected_text = expected_sentences.get(sentence_id)
        returned_text = sentence.get("text")

        if expected_text is None:
            flags.append(f"{label}: sentence_id not in the input")
        elif normalise_whitespace(returned_text) != normalise_whitespace(expected_text):
            flags.append(f"{label}: text was altered")

        if not isinstance(sentence.get("rule_mentioned"), bool):
            flags.append(
                f"{label}: rule_mentioned is {sentence.get('rule_mentioned')!r}, not a boolean"
            )

        annotations = sentence.get("annotations")
        if not isinstance(annotations, list):
            flags.append(f"{label}: annotations is not a list")
            continue

        seen = set()
        for position, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                flags.append(f"{label}: annotations[{position}] is not an object")
                continue

            category = annotation.get("category")
            use = annotation.get("use")
            span = annotation.get("evidence_span")
            description = annotation.get("other_description")

            if category not in ALLOWED_CATEGORIES:
                flags.append(f"{label}: invalid category {category!r}")
            if use not in ALLOWED_USES:
                flags.append(f"{label}: invalid use {use!r}")

            # evidence_span exactness is the load-bearing check. If spans are
            # paraphrased they cannot be aligned to the text, and every
            # downstream span-level measure becomes unusable.
            if not isinstance(span, str) or not span.strip():
                flags.append(f"{label}: missing evidence_span for {category!r}")
            elif expected_text is not None and span not in expected_text:
                if normalise_whitespace(span) in normalise_whitespace(expected_text):
                    flags.append(
                        f"{label}: evidence_span matches only after whitespace "
                        f"normalisation ({category!r})"
                    )
                else:
                    flags.append(
                        f"{label}: evidence_span is not a substring of the sentence "
                        f"({category!r}): {span!r}"
                    )

            if category == "Other":
                if not description:
                    flags.append(f"{label}: Other without other_description")
            elif description not in (None, ""):
                flags.append(
                    f"{label}: other_description set on non-Other category {category!r}"
                )

            key = (category, use, span)
            if key in seen:
                flags.append(f"{label}: duplicate annotation {category!r}/{use!r}")
            seen.add(key)

    expected_ids = set(expected_sentences)
    returned_set = {i for i in returned_ids if i is not None}

    missing = expected_ids - returned_set
    extra = returned_set - expected_ids
    if missing:
        flags.append(f"missing sentences: {sorted(missing)}")
    if extra:
        flags.append(f"sentences not in the input: {sorted(extra)}")

    duplicates = sorted({i for i in returned_ids if returned_ids.count(i) > 1})
    if duplicates:
        flags.append(f"duplicate sentence_id(s): {duplicates}")

    return flags


# ============================================================
# DeepSeek call
# ============================================================

def is_retryable_error(error):
    text = str(error).lower()
    markers = [
        "429", "rate limit", "500", "502", "503", "504",
        "deadline", "timeout", "temporarily", "overloaded",
    ]
    return any(marker in text for marker in markers)


class RetryableModelOutputError(RuntimeError):
    def __init__(self, message, *, length_limited=False):
        super().__init__(message)
        self.length_limited = length_limited


def call_model(client, model, system_prompt, user_message, *, max_retries, retry_sleep_seconds):
    """Call DeepSeek with thinking enabled and return the parsed JSON.

    temperature/top_p are intentionally omitted -- DeepSeek ignores both in
    thinking mode. This mirrors accusation_annotation_deepseek.py so the two
    annotation layers were produced under identical decoding settings.
    """
    current_max_tokens = DEFAULT_MAX_TOKENS
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                seed=DEFAULT_GENERATION_SEED,
                max_tokens=current_max_tokens,
                response_format={"type": "json_object"},
                reasoning_effort=DEFAULT_REASONING_EFFORT,
                extra_body={"thinking": {"type": "enabled"}},
            )

            if not response.choices:
                raise RetryableModelOutputError("API response contained no choices")

            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
            length_limited = finish_reason == "length"

            if not content.strip():
                raise RetryableModelOutputError(
                    f"empty model content (finish_reason={finish_reason!r}, "
                    f"max_tokens={current_max_tokens})",
                    length_limited=length_limited,
                )

            try:
                parsed = parse_json_response(content)
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                preview = content[:300].replace("\n", "\\n")
                raise RetryableModelOutputError(
                    f"invalid JSON (finish_reason={finish_reason!r}, "
                    f"preview={preview!r}): {error}",
                    length_limited=length_limited,
                ) from error

            usage = getattr(response, "usage", None)
            usage_record = None
            if usage is not None:
                usage_record = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                details = getattr(usage, "completion_tokens_details", None)
                if details is not None:
                    usage_record["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)

            return parsed, {
                "finish_reason": finish_reason,
                "max_tokens_requested": current_max_tokens,
                "usage": usage_record,
            }

        except Exception as error:
            last_error = error
            retryable = isinstance(error, RetryableModelOutputError) or is_retryable_error(error)

            if attempt >= max_retries or not retryable:
                raise

            if (
                isinstance(error, RetryableModelOutputError)
                and error.length_limited
                and current_max_tokens < DEFAULT_MAX_TOKENS_CAP
            ):
                current_max_tokens = min(current_max_tokens * 2, DEFAULT_MAX_TOKENS_CAP)
                print(f"    Increasing max_tokens -> {current_max_tokens}")

            print(f"    Retrying after attempt {attempt + 1}/{max_retries + 1}: {error}")
            time.sleep(retry_sleep_seconds)

    raise last_error


# ============================================================
# Processing
# ============================================================

def build_user_message(item):
    """Exactly the input schema the prompt documents: vote plus sentences.

    Nothing else is sent. The transcript, the model that wrote the
    justification, and whether the vote was correct are all withheld. The
    prompt's closing rules forbid assessing correctness and reconstructing
    information from the original transcript, and withholding the material is
    a stronger guarantee than instructing against using it.
    """
    payload = {
        "vote": item["vote"],
        "sentences": [
            {"sentence_id": sentence["sentence_id"], "text": sentence["text"]}
            for sentence in item["sentences"]
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def process_item(client, item, system_prompt, model, args):
    base_metadata = {
        "justification_id": item["justification_id"],
        "model_under_annotation": item["model"],
        "game_id": item["game_id"],
        "run_label": item["run_label"],
        "is_correct": item["is_correct"],
        "n_sentences": len(item["sentences"]),
        "annotator_model": model,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "seed": DEFAULT_GENERATION_SEED,
        "prompt_path": str(args.prompt_path.relative_to(REPO_ROOT)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    user_message = build_user_message(item)

    if args.dry_run:
        return {
            "metadata": {**base_metadata, "dry_run": True},
            "input": json.loads(user_message),
            "annotation": None,
        }

    try:
        parsed, call_info = call_model(
            client,
            model,
            system_prompt=system_prompt,
            user_message=user_message,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )
        validation_flags = validate_annotation(parsed, item)
        return {
            "metadata": {
                **base_metadata,
                **call_info,
                "validation_flags": validation_flags,
            },
            "annotation": parsed,
        }
    except Exception as error:
        return {
            "metadata": {
                **base_metadata,
                "error": str(error),
                "error_type": type(error).__name__,
            },
            "annotation": None,
        }


def completed_ids(output_path):
    """Ids already annotated without error, for --resume."""
    if not output_path.exists():
        return set()

    done = set()
    for record in read_jsonl(output_path):
        metadata = record.get("metadata", {})
        if "error" in metadata or metadata.get("dry_run"):
            continue
        if record.get("annotation") is None:
            continue
        done.add(metadata.get("justification_id"))
    return done


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate LLM vote justifications with the epistemic-basis scheme via DeepSeek."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-sleep-seconds", type=float, default=DEFAULT_RETRY_SLEEP_SECONDS)
    parser.add_argument("--max-items", type=int, default=None, help="Annotate at most N justifications.")
    parser.add_argument("--resume", action="store_true", help="Skip justifications already annotated.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the payloads without calling the API.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.dry_run:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Set the DEEPSEEK_API_KEY environment variable before running.")
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    else:
        client = None

    system_prompt = args.prompt_path.read_text(encoding="utf-8")
    items = read_jsonl(args.input_path)

    already_done = completed_ids(args.output_path) if args.resume else set()

    print(f"Input      : {args.input_path}")
    print(f"Output     : {args.output_path}")
    print(f"Prompt     : {args.prompt_path}")
    print(f"Model      : {args.model} (effort={DEFAULT_REASONING_EFFORT}, "
          f"seed={DEFAULT_GENERATION_SEED}, max_tokens={DEFAULT_MAX_TOKENS})")
    print(f"Mode       : {'DRY RUN (no API calls)' if args.dry_run else 'live'}")
    print(f"Items      : {len(items)}"
          + (f", {len(already_done)} already annotated" if args.resume else ""))
    print()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if (args.resume and args.output_path.exists()) else "w"

    processed = skipped = errored = flagged = 0

    with args.output_path.open(write_mode, encoding="utf-8") as handle:
        for item in items:
            if args.max_items is not None and processed >= args.max_items:
                print(f"Reached --max-items={args.max_items}. Stopping.")
                break

            justification_id = item["justification_id"]

            if justification_id in already_done:
                skipped += 1
                continue

            print(f"Annotating: {justification_id} ({len(item['sentences'])} sentences)")
            record = process_item(client, item, system_prompt, args.model, args)

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            metadata = record["metadata"]
            if "error" in metadata:
                errored += 1
                print(f"  Error: {metadata['error']}")
            else:
                processed += 1
                flags = metadata.get("validation_flags", [])
                if flags:
                    flagged += 1
                    for flag in flags:
                        print(f"  FLAG: {flag}")

    print()
    print(f"Annotated          : {processed}")
    print(f"Skipped (resumed)  : {skipped}")
    print(f"Errored            : {errored}")
    print(f"With validation flags: {flagged}")


if __name__ == "__main__":
    main()
