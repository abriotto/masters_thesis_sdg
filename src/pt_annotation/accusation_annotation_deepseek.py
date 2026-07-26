"""
Runner for annotating accusation targets (werewolf / deception) in ONUW
transcripts using DeepSeek V4 Pro.

Reads .txt transcripts (already preprocessed, with <<ACCUSATION_TO_RESOLVE>>
markers) from two source folders (Ego4D and Youtube), calls DeepSeek once
per transcript with the accusation-annotation prompt, and writes one JSON
output file per transcript, mirroring the input folder structure.

Conventions (API call shape, retry/backoff, resume-by-checking-output) are
carried over from argument_mining_three_call_deepseek.py, trimmed down to a
single-call pipeline since this task has no multi-stage graph schema.

Usage:
    set DEEPSEEK_API_KEY=...
    python run_accusation_annotation.py
    python run_accusation_annotation.py --max-files 5 --dry-run
    python run_accusation_annotation.py --resume
    python run_accusation_annotation.py --overwrite
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_GENERATION_SEED = 42
DEFAULT_MAX_TOKENS = 16384
DEFAULT_MAX_TOKENS_CAP = 32768  # hard ceiling per your 32k limit -- never exceeded, even after length-cap retries
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 20.0

MARKER = "<<ACCUSATION_TO_RESOLVE>>"

ALLOWED_TYPES = {"werewolf", "deception"}

SOURCE_DIRS = ["Ego4D", "Youtube"]

DEFAULT_INPUT_ROOT = Path(
    r"C:\Users\annab\Documents\GitHub\masters_thesis_sdg\data\processed"
    r"\lai2023\accusation_transcripts\ready_for_annotation"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\annab\Documents\GitHub\masters_thesis_sdg\data\processed"
    r"\lai2023\accusation_transcripts\acc_targets"
)
DEFAULT_PROMPT_PATH = Path(
    r"C:\Users\annab\Documents\GitHub\masters_thesis_sdg\src\prompts"
    r"\accusation_annotation.txt"
)


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


def parse_header(text):
    """Pull Source / Session / Game / Players out of the transcript header."""
    def grab(field):
        match = re.search(rf"^{field}:\s*(.+)$", text, flags=re.MULTILINE)
        return match.group(1).strip() if match else None

    players_raw = grab("Players")
    players = (
        [p.strip() for p in players_raw.split(",") if p.strip()]
        if players_raw
        else []
    )
    return {
        "source": grab("Source"),
        "session": grab("Session"),
        "game": grab("Game"),
        "players": players,
    }


def find_marked_line_numbers(text):
    """Line numbers of every line ending in the ACCUSATION_TO_RESOLVE marker."""
    numbers = []
    for line in text.splitlines():
        if MARKER in line:
            match = re.match(r"^\s*\[(\d+)\]", line)
            if match:
                numbers.append(int(match.group(1)))
    return numbers


def validate_items(items, expected_line_numbers):
    """Lightweight sanity checks -- never drops or edits model output,
    only records flags for the record's metadata.

    There is no "scope" field anymore: relevance is just whether
    `relations` is empty (not relevant) or non-empty (relevant). An
    unresolved target is still a relation, just with accused=["UNKNOWN"]
    and requires_review=true on that item.
    """
    flags = []

    if not isinstance(items, list):
        return ["items is not a list"]

    returned_line_numbers = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            flags.append(f"item[{i}] is not an object")
            continue

        line_number = item.get("line_number")
        returned_line_numbers.append(line_number)

        relations = item.get("relations")
        has_unknown_target = False
        if not isinstance(relations, list):
            flags.append(f"item[{i}] (line {line_number}): relations is not a list")
            relations = []
        else:
            for j, relation in enumerate(relations):
                if not isinstance(relation, dict):
                    flags.append(f"item[{i}].relations[{j}] is not an object")
                    continue
                if relation.get("type") not in ALLOWED_TYPES:
                    flags.append(
                        f"item[{i}].relations[{j}] (line {line_number}): "
                        f"invalid type {relation.get('type')!r}"
                    )
                accused = relation.get("accused")
                if not isinstance(accused, list) or not accused:
                    flags.append(
                        f"item[{i}].relations[{j}] (line {line_number}): "
                        f"accused must be a non-empty list"
                    )
                elif "UNKNOWN" in accused:
                    has_unknown_target = True

        requires_review = item.get("requires_review")
        if has_unknown_target and requires_review is not True:
            flags.append(
                f"item[{i}] (line {line_number}): has UNKNOWN accused but requires_review is {requires_review!r}"
            )
        if requires_review is True and not has_unknown_target:
            flags.append(
                f"item[{i}] (line {line_number}): requires_review is True but no UNKNOWN accused found"
            )

    expected_set = set(expected_line_numbers)
    returned_set = set(n for n in returned_line_numbers if n is not None)

    missing = expected_set - returned_set
    extra = returned_set - expected_set
    if missing:
        flags.append(f"missing annotations for marked lines: {sorted(missing)}")
    if extra:
        flags.append(f"annotated lines not marked in transcript: {sorted(extra)}")

    duplicates = [n for n in returned_line_numbers if returned_line_numbers.count(n) > 1]
    if duplicates:
        flags.append(f"duplicate line_number(s) in output: {sorted(set(duplicates))}")

    return flags


# ============================================================
# DeepSeek call (OpenAI-compatible client), mirroring
# argument_mining_three_call_deepseek.py's call_model conventions
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
    """
    Call DeepSeek with thinking mode enabled and return the raw text content.
    Doubles max_tokens on length-cap finish reasons; retries transient errors.

    temperature/top_p are intentionally omitted -- DeepSeek ignores both in
    thinking mode (see argument_mining_three_call_deepseek.py for the source
    of this convention).
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

            return parsed

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
# File discovery, processing, and output
# ============================================================

def discover_transcripts(input_root):
    """Yield (source_dir_name, relative_path, absolute_path) for every .txt
    file under each of SOURCE_DIRS, sorted for a stable run order."""
    for source_dir_name in SOURCE_DIRS:
        source_dir = input_root / source_dir_name
        if not source_dir.is_dir():
            print(f"Warning: source folder not found, skipping: {source_dir}")
            continue
        for path in sorted(source_dir.rglob("*.txt")):
            yield source_dir_name, path.relative_to(source_dir), path


def output_path_for(output_root, source_dir_name, relative_path):
    return (output_root / source_dir_name / relative_path).with_suffix(".json")


def output_is_complete(output_path):
    """An existing output file counts as done only if it has no error, has
    an items list, and isn't missing coverage for any marked line.

    Extra/hallucinated line numbers are noted but don't block resume --
    that's noise you can filter later, not lost data. A genuine coverage
    gap ("missing annotations for marked lines...") means real accusations
    were never annotated, so those files are treated as incomplete and
    will be retried on --resume.
    """
    if not output_path.exists():
        return False
    try:
        record = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    metadata = record.get("metadata", {})
    if "error" in metadata:
        return False
    if not isinstance(record.get("items"), list):
        return False
    validation_flags = metadata.get("validation_flags", [])
    if any(flag.startswith("missing annotations for marked lines") for flag in validation_flags):
        return False
    return True


def process_file(client, source_dir_name, relative_path, absolute_path, prompt_template, model, args):
    text = absolute_path.read_text(encoding="utf-8")
    header = parse_header(text)
    expected_line_numbers = find_marked_line_numbers(text)

    base_metadata = {
        "source_file": str(Path(source_dir_name) / relative_path),
        "source": header["source"] or source_dir_name,
        "session": header["session"],
        "game": header["game"],
        "players": header["players"],
        "model": model,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "prompt_path": str(args.prompt_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_line_numbers": expected_line_numbers,
    }

    user_message = prompt_template.replace("{{TRANSCRIPT}}", text)

    if args.dry_run:
        return {
            "metadata": {**base_metadata, "dry_run": True},
            "items": [],
        }

    try:
        parsed = call_model(
            client,
            model,
            system_prompt="",  # the whole prompt (instructions + transcript) goes in one user message
            user_message=user_message,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )
        items = parsed.get("items", [])
        validation_flags = validate_items(items, expected_line_numbers)
        return {
            "metadata": {**base_metadata, "validation_flags": validation_flags},
            "items": items,
        }
    except Exception as error:
        return {
            "metadata": {
                **base_metadata,
                "error": str(error),
                "error_type": type(error).__name__,
            },
            "items": [],
        }


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_flagged_review_csv(output_root):
    """Scan every output JSON under output_root and collect requires_review
    relations (unresolved-target UNKNOWNs) into one flat CSV."""
    import csv

    rows = []
    for json_path in sorted(output_root.rglob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        metadata = record.get("metadata", {})
        for item in record.get("items", []):
            if not item.get("requires_review"):
                continue
            relations = item.get("relations") or [{}]
            for relation in relations:
                # Strict filter: only rows whose accused is actually UNKNOWN
                # count as review-worthy, in case requires_review was set
                # inconsistently by the model.
                accused = relation.get("accused") or []
                if "UNKNOWN" not in accused:
                    continue
                rows.append({
                    "output_file": str(json_path.relative_to(output_root)),
                    "source": metadata.get("source"),
                    "session": metadata.get("session"),
                    "game": metadata.get("game"),
                    "line_number": item.get("line_number"),
                    "accuser": item.get("accuser"),
                    "type": relation.get("type"),
                    "accused": ", ".join(accused),
                    "evidence": relation.get("evidence"),
                })

    csv_path = output_root / "flagged_for_review.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return csv_path, len(rows)


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate accusation targets (werewolf/deception) in ONUW transcripts via DeepSeek."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry-sleep-seconds", type=float, default=DEFAULT_RETRY_SLEEP_SECONDS)
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N transcripts (for testing).")
    parser.add_argument("--resume", action="store_true", help="Skip transcripts whose output already exists and is valid.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess everything, ignoring existing output.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk files and write stub output (no API calls) to sanity-check paths and prompt substitution.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Use either --resume or --overwrite, not both.")

    if not args.dry_run:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Set the DEEPSEEK_API_KEY environment variable before running.")
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    else:
        client = None

    prompt_template = args.prompt_path.read_text(encoding="utf-8")
    if "{{TRANSCRIPT}}" not in prompt_template:
        raise ValueError(f"Prompt file at {args.prompt_path} has no {{TRANSCRIPT}} placeholder.")

    print(f"Input root  : {args.input_root}")
    print(f"Output root : {args.output_root}")
    print(f"Prompt      : {args.prompt_path}")
    print(f"Model       : {args.model}")
    print(f"Mode        : {'DRY RUN (no API calls)' if args.dry_run else 'live'}")
    print()

    processed = 0
    skipped = 0
    errored = 0
    skipped_no_markers = 0

    for source_dir_name, relative_path, absolute_path in discover_transcripts(args.input_root):
        if args.max_files is not None and processed >= args.max_files:
            print(f"Reached --max-files={args.max_files}. Stopping.")
            break

        out_path = output_path_for(args.output_root, source_dir_name, relative_path)
        label = f"{source_dir_name}/{relative_path}"

        text_preview = absolute_path.read_text(encoding="utf-8")
        if MARKER not in text_preview:
            skipped_no_markers += 1
            continue

        if not args.overwrite and out_path.exists() and output_is_complete(out_path):
            skipped += 1
            print(f"Skipping (already done): {label}")
            continue

        print(f"Processing: {label}")
        record = process_file(
            client, source_dir_name, relative_path, absolute_path,
            prompt_template, args.model, args,
        )
        write_json(out_path, record)

        if "error" in record["metadata"]:
            errored += 1
            print(f"  Error: {record['metadata']['error']}")
        else:
            processed += 1
            flags = record["metadata"].get("validation_flags", [])
            if flags:
                print(f"  Validation flags: {flags}")

    print()
    print(f"Processed        : {processed}")
    print(f"Skipped (resumed): {skipped}")
    print(f"Skipped (no markers in file): {skipped_no_markers}")
    print(f"Errored          : {errored}")

    if not args.dry_run:
        csv_path, row_count = build_flagged_review_csv(args.output_root)
        print(f"\nFlagged-for-review CSV: {csv_path} ({row_count} rows)")


if __name__ == "__main__":
    main()