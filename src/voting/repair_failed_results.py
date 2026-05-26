
"""
Repair/diagnose failed experiment result JSON files.

This version detects explicit error records first, e.g. OutOfMemoryError,
instead of treating them as parser failures.

It reuses generic JSON cleanup from:
    src.utils.experiment_utils.prepare_response_for_json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from src.utils.experiment_utils import prepare_response_for_json, save_json

CIRCLE_VOTE = "Circle Vote"


def parse_json_object(text: Any) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    try:
        obj = json.loads(text)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def validate_voting_output(
    obj: Optional[dict[str, Any]],
    player_names: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "is_valid": False,
        "errors": [],
        "chosen_vote": None,
        "justification": None,
    }

    if obj is None:
        report["errors"].append("no_parseable_json")
        return report

    if not isinstance(obj, dict):
        report["errors"].append("parsed_output_not_dict")
        return report

    chosen_vote = obj.get("chosen_vote")
    justification = obj.get("justification")

    if not isinstance(chosen_vote, str) or not chosen_vote.strip():
        report["errors"].append("chosen_vote_not_nonempty_string")
    else:
        chosen_vote = chosen_vote.strip()
        allowed_votes = set(player_names) | {CIRCLE_VOTE}
        if chosen_vote not in allowed_votes:
            report["errors"].append(
                f"chosen_vote_not_in_player_list_or_circle_vote:{chosen_vote}"
            )

    if not isinstance(justification, str) or not justification.strip():
        report["errors"].append("justification_not_nonempty_string")
    else:
        justification = justification.strip()

    report["chosen_vote"] = chosen_vote
    report["justification"] = justification
    report["is_valid"] = not report["errors"]
    return report


def validate_preliminary_output(obj: Optional[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "is_valid": False,
        "errors": [],
        "justification": None,
        "answer": None,
    }

    if obj is None:
        report["errors"].append("no_parseable_json")
        return report

    if not isinstance(obj, dict):
        report["errors"].append("parsed_output_not_dict")
        return report

    justification = obj.get("justification")
    answer = obj.get("answer", obj.get("final_answer"))

    if not isinstance(justification, str) or not justification.strip():
        report["errors"].append("justification_not_nonempty_string")
    else:
        justification = justification.strip()

    if not isinstance(answer, str) or not answer.strip():
        report["errors"].append("answer_not_nonempty_string")
    else:
        answer = answer.strip()

    report["justification"] = justification
    report["answer"] = answer
    report["is_valid"] = not report["errors"]
    return report


def infer_task(record: dict[str, Any]) -> str:
    if "player_names" in record and "processed_txt_path" in record:
        return "voting"
    if "question_id" in record or "question" in record:
        return "preliminary"
    return "unknown"


def validate_record(
    task: str,
    parsed: Optional[dict[str, Any]],
    record: dict[str, Any],
) -> dict[str, Any]:
    if task == "auto":
        task = infer_task(record)

    if task == "voting":
        player_names = record.get("player_names")
        if not isinstance(player_names, list):
            return {
                "is_valid": False,
                "errors": ["missing_player_names_for_voting_validation"],
                "chosen_vote": None,
                "justification": None,
            }
        return validate_voting_output(parsed, [str(x) for x in player_names])

    if task == "preliminary":
        return validate_preliminary_output(parsed)

    return {
        "is_valid": False,
        "errors": [f"unknown_task:{task}"],
    }


def existing_valid(record: dict[str, Any]) -> bool:
    validation = record.get("validation")
    return isinstance(validation, dict) and validation.get("is_valid") is True


def output_hit_cap(record: dict[str, Any]) -> bool:
    debug = record.get("debug_info") or {}
    max_new_tokens = record.get("max_new_tokens")
    output_token_count = debug.get("output_token_count")

    return (
        isinstance(output_token_count, int)
        and isinstance(max_new_tokens, int)
        and output_token_count >= max_new_tokens
    )


def is_error_record(record: dict[str, Any]) -> bool:
    return "error" in record or "error_type" in record


def classify_error_record(record: dict[str, Any]) -> tuple[str, list[str], bool]:
    error_type = str(record.get("error_type", "UnknownError"))
    error = str(record.get("error", ""))

    reasons = [f"error_record:{error_type}"]

    low_error = error.lower()
    if "out of memory" in low_error or error_type.lower() in {"outofmemoryerror", "cudaoutofmemoryerror"}:
        reasons.append("cuda_oom")
    if "max_new_tokens" in low_error or "trunc" in low_error:
        reasons.append("possible_truncation")

    return "error_record", reasons, True


def classify_after_repair(
    original: dict[str, Any],
    repaired_validation: dict[str, Any],
    parsed_output: Optional[dict[str, Any]],
) -> tuple[str, list[str], bool]:
    reasons: list[str] = []

    if existing_valid(original):
        return "already_valid", reasons, False

    if output_hit_cap(original):
        reasons.append("output_token_count_hit_max_new_tokens")

    if parsed_output is None:
        reasons.append("no_parseable_json_after_repair")
        return "needs_rerun", reasons, True

    if repaired_validation.get("is_valid"):
        if reasons:
            reasons.append("valid_but_possible_truncation")
        return "repaired_valid", reasons, False

    errors = repaired_validation.get("errors") or []
    reasons.extend(str(x) for x in errors)
    return "invalid_schema", reasons, True


def add_repair_metadata(
    record: dict[str, Any],
    status: str,
    reasons: list[str],
    response_for_parsing: Any = None,
    parsed_output: Optional[dict[str, Any]] = None,
    validation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    updated = dict(record)

    if response_for_parsing is not None:
        updated["response_for_parsing"] = response_for_parsing
    if parsed_output is not None or "parsed_output" in updated:
        updated["parsed_output"] = parsed_output
    if validation is not None:
        updated["validation"] = validation

    repair_info = dict(updated.get("repair_info") or {})
    repair_info.update(
        {
            "repair_attempted": True,
            "repair_status": status,
            "repair_reasons": reasons,
        }
    )
    updated["repair_info"] = repair_info

    soft_warnings = list(updated.get("soft_warnings") or [])

    if response_for_parsing is not None and response_for_parsing != record.get("raw_response"):
        if "response_was_cleaned_before_parsing" not in soft_warnings:
            soft_warnings.append("response_was_cleaned_before_parsing")

    if output_hit_cap(record) and "output_token_count_hit_max_new_tokens" not in soft_warnings:
        soft_warnings.append("output_token_count_hit_max_new_tokens")

    if "cuda_oom" in reasons and "cuda_oom" not in soft_warnings:
        soft_warnings.append("cuda_oom")

    updated["soft_warnings"] = soft_warnings
    return updated


def chosen_or_answer(validation: Any) -> Optional[str]:
    if not isinstance(validation, dict):
        return None

    for key in ("chosen_vote", "answer", "final_answer"):
        value = validation.get(key)
        if isinstance(value, str):
            return value

    return None


def scan_file(path: Path, task: str, write: bool) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path),
            "status": "unreadable_json_file",
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "needs_rerun": True,
            "repaired": False,
        }

    if existing_valid(record):
        return {
            "path": str(path),
            "status": "already_valid",
            "reasons": [],
            "needs_rerun": False,
            "repaired": False,
            "chosen_or_answer": chosen_or_answer(record.get("validation")),
        }

    if is_error_record(record):
        status, reasons, needs_rerun = classify_error_record(record)
        if write:
            updated = add_repair_metadata(record, status=status, reasons=reasons)
            save_json(path, updated)
        return {
            "path": str(path),
            "status": status,
            "reasons": reasons,
            "needs_rerun": needs_rerun,
            "repaired": False,
            "chosen_or_answer": None,
        }

    raw_response = record.get("raw_response")
    response_for_parsing = prepare_response_for_json(raw_response)
    parsed_output = parse_json_object(response_for_parsing)
    validation = validate_record(task=task, parsed=parsed_output, record=record)

    status, reasons, needs_rerun = classify_after_repair(record, validation, parsed_output)
    repaired = status == "repaired_valid"

    if write and (repaired or needs_rerun):
        updated = add_repair_metadata(
            record=record,
            status=status,
            reasons=reasons,
            response_for_parsing=response_for_parsing,
            parsed_output=parsed_output,
            validation=validation,
        )
        save_json(path, updated)

    return {
        "path": str(path),
        "status": status,
        "reasons": reasons,
        "needs_rerun": needs_rerun,
        "repaired": repaired,
        "chosen_or_answer": chosen_or_answer(validation),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_root",
        type=Path,
        required=True,
        help="Root folder containing result JSON files, e.g. results/voting/<model>/prompt_v3.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["auto", "voting", "preliminary"],
        default="auto",
        help="Validation schema to use.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually update JSON files with repaired parsed_output/validation/repair_info.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--rerun_list_path",
        type=Path,
        default=None,
        help="Optional text file listing paths that should be rerun.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(args.results_root.rglob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found under {args.results_root}")

    summaries = [scan_file(path, task=args.task, write=args.write) for path in files]

    counts: dict[str, int] = {}
    for item in summaries:
        status = item["status"]
        counts[status] = counts.get(status, 0) + 1

    needs_rerun = [x for x in summaries if x["needs_rerun"]]
    repaired = [x for x in summaries if x["repaired"]]

    print("=== Repair scan summary ===")
    print(f"results_root: {args.results_root}")
    print(f"task: {args.task}")
    print(f"write: {args.write}")
    print(f"total_files: {len(files)}")
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"repaired_valid: {len(repaired)}")
    print(f"needs_rerun: {len(needs_rerun)}")

    if needs_rerun:
        print("\nFiles to rerun:")
        for item in needs_rerun:
            print(f"- {item['path']} | reasons={item['reasons']}")

    report = {
        "results_root": str(args.results_root),
        "task": args.task,
        "write": args.write,
        "total_files": len(files),
        "counts": counts,
        "repaired_count": len(repaired),
        "needs_rerun_count": len(needs_rerun),
        "items": summaries,
    }

    if args.report_path is not None:
        save_json(args.report_path, report)
        print(f"\nWrote report: {args.report_path}")

    if args.rerun_list_path is not None:
        args.rerun_list_path.parent.mkdir(parents=True, exist_ok=True)
        args.rerun_list_path.write_text(
            "\n".join(item["path"] for item in needs_rerun)
            + ("\n" if needs_rerun else ""),
            encoding="utf-8",
        )
        print(f"Wrote rerun list: {args.rerun_list_path}")


if __name__ == "__main__":
    main()
