from __future__ import annotations

import unsloth

import argparse
import traceback
from typing import Any, Optional

from src.utils.experiment_utils import (
    add_common_soft_warnings,
    build_results_root,
    count_sentences,
    get_internal_thoughts,
    prepare_response_for_json,
    print_generation_debug,
    print_local_model_summary,
    remove_internal_thoughts_from_debug,
    save_json,
    select_rows,
)
from src.utils.io_utils import find_repo_root, load_json, load_text
from src.utils.json_utils import parse_model_json
from src.utils.model_utils import call_local_model, load_local_model


def build_preliminary_prompt(eval_prompt: str, rules_text: str, question_text: str) -> str:
    return f"""{eval_prompt}

Here are the game rules:

{rules_text}

Now answer the following question.

Question:
{question_text}
""".strip()


def normalize_preliminary_output(
    obj: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Normalize legacy `final_answer` to the preferred `answer` field."""
    if not isinstance(obj, dict):
        return obj

    normalized = dict(obj)
    if "answer" not in normalized and "final_answer" in normalized:
        normalized["answer"] = normalized["final_answer"]
    return normalized


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

    if "justification" not in obj:
        report["errors"].append("missing_justification")
    if "answer" not in obj and "final_answer" not in obj:
        report["errors"].append("missing_answer")

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


def add_preliminary_soft_warnings(
    validation: dict[str, Any],
    raw_response: Any,
    response_for_parsing: Any,
    parsed_output: Optional[dict[str, Any]],
    debug_info: Optional[dict[str, Any]],
    max_new_tokens: int,
) -> list[str]:
    warnings = add_common_soft_warnings(
        raw_response=raw_response,
        response_for_parsing=response_for_parsing,
        parsed_output=parsed_output,
        debug_info=debug_info,
        max_new_tokens=max_new_tokens,
    )

    answer = validation.get("answer")
    justification = validation.get("justification")

    if isinstance(answer, str) and len(answer.strip()) > 80:
        warnings.append("answer_may_be_too_long")

    if isinstance(justification, str):
        sentence_count = count_sentences(justification)
        if sentence_count < 1:
            warnings.append("justification_may_be_empty_or_sentence_fragment")
        if sentence_count > 2:
            warnings.append("justification_may_exceed_2_sentences")

    if isinstance(parsed_output, dict):
        expected_fields = {"justification", "answer"}
        allowed_legacy_fields = {"final_answer"}
        unexpected_fields = sorted(
            set(parsed_output.keys()) - expected_fields - allowed_legacy_fields
        )
        if unexpected_fields:
            warnings.append(f"response_contains_unexpected_fields:{unexpected_fields}")
        if "final_answer" in parsed_output and "answer" not in parsed_output:
            warnings.append("response_uses_legacy_final_answer_field")

    return warnings


def filter_questions(
    rows: list[dict[str, Any]], question_ids: Optional[list[str]]
) -> list[dict[str, Any]]:
    if not question_ids:
        return rows

    wanted = set(question_ids)
    filtered = [row for row in rows if row.get("id") in wanted]
    found = {row.get("id") for row in filtered}
    missing = wanted - found

    if missing:
        raise ValueError(f"Question IDs not found in question file: {sorted(missing)}")

    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preliminary ONUW rule-understanding evals with local Unsloth models."
    )

    parser.add_argument(
        "--questions_path",
        type=str,
        default="src/preliminary_eval/onuw_questions.json",
    )
    parser.add_argument("--eval_prompt_path", type=str, required=True)
    parser.add_argument("--rules_path", type=str, default="src/prompts/onuw_rules_v2.txt")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_version", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="preliminary_eval")

    parser.add_argument("--start_index", type=int, default=0, help="Zero-based start index.")
    parser.add_argument("--max_questions", type=int, default=-1, help="Use -1 for all questions.")
    parser.add_argument(
        "--question_ids",
        nargs="*",
        default=None,
        help="Optional list of question IDs, e.g. --question_ids q01 q03 q07.",
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="Used by GPT-OSS models.",
    )
    parser.add_argument(
        "--gemma_enable_thinking",
        action="store_true",
        help="Used by Gemma 4 models.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument("--save_internal_thoughts", action="store_true")
    parser.add_argument("--debug_timing", action="store_true")

    return parser.parse_args()


def build_result_record(
    args: argparse.Namespace,
    row: dict[str, Any],
    raw_response: str,
    response_for_parsing: Any,
    internal_thoughts: Optional[str],
    parsed_output: Optional[dict[str, Any]],
    validation: dict[str, Any],
    soft_warnings: list[str],
    debug_info: Optional[dict[str, Any]],
    prompt: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "question_id": row["id"],
        "dimension": row.get("dimension"),
        "question": row["question"],
        "backend": "unsloth_local",
        "model_name": args.model_name,
        "prompt_version": args.prompt_version,
        "eval_prompt_path": args.eval_prompt_path,
        "rules_path": args.rules_path,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_length": args.max_seq_length,
        "reasoning_effort": args.reasoning_effort,
        "gemma_enable_thinking": args.gemma_enable_thinking,
        "save_internal_thoughts": args.save_internal_thoughts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "internal_thoughts": internal_thoughts,
        "parsed_output": parsed_output,
        "raw_response": raw_response,
        "response_for_parsing": response_for_parsing,
        "validation": validation,
        "soft_warnings": soft_warnings,
    }

    if args.save_prompt:
        result["rendered_prompt"] = prompt

    cleaned_debug = remove_internal_thoughts_from_debug(debug_info)
    if cleaned_debug is not None:
        result["debug_info"] = cleaned_debug

    return result


def build_error_record(
    args: argparse.Namespace,
    row: dict[str, Any],
    error: Exception,
    prompt: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "question_id": row.get("id"),
        "dimension": row.get("dimension"),
        "question": row.get("question"),
        "backend": "unsloth_local",
        "model_name": args.model_name,
        "prompt_version": args.prompt_version,
        "eval_prompt_path": args.eval_prompt_path,
        "rules_path": args.rules_path,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_length": args.max_seq_length,
        "reasoning_effort": args.reasoning_effort,
        "gemma_enable_thinking": args.gemma_enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "error": str(error),
        "error_type": type(error).__name__,
        "traceback": traceback.format_exc(),
    }

    if args.save_prompt:
        result["rendered_prompt"] = prompt

    return result


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    questions_path = repo_root / args.questions_path
    eval_prompt_path = repo_root / args.eval_prompt_path
    rules_path = repo_root / args.rules_path

    questions = load_json(questions_path)
    eval_prompt = load_text(eval_prompt_path)
    rules_text = load_text(rules_path)

    rows = filter_questions(questions, args.question_ids)
    rows = select_rows(rows, start_index=args.start_index, max_items=args.max_questions)

    results_root = build_results_root(
        repo_root=repo_root,
        task_name=args.task_name,
        model_name=args.model_name,
        prompt_version=args.prompt_version,
    )

    model_io, model = load_local_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
    )

    print(f"Loaded {len(rows)} questions from {questions_path}", flush=True)
    print(f"Eval prompt file: {eval_prompt_path}", flush=True)
    print(f"Rules file: {rules_path}", flush=True)
    print("Backend: unsloth_local", flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print("Expected schema: {\"justification\": str, \"answer\": str}", flush=True)
    print(f"Reasoning effort: {args.reasoning_effort}", flush=True)
    print(f"Gemma thinking enabled: {args.gemma_enable_thinking}", flush=True)
    print(f"Max new tokens: {args.max_new_tokens}", flush=True)
    print(f"Max seq length: {args.max_seq_length}", flush=True)
    print(f"Temperature: {args.temperature}", flush=True)
    print(f"Top P: {args.top_p}", flush=True)
    print(f"Top K: {args.top_k}", flush=True)
    print(f"Repetition penalty: {args.repetition_penalty}", flush=True)
    print(f"No-repeat ngram size: {args.no_repeat_ngram_size}", flush=True)
    print(f"Results root: {results_root}", flush=True)
    print_local_model_summary(args.model_name, model_io, model)

    for local_i, row in enumerate(rows, start=1):
        global_i = args.start_index + local_i
        question_id = row["id"]
        output_path = results_root / f"{question_id}.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{global_i}] SKIP - {output_path.name} already exists", flush=True)
            continue

        prompt = build_preliminary_prompt(
            eval_prompt=eval_prompt,
            rules_text=rules_text,
            question_text=row["question"],
        )

        try:
            raw_response, debug_info = call_local_model(
                model=model,
                model_io=model_io,
                prompt=prompt,
                model_name=args.model_name,
                max_new_tokens=args.max_new_tokens,
                reasoning_effort=args.reasoning_effort,
                gemma_enable_thinking=args.gemma_enable_thinking,
                return_debug_info=True,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            if args.debug_timing:
                prefix = f"[{global_i}/{args.start_index + len(rows)}] {question_id}"
                print_generation_debug(prefix, debug_info)

            response_for_parsing = prepare_response_for_json(raw_response)
            parsed_raw = (
                parse_model_json(response_for_parsing)
                if isinstance(response_for_parsing, str)
                else None
            )
            parsed_output = normalize_preliminary_output(parsed_raw)
            validation = validate_preliminary_output(parsed_output)
            soft_warnings = add_preliminary_soft_warnings(
                validation=validation,
                raw_response=raw_response,
                response_for_parsing=response_for_parsing,
                parsed_output=parsed_raw,
                debug_info=debug_info,
                max_new_tokens=args.max_new_tokens,
            )
            internal_thoughts = get_internal_thoughts(
                debug_info=debug_info,
                save_internal_thoughts=args.save_internal_thoughts,
            )

            result = build_result_record(
                args=args,
                row=row,
                raw_response=raw_response,
                response_for_parsing=response_for_parsing,
                internal_thoughts=internal_thoughts,
                parsed_output=parsed_output,
                validation=validation,
                soft_warnings=soft_warnings,
                debug_info=debug_info,
                prompt=prompt,
            )
            save_json(output_path, result)

            status = "OK" if validation["is_valid"] else "PARSED_BUT_INVALID"
            print(
                f"[{global_i}/{args.start_index + len(rows)}] {status} - "
                f"{question_id}.json -> {validation.get('answer')}",
                flush=True,
            )

        except Exception as exc:
            error_result = build_error_record(args=args, row=row, error=exc, prompt=prompt)
            save_json(output_path, error_result)
            print(
                f"[{global_i}/{args.start_index + len(rows)}] ERROR - "
                f"{question_id}.json -> {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
