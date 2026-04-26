import argparse
import json
import time
from pathlib import Path
from typing import Optional

from google import genai

from src.utils.io_utils import find_repo_root, load_json, load_text
from src.utils.json_utils import parse_model_json
from src.utils.model_utils import (
    call_gemini,
    load_local_model,
    call_local_model,
    get_model_io_info,
)


def build_preliminary_prompt(
    eval_prompt: str,
    rules_text: str,
    question_text: str,
) -> str:
    return f"""{eval_prompt}

Here are the game rules:

{rules_text}

Now answer the following question.

Question:
{question_text}
""".strip()


def validate_preliminary_output(obj: Optional[dict]) -> dict:
    report = {
        "is_valid": False,
        "errors": [],
        "final_answer": None,
        "reasoning": None,
    }

    if obj is None:
        report["errors"].append("no_parseable_json")
        return report

    if not isinstance(obj, dict):
        report["errors"].append("parsed_output_not_dict")
        return report

    final_answer = obj.get("final_answer")
    reasoning = obj.get("reasoning")

    if not isinstance(final_answer, str) or not final_answer.strip():
        report["errors"].append("final_answer_not_nonempty_string")

    if not isinstance(reasoning, str) or not reasoning.strip():
        report["errors"].append("reasoning_not_nonempty_string")

    report["final_answer"] = final_answer
    report["reasoning"] = reasoning

    if not report["errors"]:
        report["is_valid"] = True

    return report


def add_soft_warnings(validation: dict, raw_response: str, parsed_output: Optional[dict]) -> list[str]:
    warnings = []

    final_answer = validation.get("final_answer")
    reasoning = validation.get("reasoning")

    if isinstance(final_answer, str) and len(final_answer.strip()) > 80:
        warnings.append("final_answer_may_be_too_long")

    if isinstance(reasoning, str):
        low = reasoning.lower()
        if any(x in low for x in ["i think", "probably", "maybe", "i guess"]):
            warnings.append("reasoning_contains_uncertain_language")

    if "```" in raw_response:
        warnings.append("response_contains_markdown_fences")

    if raw_response.count("{") > 1 and parsed_output is not None:
        warnings.append("response_may_contain_extra_json_or_wrapper_text")

    if isinstance(parsed_output, dict) and "id" in parsed_output:
        warnings.append("response_contains_unexpected_id_field")

    return warnings


def filter_questions(rows: list[dict], question_ids: list[str] | None) -> list[dict]:
    if not question_ids:
        return rows

    wanted = set(question_ids)
    filtered = [row for row in rows if row.get("id") in wanted]

    found = {row.get("id") for row in filtered}
    missing = wanted - found
    if missing:
        raise ValueError(f"Question IDs not found in question file: {sorted(missing)}")

    return filtered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions_path",
        type=str,
        default="src/preliminary_eval/onuw_questions.json",
    )
    parser.add_argument(
        "--eval_prompt_path",
        type=str,
        default="src/prompts/preliminary_eval_prompt.txt",
    )
    parser.add_argument(
        "--rules_path",
        type=str,
        default="src/prompts/onuw_rules.txt",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gemini", "hf_local"],
        default="hf_local",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="v1",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="preliminary_eval",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=-1,
        help="Number of questions to process. Use -1 for all.",
    )
    parser.add_argument(
        "--question_ids",
        nargs="*",
        default=None,
        help="Optional list of specific question IDs to run, e.g. --question_ids q01 q03 q07",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--save_prompt",
        action="store_true",
        help="Save the fully rendered prompt inside each result JSON.",
    )
    parser.add_argument(
        "--require_chat_template_for_gpt_oss",
        action="store_true",
        help="Raise an error if a gpt-oss model is used without a tokenizer chat template.",
    )
    parser.add_argument(
        "--debug_timing",
        action="store_true",
        help="Print prompt length, token counts, and generation timing.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()

    questions_path = repo_root / args.questions_path
    eval_prompt_path = repo_root / args.eval_prompt_path
    rules_path = repo_root / args.rules_path

    questions = load_json(questions_path)
    eval_prompt = load_text(eval_prompt_path)
    rules_text = load_text(rules_path)

    rows = filter_questions(questions, args.question_ids)

    if args.max_questions != -1:
        rows = rows[: args.max_questions]

    safe_model_name = args.model_name.replace("/", "_")
    results_root = (
        repo_root
        / "results"
        / args.task_name
        / safe_model_name
        / f"prompt_{args.prompt_version}"
    )
    results_root.mkdir(parents=True, exist_ok=True)

    client = None
    model_io = None
    model = None

    if args.backend == "gemini":
        client = genai.Client()
    elif args.backend == "hf_local":
        model_io, model = load_local_model(args.model_name)

    print(f"Loaded {len(rows)} questions from {questions_path}")
    print(f"Eval prompt file: {eval_prompt_path}")
    print(f"Rules file: {rules_path}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_name}")
    print(f"Results root: {results_root}")

    if args.backend == "hf_local":
        io_info = get_model_io_info(args.model_name, model_io)
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model IO type: {io_info['io_type']}")
        print(f"Has chat template: {io_info['has_chat_template']}")

    for i, row in enumerate(rows, start=1):
        question_id = row["id"]
        question_text = row["question"]

        output_path = results_root / f"{question_id}.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{i}/{len(rows)}] SKIP - {output_path.name} already exists")
            continue

        prompt = build_preliminary_prompt(
            eval_prompt=eval_prompt,
            rules_text=rules_text,
            question_text=question_text,
        )

        try:
            debug_info = None

            if args.backend == "gemini":
                t0 = time.time()
                raw_response = call_gemini(
                    client=client,
                    model_name=args.model_name,
                    prompt=prompt,
                )
                t1 = time.time()
                debug_info = {
                    "generation_time_sec": float(t1 - t0),
                    "input_char_count": len(prompt),
                }

            elif args.backend == "hf_local":
                if args.debug_timing:
                    raw_response, debug_info = call_local_model(
                        model=model,
                        model_io=model_io,
                        prompt=prompt,
                        model_name=args.model_name,
                        max_new_tokens=args.max_new_tokens,
                        require_chat_template_for_gpt_oss=args.require_chat_template_for_gpt_oss,
                        return_debug_info=True,
                    )
                else:
                    raw_response = call_local_model(
                        model=model,
                        model_io=model_io,
                        prompt=prompt,
                        model_name=args.model_name,
                        max_new_tokens=args.max_new_tokens,
                        require_chat_template_for_gpt_oss=args.require_chat_template_for_gpt_oss,
                        return_debug_info=False,
                    )
            else:
                raise ValueError(f"Unsupported backend: {args.backend}")

            if args.debug_timing and debug_info is not None:
                if args.backend == "hf_local":
                    print(
                        f"[{i}/{len(rows)}] {question_id} | "
                        f"chars={debug_info['input_char_count']} | "
                        f"in_tokens={debug_info['input_token_count']} | "
                        f"out_tokens={debug_info['output_token_count']} | "
                        f"time={debug_info['generation_time_sec']:.2f}s | "
                        f"device={debug_info['model_device']}"
                    )
                else:
                    print(
                        f"[{i}/{len(rows)}] {question_id} | "
                        f"chars={debug_info['input_char_count']} | "
                        f"time={debug_info['generation_time_sec']:.2f}s"
                    )

            parsed_output = parse_model_json(raw_response)
            validation = validate_preliminary_output(parsed_output)
            soft_warnings = add_soft_warnings(validation, raw_response, parsed_output)

            result = {
                "question_id": question_id,
                "dimension": row.get("dimension"),
                "question": question_text,
                "backend": args.backend,
                "model_name": args.model_name,
                "prompt_version": args.prompt_version,
                "eval_prompt_path": args.eval_prompt_path,
                "rules_path": args.rules_path,
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "validation": validation,
                "soft_warnings": soft_warnings,
            }

            if args.save_prompt:
                result["rendered_prompt"] = prompt

            if debug_info is not None:
                result["debug_info"] = debug_info

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            status = "OK" if validation["is_valid"] else "PARSED_BUT_INVALID"
            print(f"[{i}/{len(rows)}] {status} - {question_id}.json")

        except Exception as e:
            error_result = {
                "question_id": question_id,
                "dimension": row.get("dimension"),
                "question": question_text,
                "backend": args.backend,
                "model_name": args.model_name,
                "prompt_version": args.prompt_version,
                "eval_prompt_path": args.eval_prompt_path,
                "rules_path": args.rules_path,
                "error": str(e),
            }

            if args.save_prompt:
                error_result["rendered_prompt"] = prompt

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)

            print(f"[{i}/{len(rows)}] ERROR - {question_id}.json -> {e}")

        if args.backend == "gemini":
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()