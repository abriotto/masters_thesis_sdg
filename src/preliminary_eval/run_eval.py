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
    call_local_model,
    get_model_io_info,
    load_local_model,
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
        "justification": None,
        "mechanical_analysis": None,
    }

    if obj is None:
        report["errors"].append("no_parseable_json")
        return report

    if not isinstance(obj, dict):
        report["errors"].append("parsed_output_not_dict")
        return report

    final_answer = obj.get("final_answer")
    justification = obj.get("justification")
    mechanical_analysis = obj.get("mechanical_analysis")

    if not isinstance(final_answer, str) or not final_answer.strip():
        report["errors"].append("final_answer_not_nonempty_string")

    if not isinstance(justification, str) or not justification.strip():
        report["errors"].append("justification_not_nonempty_string")

    if not isinstance(mechanical_analysis, str) or not mechanical_analysis.strip():
        report["errors"].append("mechanical_analysis_not_nonempty_string")

    report["final_answer"] = final_answer
    report["justification"] = justification
    report["mechanical_analysis"] = mechanical_analysis

    if not report["errors"]:
        report["is_valid"] = True

    return report


def add_soft_warnings(
    validation: dict,
    raw_response: str,
    parsed_output: Optional[dict],
) -> list[str]:
    warnings = []

    final_answer = validation.get("final_answer")
    justification = validation.get("justification")
    mechanical_analysis = validation.get("mechanical_analysis")

    if isinstance(final_answer, str) and len(final_answer.strip()) > 80:
        warnings.append("final_answer_may_be_too_long")

    if isinstance(justification, str):
        low = justification.lower()

        if any(x in low for x in ["i think", "probably", "maybe", "i guess"]):
            warnings.append("justification_contains_uncertain_language")

        sentence_count = (
            justification.count(".")
            + justification.count("!")
            + justification.count("?")
        )
        if sentence_count > 2:
            warnings.append("justification_may_exceed_2_sentences")

    if isinstance(mechanical_analysis, str):
        if len(mechanical_analysis.strip()) > 1500:
            warnings.append("mechanical_analysis_may_be_too_long")

        low = mechanical_analysis.lower()
        if any(x in low for x in ["i think", "probably", "maybe", "i guess"]):
            warnings.append("mechanical_analysis_contains_uncertain_language")

    if "```" in raw_response:
        warnings.append("response_contains_markdown_fences")

    if raw_response.count("{") > 1 and parsed_output is not None:
        warnings.append("response_may_contain_extra_json_or_wrapper_text")

    if isinstance(parsed_output, dict):
        expected_fields = {
            "mechanical_analysis",
            "justification",
            "final_answer",
        }

        unexpected_fields = sorted(set(parsed_output.keys()) - expected_fields)
        if unexpected_fields:
            warnings.append(f"response_contains_unexpected_fields:{unexpected_fields}")

        if "id" in parsed_output:
            warnings.append("response_contains_unexpected_id_field")

        if "reasoning" in parsed_output:
            warnings.append("response_contains_old_reasoning_field")

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


def safe_path_name(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
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
        help="Optional list of specific question IDs to run, e.g. --question_ids q01 q03 q07.",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls. Only used with the Gemini backend.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens generated by local HF models.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Optional HF local model quantization. Defaults to none.",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="Reasoning effort for GPT-OSS models. Ignored by non-GPT-OSS local models.",
    )
    parser.add_argument(
        "--gemma_enable_thinking",
        action="store_true",
        help="Enable Gemma 4 thinking mode. Ignored by non-Gemma models. Defaults to disabled.",
    )
    parser.add_argument(
        "--save_prompt",
        action="store_true",
        help="Save the fully rendered prompt inside each result JSON.",
    )
    parser.add_argument(
        "--debug_timing",
        action="store_true",
        help="Print prompt length, token counts, and generation timing.",
    )

    return parser.parse_args()


def load_backend(args: argparse.Namespace):
    if args.backend == "gemini":
        return genai.Client(), None, None

    if args.backend == "hf_local":
        model_io, model = load_local_model(
            model_name=args.model_name,
            quantization=args.quantization,
        )
        return None, model_io, model

    raise ValueError(f"Unsupported backend: {args.backend}")


def call_backend(
    args: argparse.Namespace,
    prompt: str,
    client,
    model_io,
    model,
):
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

        return raw_response, debug_info

    if args.backend == "hf_local":
        return call_local_model(
            model=model,
            model_io=model_io,
            prompt=prompt,
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            reasoning_effort=args.reasoning_effort,
            gemma_enable_thinking=args.gemma_enable_thinking,
            return_debug_info=args.debug_timing,
        )

    raise ValueError(f"Unsupported backend: {args.backend}")


def unpack_backend_response(args: argparse.Namespace, response):
    if args.backend == "hf_local" and args.debug_timing:
        return response

    if isinstance(response, tuple):
        return response[0], response[1]

    return response, None


def print_debug_info(
    args: argparse.Namespace,
    question_index: int,
    total_questions: int,
    question_id: str,
    debug_info: Optional[dict],
) -> None:
    if not args.debug_timing or debug_info is None:
        return

    if args.backend == "hf_local":
        print(
            f"[{question_index}/{total_questions}] {question_id} | "
            f"chars={debug_info['input_char_count']} | "
            f"in_tokens={debug_info['input_token_count']} | "
            f"out_tokens={debug_info['output_token_count']} | "
            f"time={debug_info['generation_time_sec']:.2f}s | "
            f"device={debug_info['model_device']} | "
            f"handler={debug_info.get('handler', 'unknown')} | "
            f"reasoning_effort={debug_info.get('reasoning_effort')} | "
            f"gemma_thinking={debug_info.get('gemma_enable_thinking')}"
        )
        return

    print(
        f"[{question_index}/{total_questions}] {question_id} | "
        f"chars={debug_info['input_char_count']} | "
        f"time={debug_info['generation_time_sec']:.2f}s"
    )


def build_result_record(
    args: argparse.Namespace,
    row: dict,
    raw_response: str,
    parsed_output: Optional[dict],
    validation: dict,
    soft_warnings: list[str],
    debug_info: Optional[dict],
    prompt: str,
) -> dict:
    result = {
        "question_id": row["id"],
        "dimension": row.get("dimension"),
        "question": row["question"],
        "backend": args.backend,
        "model_name": args.model_name,
        "prompt_version": args.prompt_version,
        "eval_prompt_path": args.eval_prompt_path,
        "rules_path": args.rules_path,
        "max_new_tokens": args.max_new_tokens,
        "quantization": args.quantization,
        "reasoning_effort": args.reasoning_effort,
        "gemma_enable_thinking": args.gemma_enable_thinking,
        "raw_response": raw_response,
        "parsed_output": parsed_output,
        "validation": validation,
        "soft_warnings": soft_warnings,
    }

    if args.save_prompt:
        result["rendered_prompt"] = prompt

    if debug_info is not None:
        result["debug_info"] = debug_info

    return result


def build_error_record(
    args: argparse.Namespace,
    row: dict,
    error: Exception,
    prompt: str,
) -> dict:
    result = {
        "question_id": row.get("id"),
        "dimension": row.get("dimension"),
        "question": row.get("question"),
        "backend": args.backend,
        "model_name": args.model_name,
        "prompt_version": args.prompt_version,
        "eval_prompt_path": args.eval_prompt_path,
        "rules_path": args.rules_path,
        "max_new_tokens": args.max_new_tokens,
        "quantization": args.quantization,
        "reasoning_effort": args.reasoning_effort,
        "gemma_enable_thinking": args.gemma_enable_thinking,
        "error": str(error),
        "error_type": type(error).__name__,
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

    if args.max_questions != -1:
        rows = rows[: args.max_questions]

    safe_model_name = safe_path_name(args.model_name)
    results_root = (
        repo_root
        / "results"
        / args.task_name
        / safe_model_name
        / f"prompt_{args.prompt_version}"
    )
    results_root.mkdir(parents=True, exist_ok=True)

    client, model_io, model = load_backend(args)

    print(f"Loaded {len(rows)} questions from {questions_path}")
    print(f"Eval prompt file: {eval_prompt_path}")
    print(f"Rules file: {rules_path}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_name}")
    print(f"Quantization: {args.quantization}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Results root: {results_root}")

    if args.backend == "hf_local":
        io_info = get_model_io_info(args.model_name, model_io)
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model family: {io_info['family']}")
        print(f"Model IO type: {io_info['io_type']}")
        print(f"Has chat template: {io_info['has_chat_template']}")
        print(f"Gemma thinking enabled: {args.gemma_enable_thinking}")

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
            response = call_backend(
                args=args,
                prompt=prompt,
                client=client,
                model_io=model_io,
                model=model,
            )

            raw_response, debug_info = unpack_backend_response(args, response)

            print_debug_info(
                args=args,
                question_index=i,
                total_questions=len(rows),
                question_id=question_id,
                debug_info=debug_info,
            )

            parsed_output = parse_model_json(raw_response)
            validation = validate_preliminary_output(parsed_output)
            soft_warnings = add_soft_warnings(
                validation=validation,
                raw_response=raw_response,
                parsed_output=parsed_output,
            )

            result = build_result_record(
                args=args,
                row=row,
                raw_response=raw_response,
                parsed_output=parsed_output,
                validation=validation,
                soft_warnings=soft_warnings,
                debug_info=debug_info,
                prompt=prompt,
            )

            save_json(output_path, result)

            status = "OK" if validation["is_valid"] else "PARSED_BUT_INVALID"
            print(f"[{i}/{len(rows)}] {status} - {question_id}.json")

        except Exception as e:
            error_result = build_error_record(
                args=args,
                row=row,
                error=e,
                prompt=prompt,
            )

            save_json(output_path, error_result)

            print(f"[{i}/{len(rows)}] ERROR - {question_id}.json -> {e}")

        if args.backend == "gemini":
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()