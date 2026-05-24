from __future__ import annotations

import sys
print(
    "BEFORE UNSLOTH | transformers loaded?",
    "transformers" in sys.modules,
    file=sys.stderr,
    flush=True,
)

import unsloth

print(
    "AFTER UNSLOTH | transformers loaded?",
    "transformers" in sys.modules,
    file=sys.stderr,
    flush=True,
)


import argparse
import traceback
from pathlib import Path
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


def build_full_prompt(
    base_prompt: str,
    rules_text: str,
    player_names: list[str],
    transcript_text: str,
) -> str:
    players_str = ", ".join(player_names)

    return f"""{base_prompt}

Here are the game rules:

{rules_text}

## Player list

{players_str}

## Transcript

{transcript_text}
""".strip()


def validate_vote_output(obj: Optional[dict[str, Any]], player_names: list[str]) -> dict[str, Any]:
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

    if "justification" not in obj:
        report["errors"].append("missing_justification")
    if "chosen_vote" not in obj:
        report["errors"].append("missing_chosen_vote")

    justification = obj.get("justification")
    chosen_vote = obj.get("chosen_vote")

    if not isinstance(justification, str) or not justification.strip():
        report["errors"].append("justification_not_nonempty_string")
    else:
        justification = justification.strip()

    if not isinstance(chosen_vote, str) or not chosen_vote.strip():
        report["errors"].append("chosen_vote_not_nonempty_string")
    else:
        chosen_vote = chosen_vote.strip()
        if chosen_vote not in player_names:
            report["errors"].append(f"chosen_vote_not_in_player_list:{chosen_vote}")

    report["justification"] = justification
    report["chosen_vote"] = chosen_vote
    report["is_valid"] = not report["errors"]
    return report


def add_vote_soft_warnings(
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

    justification = validation.get("justification")
    if isinstance(justification, str):
        sentence_count = count_sentences(justification)
        if sentence_count < 3:
            warnings.append("justification_may_be_shorter_than_3_sentences")
        if sentence_count > 5:
            warnings.append("justification_may_exceed_5_sentences")

    if isinstance(parsed_output, dict):
        expected_fields = {"justification", "chosen_vote"}
        unexpected_fields = sorted(set(parsed_output.keys()) - expected_fields)
        if unexpected_fields:
            warnings.append(f"response_contains_unexpected_fields:{unexpected_fields}")

    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONUW voting experiments with local Unsloth models."
    )

    parser.add_argument(
        "--index_path",
        type=str,
        default="data/processed/lai2023/onuw_transcripts_ready/index_cleaned.json",
    )
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--rules_path", type=str, default="src/prompts/onuw_rules_v2.txt")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_version", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="voting")

    parser.add_argument("--start_index", type=int, default=0, help="Zero-based start index.")
    parser.add_argument("--max_games", type=int, default=-1, help="Use -1 for all games.")
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=768)
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
        "source": row["source"],
        "session_name": row["session_name"],
        "game_key": row["game_key"],
        "processed_txt_path": row["processed_txt_path"],
        "player_names": row["player_names"],
        "backend": "unsloth_local",
        "model_name": args.model_name,
        "prompt_version": args.prompt_version,
        "prompt_path": args.prompt_path,
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
        "source": row.get("source"),
        "session_name": row.get("session_name"),
        "game_key": row.get("game_key"),
        "processed_txt_path": row.get("processed_txt_path"),
        "player_names": row.get("player_names"),
        "backend": "unsloth_local",
        "model_name": args.model_name,
        "prompt_version": args.prompt_version,
        "prompt_path": args.prompt_path,
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

    index_path = repo_root / args.index_path
    prompt_path = repo_root / args.prompt_path
    rules_path = repo_root / args.rules_path

    index_data = load_json(index_path)
    base_prompt = load_text(prompt_path)
    rules_text = load_text(rules_path)
    rows = select_rows(index_data, start_index=args.start_index, max_items=args.max_games)

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

    print(f"Loaded {len(rows)} games from {index_path}", flush=True)
    print(f"Prompt file: {prompt_path}", flush=True)
    print(f"Rules file: {rules_path}", flush=True)
    print("Backend: unsloth_local", flush=True)
    print(f"Model: {args.model_name}", flush=True)
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
        transcript_rel = Path(row["processed_txt_path"])
        transcript_path = repo_root / transcript_rel
        source = row["source"]
        transcript_stem = transcript_path.stem
        game_id = f"{source}/{transcript_stem}"

        output_dir = results_root / source
        output_path = output_dir / f"{transcript_stem}.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{global_i}] SKIP - {output_path.name} already exists", flush=True)
            continue

        transcript_text = load_text(transcript_path)
        prompt = build_full_prompt(
            base_prompt=base_prompt,
            rules_text=rules_text,
            player_names=row["player_names"],
            transcript_text=transcript_text,
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
                prefix = f"[{global_i}/{args.start_index + len(rows)}] {game_id}"
                print_generation_debug(prefix, debug_info)

            response_for_parsing = prepare_response_for_json(raw_response)
            parsed_output = (
                parse_model_json(response_for_parsing)
                if isinstance(response_for_parsing, str)
                else None
            )
            validation = validate_vote_output(parsed_output, row["player_names"])
            soft_warnings = add_vote_soft_warnings(
                validation=validation,
                raw_response=raw_response,
                response_for_parsing=response_for_parsing,
                parsed_output=parsed_output,
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
                f"{game_id}.json -> {validation.get('chosen_vote')}",
                flush=True,
            )

        except Exception as exc:
            error_result = build_error_record(args=args, row=row, error=exc, prompt=prompt)
            save_json(output_path, error_result)
            print(f"[{global_i}/{args.start_index + len(rows)}] ERROR - {game_id}.json -> {exc}", flush=True)


if __name__ == "__main__":
    main()
