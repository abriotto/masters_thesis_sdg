import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Optional

from google import genai

from src.utils.io_utils import find_repo_root, load_json, load_text
from src.utils.json_utils import parse_model_json
from src.utils.model_utils import (
        call_gemini,
        call_local_model,
        get_model_input_device,
        get_model_io_info,
        load_local_model,
    )


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


def extract_response_for_parsing(raw_response: str, model_name: str) -> str:
    if not isinstance(raw_response, str):
        return raw_response

    cleaned = raw_response.strip()

    if "gpt-oss" in model_name.lower():
        markers = [
            "assistantfinal",
            "<|channel|>final<|message|>",
        ]
        for marker in markers:
            idx = cleaned.rfind(marker)
            if idx != -1:
                candidate = cleaned[idx + len(marker):].strip()
                if "{" in candidate:
                    cleaned = candidate
                    break

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    return cleaned


def extract_internal_thoughts(
    raw_response: str, 
    model_name: str, 
    debug_info: Optional[dict]
) -> Optional[str]:
    """
    Unifies the internal reasoning into a single field for all models.
    """
    # 1. Gemma 4 (Captured by the model handler in debug_info)
    if debug_info and debug_info.get("internal_thoughts"):
        return debug_info["internal_thoughts"]
        
    # 2. GPT-OSS (Extract everything before the final answer marker)
    if "gpt-oss" in model_name.lower() and isinstance(raw_response, str):
        markers = [
            "assistantfinal",
            "<|channel|>final<|message|>",
        ]
        for marker in markers:
            idx = raw_response.rfind(marker)
            if idx != -1:
                # Everything before the marker is the reasoning/analysis
                return raw_response[:idx].strip()

    return None


def validate_vote_output(obj: Optional[dict], player_names: list[str]) -> dict:
    report = {
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

    if "chosen_vote" not in obj:
        report["errors"].append("missing_chosen_vote")
    if "justification" not in obj:
        report["errors"].append("missing_justification")

    chosen_vote = obj.get("chosen_vote")
    justification = obj.get("justification")

    if not isinstance(chosen_vote, str) or not chosen_vote.strip():
        report["errors"].append("chosen_vote_not_nonempty_string")
    else:
        chosen_vote = chosen_vote.strip()
        if chosen_vote not in player_names:
            report["errors"].append(f"chosen_vote_not_in_player_list:{chosen_vote}")

    if not isinstance(justification, str) or not justification.strip():
        report["errors"].append("justification_not_nonempty_string")
    else:
        justification = justification.strip()

    report["chosen_vote"] = chosen_vote
    report["justification"] = justification

    if not report["errors"]:
        report["is_valid"] = True

    return report


def add_soft_warnings(
    validation: dict,
    raw_response: str,
    response_for_parsing: str,
) -> list[str]:
    warnings = []

    if not isinstance(raw_response, str):
        warnings.append(f"raw_response_not_string:{type(raw_response).__name__}")
        return warnings

    if not isinstance(response_for_parsing, str):
        warnings.append(f"response_for_parsing_not_string:{type(response_for_parsing).__name__}")
        return warnings

    justification = validation.get("justification")
    if isinstance(justification, str):
        low = justification.lower()

        sentence_count = (
            justification.count(".")
            + justification.count("!")
            + justification.count("?")
        )

        if sentence_count < 3:
            warnings.append("justification_may_be_shorter_than_3_sentences")
        if sentence_count > 5:
            warnings.append("justification_may_exceed_5_sentences")

    raw_low = raw_response.lower()

    if raw_low.count("```") > 0:
        warnings.append("response_contains_code_fence")

    if "analysis" in raw_low and "assistantfinal" in raw_low:
        warnings.append("response_contains_gpt_oss_analysis_channel")

    if response_for_parsing.strip() != raw_response.strip():
        warnings.append("response_was_cleaned_before_parsing")

    return warnings


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
        "--index_path",
        type=str,
        default="data/processed/lai2023/onuw_transcripts_ready/index_cleaned.json",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/prompts/vote_prompt_v2.txt",
    )
    parser.add_argument(
        "--rules_path",
        type=str,
        default="src/prompts/onuw_rules_v2.txt",
        help="Path to the ONUW rules file.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gemini", "unsloth_local"],
        default="gemini",
        help="Backend to use. Use 'unsloth_local' for local Unsloth models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="voting",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=-1,
        help="Number of games to process. Use -1 for all.",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls. Only used with Gemini.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=768,
        help="Maximum number of new tokens generated by local Unsloth models.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=8192,
        help=(
            "Maximum sequence length used when loading local Unsloth models. "
            "Increase this if long transcripts are being truncated or rejected, "
            "but remember that larger contexts use more VRAM."
        ),
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
        help="Enable Gemma 4 thinking mode. Ignored by non-Gemma models.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling threshold.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=64,
        help="Top-k sampling threshold.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty for local Unsloth generation. Use 1.0 to disable.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Optional no-repeat ngram size for local Unsloth generation. 0 disables it.",
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
    parser.add_argument(
        "--save_internal_thoughts",
        action="store_true",
        help=(
            "Save extracted thinking / reasoning traces when the local backend exposes them. "
            "This does not print timing debug info unless --debug_timing is also set."
        ),
    )

    return parser.parse_args()


def load_backend(args: argparse.Namespace):
    if args.backend == "gemini":
        return genai.Client(), None, None

    if args.backend == "unsloth_local":
        model_io, model = load_local_model(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
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
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        t1 = time.time()

        debug_info = {
            "generation_time_sec": float(t1 - t0),
            "input_char_count": len(prompt),
        }

        return raw_response, debug_info

    if args.backend == "unsloth_local":
        return call_local_model(
            model=model,
            model_io=model_io,
            prompt=prompt,
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            reasoning_effort=args.reasoning_effort,
            gemma_enable_thinking=args.gemma_enable_thinking,
            return_debug_info=(args.debug_timing or args.save_internal_thoughts),
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    raise ValueError(f"Unsupported backend: {args.backend}")


def unpack_backend_response(args: argparse.Namespace, response):
    # Local calls return either text or (text, debug_info). Be defensive because
    # backend failures can otherwise turn into opaque unpacking errors.
    if isinstance(response, tuple):
        if len(response) != 2:
            raise RuntimeError(
                f"Expected backend response tuple of length 2, got length {len(response)}"
            )
        return response[0], response[1]

    if args.backend == "unsloth_local" and (args.debug_timing or args.save_internal_thoughts):
        raise RuntimeError(
            "Local backend was expected to return (text, debug_info), but got "
            f"{type(response).__name__}: {repr(response)[:500]}"
        )

    return response, None

def print_debug_info(
    args: argparse.Namespace,
    game_index: int,
    total_games: int,
    game_id: str,
    debug_info: Optional[dict],
) -> None:
    if not args.debug_timing or debug_info is None:
        return

    if args.backend == "unsloth_local":
        print(
            f"[{game_index}/{total_games}] {game_id} | "
            f"chars={debug_info['input_char_count']} | "
            f"in_tokens={debug_info['input_token_count']} | "
            f"out_tokens={debug_info['output_token_count']} | "
            f"time={debug_info['generation_time_sec']:.2f}s | "
            f"device={debug_info['model_device']} | "
            f"loader={debug_info.get('loader')} | "
            f"handler={debug_info.get('handler', 'unknown')} | "
            f"quantization={debug_info.get('quantization')} | "
            f"resolved_model={debug_info.get('resolved_model_name')} | "
            f"reasoning_effort={debug_info.get('reasoning_effort')} | "
            f"gemma_thinking={debug_info.get('gemma_enable_thinking')} | "
            f"rep_penalty={debug_info.get('repetition_penalty')} | "
            f"no_repeat_ngram={debug_info.get('no_repeat_ngram_size')}"
        )
        return

    print(
        f"[{game_index}/{total_games}] {game_id} | "
        f"chars={debug_info['input_char_count']} | "
        f"time={debug_info['generation_time_sec']:.2f}s"
    )


def build_result_record(
    args: argparse.Namespace,
    row: dict,
    raw_response: str,
    response_for_parsing: str,
    internal_thoughts: Optional[str],
    parsed_output: Optional[dict],
    validation: dict,
    soft_warnings: list[str],
    debug_info: Optional[dict],
    prompt: str,
) -> dict:
    result = {
        "source": row["source"],
        "session_name": row["session_name"],
        "game_key": row["game_key"],
        "processed_txt_path": row["processed_txt_path"],
        "player_names": row["player_names"],
        "backend": args.backend,
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
        
        # Unified fields
        "internal_thoughts": internal_thoughts,
        "parsed_output": parsed_output,
        "raw_response": raw_response,
        "response_for_parsing": response_for_parsing,
        
        "validation": validation,
        "soft_warnings": soft_warnings,
    }

    if args.save_prompt:
        result["rendered_prompt"] = prompt

    if debug_info is not None:
        # We can safely delete the duplicated thoughts from debug_info 
        # now that they are at the root level.
        if "internal_thoughts" in debug_info:
            del debug_info["internal_thoughts"]
        result["debug_info"] = debug_info

    return result


def build_error_record(
    args: argparse.Namespace,
    row: dict,
    error: Exception,
    prompt: str,
) -> dict:
    result = {
        "source": row.get("source"),
        "session_name": row.get("session_name"),
        "game_key": row.get("game_key"),
        "processed_txt_path": row.get("processed_txt_path"),
        "player_names": row.get("player_names"),
        "backend": args.backend,
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

    rows = index_data if args.max_games == -1 else index_data[: args.max_games]

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

    print(f"Loaded {len(rows)} games from {index_path}")
    print(f"Prompt file: {prompt_path}")
    print(f"Rules file: {rules_path}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_name}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Top P: {args.top_p}")
    print(f"Top K: {args.top_k}")
    print(f"Save internal thoughts: {args.save_internal_thoughts}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print(f"No-repeat ngram size: {args.no_repeat_ngram_size}")
    print(f"Results root: {results_root}")

    if args.backend == "unsloth_local":
        io_info = get_model_io_info(args.model_name, model_io)
        print(f"Model input device: {get_model_input_device(model)}")
        print(f"Local backend: {getattr(model, '_local_backend', 'unsloth')}")
        print(f"Model loader: {getattr(model, '_local_loader', 'unknown')}")
        print(f"Model quantization: {getattr(model, '_local_quantization', 'unknown')}")
        print(f"Requested model name: {getattr(model, '_requested_model_name', args.model_name)}")
        print(f"Resolved model name: {getattr(model, '_resolved_model_name', args.model_name)}")
        print(f"Model max seq length: {getattr(model, '_max_seq_length', args.max_seq_length)}")
        print(f"Model family: {io_info['family']}")
        print(f"Model IO type: {io_info['io_type']}")
        print(f"Has chat template: {io_info['has_chat_template']}")
        print(f"Gemma thinking enabled: {args.gemma_enable_thinking}")

    for i, row in enumerate(rows, start=1):
        transcript_rel = Path(row["processed_txt_path"])
        transcript_path = repo_root / transcript_rel

        source = row["source"]
        transcript_stem = transcript_path.stem

        output_dir = results_root / source
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{transcript_stem}.json"

        if output_path.exists() and not args.overwrite:
            print(f"[{i}/{len(rows)}] SKIP - {output_path.name} already exists")
            continue

        transcript_text = load_text(transcript_path)
        prompt = build_full_prompt(
            base_prompt=base_prompt,
            rules_text=rules_text,
            player_names=row["player_names"],
            transcript_text=transcript_text,
        )

        game_id = f"{source}/{transcript_stem}"

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
                game_index=i,
                total_games=len(rows),
                game_id=game_id,
                debug_info=debug_info,
            )

            response_for_parsing = extract_response_for_parsing(
                raw_response=raw_response,
                model_name=args.model_name,
            )
            
            # Extract unified thoughts
            internal_thoughts = extract_internal_thoughts(
                raw_response=raw_response,
                model_name=args.model_name,
                debug_info=debug_info,
            )

            if isinstance(response_for_parsing, str):
                parsed_output = parse_model_json(response_for_parsing)
            else:
                parsed_output = None

            validation = validate_vote_output(parsed_output, row["player_names"])
            soft_warnings = add_soft_warnings(
                validation=validation,
                raw_response=raw_response,
                response_for_parsing=response_for_parsing,
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
            chosen_vote = validation.get("chosen_vote")
            print(f"[{i}/{len(rows)}] {status} - {game_id}.json -> {chosen_vote}")

        except Exception as e:
            error_result = build_error_record(
                args=args,
                row=row,
                error=e,
                prompt=prompt,
            )

            save_json(output_path, error_result)

            print(f"[{i}/{len(rows)}] ERROR - {game_id}.json -> {e}")

        if args.backend == "gemini":
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()