from __future__ import annotations

import unsloth

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence


def safe_path_name(name: str) -> str:
    """Return a filesystem-safe version of a model / run name."""
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_results_root(
    repo_root: Path,
    task_name: str,
    model_name: str,
    prompt_version: str,
) -> Path:
    results_root = (
        repo_root
        / "results"
        / task_name
        / safe_path_name(model_name)
        / f"prompt_{prompt_version}"
    )
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def select_rows(
    rows: Sequence[dict[str, Any]],
    start_index: int = 0,
    max_items: int = -1,
) -> list[dict[str, Any]]:
    """
    Select a resumable/batchable slice.

    `start_index` is zero-based. `max_items=-1` means all rows after start_index.
    """
    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    sliced = list(rows[start_index:])
    if max_items != -1:
        if max_items < 0:
            raise ValueError("max_items must be -1 or >= 0")
        sliced = sliced[:max_items]
    return sliced


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    return cleaned


def _remove_common_generation_stops(text: str) -> str:
    cleaned = text.strip()
    for stop in ("<|return|>", "<|end|>"):
        if stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0].strip()
    return cleaned


def extract_balanced_json_object(text: str) -> Optional[str]:
    """
    Extract the first balanced JSON object substring.

    This is intentionally generic and model-agnostic. Model-specific channel parsing
    should happen in `model_utils.py` before the runner sees the response.
    """
    if not isinstance(text, str):
        return None

    for start in [i for i, ch in enumerate(text) if ch == "{"]:
        depth = 0
        in_string = False
        escape = False

        for pos in range(start, len(text)):
            ch = text[pos]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : pos + 1].strip()
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break

    return None


def prepare_response_for_json(raw_response: Any) -> Any:
    """
    Generic response cleanup before JSON parsing.

    The model-specific parsing boundary is:
    - `model_utils.py` removes Harmony/Gemma channel wrappers and returns visible text.
    - this function only handles generic wrappers: markdown fences, stop tokens, and
      extracting a balanced JSON object from surrounding text.
    """
    if not isinstance(raw_response, str):
        return raw_response

    cleaned = raw_response.strip()
    cleaned = strip_code_fences(cleaned)
    cleaned = _remove_common_generation_stops(cleaned)

    extracted = extract_balanced_json_object(cleaned)
    if extracted is not None:
        return extracted

    return cleaned


def count_sentences(text: str) -> int:
    """A lightweight sentence counter suitable for soft warnings only."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(re.findall(r"[.!?]+(?:\s|$)", text.strip()))


def add_common_soft_warnings(
    raw_response: Any,
    response_for_parsing: Any,
    parsed_output: Optional[dict[str, Any]],
    debug_info: Optional[dict[str, Any]],
    max_new_tokens: int,
) -> list[str]:
    warnings: list[str] = []

    if not isinstance(raw_response, str):
        warnings.append(f"raw_response_not_string:{type(raw_response).__name__}")
        return warnings

    if not isinstance(response_for_parsing, str):
        warnings.append(
            f"response_for_parsing_not_string:{type(response_for_parsing).__name__}"
        )
        return warnings

    if "```" in raw_response:
        warnings.append("response_contains_code_fence")

    if response_for_parsing.strip() != raw_response.strip():
        warnings.append("response_was_cleaned_before_parsing")

    if parsed_output is not None and not isinstance(parsed_output, dict):
        warnings.append("parsed_output_not_dict")

    if debug_info is not None:
        output_token_count = debug_info.get("output_token_count")
        if isinstance(output_token_count, int) and output_token_count >= max_new_tokens:
            warnings.append("output_token_count_hit_max_new_tokens")

    return warnings


def remove_internal_thoughts_from_debug(
    debug_info: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if debug_info is None:
        return None
    cleaned = dict(debug_info)
    cleaned.pop("internal_thoughts", None)
    return cleaned


def get_internal_thoughts(
    debug_info: Optional[dict[str, Any]],
    save_internal_thoughts: bool,
) -> Optional[str]:
    if not save_internal_thoughts or debug_info is None:
        return None
    thoughts = debug_info.get("internal_thoughts")
    return thoughts if isinstance(thoughts, str) and thoughts.strip() else None


def print_local_model_summary(model_name: str, model_io: Any, model: Any) -> None:
    """Print local model metadata exposed by `model_utils.py`."""
    from src.utils.model_utils import get_model_input_device, get_model_io_info

    io_info = get_model_io_info(model_name, model_io)
    print(f"Model input device: {get_model_input_device(model)}", flush=True)
    print(f"Local backend: {getattr(model, '_local_backend', 'unsloth')}", flush=True)
    print(f"Model loader: {getattr(model, '_local_loader', 'unknown')}", flush=True)
    print(f"Model quantization: {getattr(model, '_local_quantization', 'unknown')}", flush=True)
    print(
        f"Requested model name: {getattr(model, '_requested_model_name', model_name)}",
        flush=True,
    )
    print(
        f"Resolved model name: {getattr(model, '_resolved_model_name', model_name)}",
        flush=True,
    )
    print(
        f"Model max seq length: {getattr(model, '_max_seq_length', 'unknown')}",
        flush=True,
    )
    print(f"Model family: {io_info['family']}", flush=True)
    print(f"Model IO type: {io_info['io_type']}", flush=True)
    print(f"Has chat template: {io_info['has_chat_template']}", flush=True)


def print_generation_debug(
    prefix: str,
    debug_info: Optional[dict[str, Any]],
) -> None:
    if debug_info is None:
        return

    print(
        f"{prefix} | "
        f"chars={debug_info.get('input_char_count')} | "
        f"in_tokens={debug_info.get('input_token_count')} | "
        f"out_tokens={debug_info.get('output_token_count')} | "
        f"time={debug_info.get('generation_time_sec', 0.0):.2f}s | "
        f"device={debug_info.get('model_device')} | "
        f"loader={debug_info.get('loader')} | "
        f"handler={debug_info.get('handler')} | "
        f"quantization={debug_info.get('quantization')} | "
        f"reasoning_effort={debug_info.get('reasoning_effort')} | "
        f"gemma_thinking={debug_info.get('gemma_enable_thinking')} | "
        f"rep_penalty={debug_info.get('repetition_penalty')} | "
        f"no_repeat_ngram={debug_info.get('no_repeat_ngram_size')}",
        flush=True,
    )
