import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from google import genai


def find_repo_root(start: Optional[Path] = None, repo_name: str = "masters_thesis_sdg") -> Path:
    if start is None:
        start = Path.cwd().resolve()

    current = start
    while True:
        if current.name == repo_name:
            return current
        if current.parent == current:
            raise FileNotFoundError(f"Could not find repo root '{repo_name}' from {start}")
        current = current.parent


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_full_prompt(base_prompt: str, player_names: list[str], transcript_text: str) -> str:
    players_str = ", ".join(player_names)

    return f"""{base_prompt}

## Player list

{players_str}

## Transcript

{transcript_text}
""".strip()


def extract_first_json_block(text: str) -> Optional[str]:
    """
    Try to extract a JSON object from model output.

    Priority:
    1. fenced ```json ... ```
    2. fenced ``` ... ```
    3. first balanced {...} block
    """
    fenced_json = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_json:
        return fenced_json.group(1)

    fenced_any = re.search(r"```\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced_any:
        return fenced_any.group(1)

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def parse_model_json(text: str) -> Optional[dict]:
    candidate = extract_first_json_block(text)
    if candidate is None:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def validate_vote_output(obj: Optional[dict], player_names: list[str]) -> dict:
    report = {
        "is_valid": False,
        "errors": [],
        "chosen_vote": None,
        "reasoning": None,
    }

    if obj is None:
        report["errors"].append("no_parseable_json")
        return report

    if not isinstance(obj, dict):
        report["errors"].append("parsed_output_not_dict")
        return report

    if "chosen_vote" not in obj:
        report["errors"].append("missing_chosen_vote")
    if "reasoning" not in obj:
        report["errors"].append("missing_reasoning")

    chosen_vote = obj.get("chosen_vote")
    reasoning = obj.get("reasoning")

    if not isinstance(chosen_vote, str) or not chosen_vote.strip():
        report["errors"].append("chosen_vote_not_nonempty_string")
    elif chosen_vote not in player_names:
        report["errors"].append(f"chosen_vote_not_in_player_list:{chosen_vote}")

    if not isinstance(reasoning, str) or not reasoning.strip():
        report["errors"].append("reasoning_not_nonempty_string")

    report["chosen_vote"] = chosen_vote
    report["reasoning"] = reasoning

    if not report["errors"]:
        report["is_valid"] = True

    return report


def add_soft_warnings(validation: dict, raw_response: str) -> list[str]:
    warnings = []

    reasoning = validation.get("reasoning")
    if isinstance(reasoning, str):
        low = reasoning.lower()

        if "confirmed" in low:
            warnings.append("reasoning_contains_confirmed_language")
        if "all players" in low or "everyone is" in low:
            warnings.append("reasoning_may_overgeneralize")
        if "likely a villager" in low and validation.get("chosen_vote") is not None:
            warnings.append("reasoning_may_target_player_described_as_villager")

    raw_low = raw_response.lower()
    if raw_low.count("{") > 1 or raw_low.count("```") > 0:
        warnings.append("response_contains_extra_wrapper_text")

    return warnings


def call_gemini(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return (response.text or "").strip()


def load_local_model(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def call_local_model(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = prompt

    model_device = next(model.parameters()).device
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
    ).to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index_path",
        type=str,
        default="data/processed/lai2023/onuw_transcripts_ready/index_cleaned.json",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/prompts/vote_prompt_v1.txt",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["gemini", "hf_local"],
        default="gemini",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--prompt_version",
        type=str,
        default="v1",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="voting",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=5,
        help="Number of games to process. Use -1 for all.",
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
    args = parser.parse_args()

    repo_root = find_repo_root()

    index_path = repo_root / args.index_path
    prompt_path = repo_root / args.prompt_path

    index_data = load_json(index_path)
    base_prompt = load_text(prompt_path)

    if args.max_games == -1:
        rows = index_data
    else:
        rows = index_data[: args.max_games]

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
    tokenizer = None
    model = None

    if args.backend == "gemini":
        client = genai.Client()
    elif args.backend == "hf_local":
        tokenizer, model = load_local_model(args.model_name)

    print(f"Loaded {len(rows)} games from {index_path}")
    print(f"Prompt file: {prompt_path}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_name}")
    print(f"Results root: {results_root}")

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
            player_names=row["player_names"],
            transcript_text=transcript_text,
        )

        try:
            if args.backend == "gemini":
                raw_response = call_gemini(
                    client=client,
                    model_name=args.model_name,
                    prompt=prompt,
                )
            elif args.backend == "hf_local":
                raw_response = call_local_model(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                )
            else:
                raise ValueError(f"Unsupported backend: {args.backend}")

            parsed_output = parse_model_json(raw_response)
            validation = validate_vote_output(parsed_output, row["player_names"])
            soft_warnings = add_soft_warnings(validation, raw_response)

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
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "validation": validation,
                "soft_warnings": soft_warnings,
            }

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            status = "OK" if validation["is_valid"] else "PARSED_BUT_INVALID"
            print(f"[{i}/{len(rows)}] {status} - {source}/{transcript_stem}.json")

        except Exception as e:
            error_result = {
                "source": row["source"],
                "session_name": row["session_name"],
                "game_key": row["game_key"],
                "processed_txt_path": row["processed_txt_path"],
                "player_names": row["player_names"],
                "backend": args.backend,
                "model_name": args.model_name,
                "prompt_version": args.prompt_version,
                "prompt_path": args.prompt_path,
                "error": str(e),
            }

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2, ensure_ascii=False)

            print(f"[{i}/{len(rows)}] ERROR - {source}/{transcript_stem}.json -> {e}")

        if args.backend == "gemini":
            time.sleep(args.sleep_sec)


if __name__ == "__main__":
    main()