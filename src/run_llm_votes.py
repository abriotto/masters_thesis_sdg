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


def extract_json_object(text: str) -> Optional[dict]:
    """
    Try to extract the first JSON object from the model response.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def call_gemini(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return (response.text or "").strip()


def load_local_model(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def call_local_model(model, tokenizer, prompt: str, max_new_tokens: int = 700) -> str:
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
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
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

            parsed_output = extract_json_object(raw_response)

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
            }

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[{i}/{len(rows)}] OK - {source}/{transcript_stem}.json")

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