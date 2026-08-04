from __future__ import annotations

"""
Build the role-inference SFT dataset from the preprocessed Jin et al. transcripts.

One example per episode (120 total). The prompt is assembled with the same
`build_full_prompt` layout the voting task uses, so the only intended difference
between finetuning and evaluation is the task instruction itself.

Target format
-------------
    {"roles": {"player1": "Troublemaker", ..., "player5": "Villager"}}

End roles, i.e. after the Night phase swaps are resolved - the same notion of "role"
that decides the winner, and the one the voting task cares about.

Deliberately NOT included: a reasoning / thinking block. The targets are answer-only.
Training on reasoning traces would contaminate the dependent variable - the research
question is whether familiarisation changes the model's reasoning on the voting task,
which is not answerable if the finetune taught the reasoning directly. The trainer must
therefore render the chat template in NON-thinking mode; pairing thinking mode with
these answer-only targets would train the model to emit empty thought blocks.

Augmentation was considered and rejected: round-prefix truncation makes the target
unanswerable at short prefixes (the label is fixed regardless of how much transcript is
shown), the prefixes are nested rather than independent, and the voting task always
shows the full transcript.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from src.utils.io_utils import find_repo_root, load_json, load_text
from src.utils.prompt_utils import build_full_prompt


ROLE_PATTERN = re.compile(r"^- ([A-Z][A-Za-z]+):", re.MULTILINE)


def extract_known_roles(rules_text: str) -> set[str]:
    """Role vocabulary, read off the rules file so the two cannot disagree."""
    roles = set(ROLE_PATTERN.findall(rules_text))
    if not roles:
        raise ValueError("Could not extract any role names from the rules text.")
    return roles


def build_target(player_names: list[str], end_roles: list[str]) -> str:
    if len(player_names) != len(end_roles):
        raise ValueError("player_names and end_roles have different lengths.")
    return json.dumps({"roles": dict(zip(player_names, end_roles))}, ensure_ascii=False)


def stratified_split(
    records: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split at episode level, stratified by the number of end-game werewolves.

    Stratification matters here because the 0-werewolf case has only 3 episodes; an
    unstratified draw could put all or none of them in validation.
    """
    rng = random.Random(seed)
    buckets: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(record["num_werewolves_end"], []).append(record)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []

    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda r: r["session_name"])
        rng.shuffle(bucket)
        n_val = round(len(bucket) * val_fraction)
        val.extend(bucket[:n_val])
        train.extend(bucket[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def build_example(
    record: dict[str, Any],
    repo_root: Path,
    base_prompt: str,
    rules_text: str,
    known_roles: set[str],
) -> dict[str, Any]:
    transcript_text = load_text(repo_root / record["processed_txt_path"])

    unknown = sorted(set(record["end_roles"]) - known_roles)
    if unknown:
        raise ValueError(
            f"{record['session_name']}: end roles absent from the rules file: {unknown}"
        )

    prompt = build_full_prompt(
        base_prompt=base_prompt,
        rules_text=rules_text,
        player_names=record["player_names"],
        transcript_text=transcript_text,
    )
    target = build_target(record["player_names"], record["end_roles"])

    return {
        "session_name": record["session_name"],
        "source": record["source"],
        "processed_txt_path": record["processed_txt_path"],
        "player_names": record["player_names"],
        "end_roles": record["end_roles"],
        "num_werewolves_end": record["num_werewolves_end"],
        "prompt": prompt,
        "completion": target,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "prompt_chars": len(prompt),
        "completion_chars": len(target),
    }


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the role-inference SFT dataset from preprocessed Jin episodes."
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="data/processed/jin2024_onuw/role_inference/index.json",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="src/prompts/role_inference_prompt_v1.txt",
    )
    parser.add_argument("--rules_path", type=str, default="src/prompts/onuw_rules_v2.txt")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/jin2024_onuw/sft_role_inference",
    )
    parser.add_argument("--val_fraction", type=float, default=0.167, help="~20 of 120.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    index = load_json(repo_root / args.index_path)
    base_prompt = load_text(repo_root / args.prompt_path)
    rules_text = load_text(repo_root / args.rules_path)
    known_roles = extract_known_roles(rules_text)

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = [
        build_example(
            record=record,
            repo_root=repo_root,
            base_prompt=base_prompt,
            rules_text=rules_text,
            known_roles=known_roles,
        )
        for record in index
    ]

    train, val = stratified_split(examples, args.val_fraction, args.seed)

    train_sessions = {e["session_name"] for e in train}
    val_sessions = {e["session_name"] for e in val}
    overlap = train_sessions & val_sessions
    if overlap:
        raise AssertionError(f"Episode leakage between train and val: {sorted(overlap)}")

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)

    def _length_stats(items: list[dict[str, Any]]) -> dict[str, int]:
        chars = [e["prompt_chars"] + e["completion_chars"] for e in items]
        return {
            "min_chars": min(chars),
            "max_chars": max(chars),
            "approx_max_tokens": max(chars) // 4,
        }

    def _wolf_dist(items: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in items:
            counts[str(e["num_werewolves_end"])] = counts.get(str(e["num_werewolves_end"]), 0) + 1
        return dict(sorted(counts.items()))

    stats = {
        "num_examples": len(examples),
        "num_train": len(train),
        "num_val": len(val),
        "prompt_path": args.prompt_path,
        "rules_path": args.rules_path,
        "known_roles": sorted(known_roles),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_lengths": _length_stats(train),
        "val_lengths": _length_stats(val),
        "train_werewolf_distribution": _wolf_dist(train),
        "val_werewolf_distribution": _wolf_dist(val),
        "targets_include_reasoning": False,
        "augmentation": "none",
    }
    (output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Train: {len(train)}  Val: {len(val)}  (episode-level, stratified by werewolf count)")
    print(f"Werewolves/episode - train {_wolf_dist(train)}  val {_wolf_dist(val)}")
    print(f"Longest example: ~{max(stats['train_lengths']['approx_max_tokens'], stats['val_lengths']['approx_max_tokens'])} tokens")
    print(f"Roles known from rules: {len(known_roles)}")
    print(f"Written to: {output_dir}")


if __name__ == "__main__":
    main()
