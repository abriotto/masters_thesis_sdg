"""
Draw the pilot sample for validating the justification-annotation prompt.

The annotation scheme is defined by src/prompts/justification_annotation.txt.
That prompt is the authority; JUSTIFICATION_CODEBOOK.md is an earlier draft
and is out of date, so nothing here is derived from it.

Sampling frame: the three gemma sizes on prompt_v4 -- the ~2,290-justification
corpus that will eventually be annotated in full. gpt-oss ran on prompt v2/v3
and the fine-tuned probes have only 7 files between them, so neither is in the
frame: prompt revisions triggered by justifications outside the corpus would
not transfer.

Stratification: model x vote correctness, equal cells (largest-remainder
allocation). At most one justification per (model, game) so that four runs
over the same transcript cannot spend the pilot budget on near-duplicates.

Writes pilot_sample.jsonl -- the annotator input: vote + pre-split sentences.
The review sheet for checking the model's output is produced afterwards by
justification_pilot_report.py, once there is output to check.

Usage:
    python src/pt_annotation/justification_sample_pilot.py
    python src/pt_annotation/justification_sample_pilot.py --n 50 --seed 7
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.sentences import build_sentence_records  # noqa: E402
from src.pt_annotation.justification_schema import DEFAULT_SCHEMA, SCHEMAS  # noqa: E402


# ============================================================
# Configuration
# ============================================================

# Display name -> results/ and analysis/ folder name.
MODEL_FOLDERS = {
    "2B": "unsloth_gemma-4-E2B-it-unsloth-bnb-4bit",
    "4B": "unsloth_gemma-4-E4B-it-unsloth-bnb-4bit",
    "31B": "unsloth_gemma-4-31B-it-unsloth-bnb-4bit",
}

MODEL_ORDER = ["2B", "4B", "31B"]

BASE_STAGE = "base"

# Relative to the model's STAGE folder, not to the model folder. The stage is
# resolved separately because fine-tuned folders carry the adapter name and so
# differ per model (ft_gemma-4-E2B-..., ft_gemma-4-E4B-...); a single literal
# stage string cannot address all three.
VOTE_TABLE_REL = Path("voting/prompt_v4/vote_stability/tables/llm_vote_file_level.csv")

DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "analysis"
RESULTS_ROOT = REPO_ROOT / "results" / "justification_annotation"

# The sample itself is schema-independent -- the same 40 justifications, split
# the same way. It is written per schema anyway so each pilot folder is
# self-contained, and because the seed is fixed the two samples are identical
# by construction: v1 and v2 output can be compared row for row.

DEFAULT_N = 40
DEFAULT_SEED = 42


# ============================================================
# Frame construction
# ============================================================

def resolve_stage_dir(analysis_root, model_name, stage):
    """The model's folder for `stage`, matched as a prefix.

    "base" matches `base`; "ft" matches `ft_gemma-4-E2B-...` and its siblings.
    Exactly one match is required, so a stage name that is ambiguous or absent
    stops here rather than quietly annotating the wrong corpus.
    """
    model_dir = analysis_root / MODEL_FOLDERS[model_name]
    matches = sorted(path for path in model_dir.glob(f"{stage}*") if path.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"stage {stage!r} matched {len(matches)} folders under {model_dir}: "
            f"{[path.name for path in matches]}. Expected exactly one."
        )
    return matches[0]


def load_frame(analysis_root, stage=BASE_STAGE):
    """One row per annotatable justification across the three models."""
    frames = []

    for model_name in MODEL_ORDER:
        table_path = resolve_stage_dir(analysis_root, model_name, stage) / VOTE_TABLE_REL
        if not table_path.exists():
            raise FileNotFoundError(
                f"Vote table not found for {model_name} at stage {stage!r}: {table_path}"
            )

        model_frame = pd.read_csv(table_path)
        model_frame["model"] = model_name
        frames.append(model_frame)

    frame = pd.concat(frames, ignore_index=True)

    # failed_parse rows carry no justification at all, and is_correct is NaN
    # for them, so they belong to neither stratum.
    frame = frame[frame["status"].ne("failed_parse")].copy()
    frame = frame[frame["justification"].notna()].copy()
    frame["justification"] = frame["justification"].astype(str).str.strip()
    frame = frame[frame["justification"].ne("")].copy()
    frame = frame[frame["is_correct"].notna()].copy()

    frame["is_correct"] = frame["is_correct"].astype(bool)

    # "No Werewolf" is a legitimate vote under prompt_v4 and is kept: the
    # reasoning behind it (typically Deduction over the deck) is exactly the
    # kind of case the prompt needs to survive.
    frame["vote"] = frame["chosen_vote_raw"].astype(str).str.strip()

    frame["sentences"] = frame["justification"].map(build_sentence_records)
    frame["n_sentences"] = frame["sentences"].map(len)
    frame = frame[frame["n_sentences"].gt(0)].copy()

    # A stable identity for the row, independent of file paths, so the same
    # seed reproduces the same sample on any checkout.
    frame["justification_id"] = (
        frame["model"] + "__"
        + frame["source"].astype(str) + "__"
        + frame["session_name"].astype(str) + "__"
        + frame["game_key"].astype(str) + "__"
        + frame["run_label"].astype(str)
    )

    return frame.sort_values("justification_id").reset_index(drop=True)


def load_excluded_ids(paths):
    """justification_ids already spent on earlier pilots.

    A prompt developed against a sample cannot be validated on that same
    sample: whatever wording was changed in response to those 40 will look
    right on those 40. Excluding them is what makes the frozen run a
    validation rather than a second look at the development set.
    """
    excluded, games = set(), set()
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"--exclude file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    excluded.add(record["justification_id"])
                    games.add(record["game_id"])
    return excluded, games


def allocate(n_total, n_cells):
    """Largest-remainder allocation, so cells differ by at most one."""
    base, remainder = divmod(n_total, n_cells)
    return [base + (1 if index < remainder else 0) for index in range(n_cells)]


def draw_sample(frame, n_total, seed):
    """Stratify by model x is_correct, one justification per (model, game)."""
    cells = [
        (model_name, is_correct)
        for model_name in MODEL_ORDER
        for is_correct in (False, True)
    ]
    quotas = allocate(n_total, len(cells))

    # Collapse the four runs first: pick one run per (model, game) at random,
    # then sample games. Doing it in this order keeps every game equally
    # likely regardless of how many runs produced a parseable vote.
    one_per_game = (
        frame
        .groupby(["model", "game_id"], sort=True, group_keys=False)
        .sample(n=1, random_state=seed)
        .reset_index(drop=True)
    )

    drawn = []
    shortfalls = []

    for (model_name, is_correct), quota in zip(cells, quotas):
        cell = one_per_game[
            one_per_game["model"].eq(model_name)
            & one_per_game["is_correct"].eq(is_correct)
        ]

        take = min(quota, len(cell))
        if take < quota:
            shortfalls.append(
                f"{model_name}/is_correct={is_correct}: wanted {quota}, only {len(cell)} available"
            )

        if take:
            drawn.append(cell.sample(n=take, random_state=seed))

    sample = pd.concat(drawn, ignore_index=True)
    return sample.sort_values("justification_id").reset_index(drop=True), shortfalls


# ============================================================
# Output
# ============================================================

def write_annotator_input(sample, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sample.itertuples(index=False):
            record = {
                "justification_id": row.justification_id,
                "model": row.model,
                "source": row.source,
                "session_name": row.session_name,
                "game_key": row.game_key,
                "game_id": row.game_id,
                "run_label": row.run_label,
                "decoding": row.decoding,
                "status": row.status,
                "is_correct": bool(row.is_correct),
                "voted_player_end_role": (
                    None if pd.isna(row.voted_player_end_role) else row.voted_player_end_role
                ),
                "vote": row.vote,
                "justification": row.justification,
                "sentences": row.sentences,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw the stratified pilot sample for the justification-annotation prompt."
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, choices=sorted(SCHEMAS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Total justifications to draw.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--exclude-games", action="store_true",
        help="Also exclude every justification from a GAME used in --exclude. "
             "Stricter: without it, the same transcript can reappear via a "
             "different model or run, which shares content with the "
             "development set even though the text differs.",
    )
    parser.add_argument(
        "--exclude", type=Path, nargs="*", default=None,
        help="Sample JSONL file(s) whose justifications must NOT be drawn. Use "
             "this to keep a validation sample disjoint from the pilots the "
             "prompt was developed on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = RESULTS_ROOT / f"pilot_{args.schema}"

    frame = load_frame(args.analysis_root)
    n_corpus = len(frame)

    excluded, excluded_games = load_excluded_ids(args.exclude or [])
    if excluded:
        frame = frame[~frame["justification_id"].isin(excluded)].copy()
        note = f"Excluded {len(excluded)} already-used justifications"
        if args.exclude_games:
            frame = frame[~frame["game_id"].isin(excluded_games)].copy()
            note += f" and every justification from their {len(excluded_games)} games"
        print(f"{note} ({n_corpus} -> {len(frame)} eligible)")

    print(f"Frame: {len(frame)} justifications across {frame['model'].nunique()} models")
    print(frame.groupby(["model", "is_correct"]).size().to_string())
    print()

    sample, shortfalls = draw_sample(frame, args.n, args.seed)

    for message in shortfalls:
        print(f"Warning: {message}")

    jsonl_path = args.output_dir / "pilot_sample.jsonl"
    write_annotator_input(sample, jsonl_path)

    # Provenance beside the sample: how it was drawn, and what it was kept
    # disjoint from. Without this the disjointness claim is unverifiable later.
    meta_path = args.output_dir / "sample_meta.json"
    meta_path.write_text(json.dumps({
        "schema": args.schema,
        "n_requested": args.n,
        "n_drawn": int(len(sample)),
        "seed": args.seed,
        "corpus_size": n_corpus,
        "eligible_after_exclusions": int(len(frame)),
        "excluded_sources": [str(p) for p in (args.exclude or [])],
        "n_excluded": len(excluded),
        "excluded_games": sorted(excluded_games) if args.exclude_games else [],
        "game_level_exclusion": bool(args.exclude_games),
        "stratified_by": ["model", "is_correct"],
        "justification_ids": sample["justification_id"].tolist(),
    }, indent=2) + chr(10), encoding="utf-8")

    print(f"Sampled {len(sample)} justifications, {int(sample['n_sentences'].sum())} sentences")
    print(sample.groupby(["model", "is_correct"]).size().to_string())
    print()
    print(
        "Sentences per justification: "
        f"min={sample['n_sentences'].min()} "
        f"median={sample['n_sentences'].median():.0f} "
        f"max={sample['n_sentences'].max()}"
    )
    print()
    print(f"Annotator input : {jsonl_path}")
    print(f"Draw provenance : {meta_path}")


if __name__ == "__main__":
    main()
