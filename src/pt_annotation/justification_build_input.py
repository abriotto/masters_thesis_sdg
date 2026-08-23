"""
Build the full-corpus annotation input, pre-split into shards.

The pilot annotated 40 justifications from one file. The full run annotates
every justification in the prompt_v4 corpus -- 3 models x 4 runs x 191 games
-- which at the pilot's measured 68 s per call is 43 h if run serially. So it
is sharded, one shard per (model, run) pair: 12 shards of 191 calls, ~3.6 h
each, run as a SLURM job array.

Sharding by (model, run) rather than by an arbitrary index is deliberate. A
shard is then a meaningful unit: if one model-run needs re-annotating after a
prompt change, it is one file and one array index, not a slice cutting across
everything.

Pre-splitting also means the annotation runner needs no sharding logic at
all. Each array task just points --input-path and --output-path at its own
files, and --resume keeps it idempotent per shard.

Layout:

  results/justification_annotation/full_<schema>/
      manifest.json              shard index -> file, with counts
      input/<model>__<run>.jsonl one line per justification
      annotations/<...>.jsonl    written later by the runner

Usage:
    python src/pt_annotation/justification_build_input.py
    python src/pt_annotation/justification_build_input.py --schema v3
    python src/pt_annotation/justification_build_input.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.pt_annotation.justification_schema import DEFAULT_SCHEMA, SCHEMAS  # noqa: E402
from src.pt_annotation.justification_sample_pilot import (  # noqa: E402
    DEFAULT_ANALYSIS_ROOT,
    MODEL_ORDER,
    load_frame,
)

RESULTS_ROOT = REPO_ROOT / "results" / "justification_annotation"

# Fixed order so shard index -> (model, run) is stable across rebuilds. An
# array task id must always mean the same shard, or a resubmit of index 7
# would annotate a different slice than the run it is meant to finish.
RUN_ORDER = ["greedy_t0", "run_1", "run_2", "run_3"]


def shard_key(model, run_label):
    return f"{model}__{run_label}"


def build_shards(frame):
    """(model, run) -> DataFrame, in the fixed order above."""
    shards = {}
    for model in MODEL_ORDER:
        for run_label in RUN_ORDER:
            subset = frame[
                frame["model"].eq(model) & frame["run_label"].eq(run_label)
            ].sort_values("justification_id")
            shards[shard_key(model, run_label)] = subset
    return shards


def write_shard(subset, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in subset.itertuples(index=False):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the sharded full-corpus input for justification annotation."
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, choices=sorted(SCHEMAS))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--seconds-per-call", type=float, default=68.0,
        help="Measured from the pilot; used only for the wall-time estimate.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the shards without writing.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_root = args.output_root or (RESULTS_ROOT / f"full_{args.schema}")
    input_dir = output_root / "input"

    frame = load_frame(args.analysis_root)
    shards = build_shards(frame)

    covered = sum(len(subset) for subset in shards.values())
    if covered != len(frame):
        # Every justification must land in exactly one shard. A mismatch means
        # a run_label exists that RUN_ORDER does not list, and those
        # justifications would be silently dropped from the corpus.
        missing = sorted(set(frame["run_label"]) - set(RUN_ORDER))
        raise ValueError(
            f"Shards cover {covered} of {len(frame)} justifications. "
            f"run_label values not in RUN_ORDER: {missing}"
        )

    manifest = {
        "schema": args.schema,
        "n_justifications": int(len(frame)),
        "n_shards": len(shards),
        "shards": [],
    }

    print(f"Corpus: {len(frame)} justifications, {len(shards)} shards\n")
    print(f"{'idx':>3}  {'shard':<16} {'n':>5}  {'est. wall':>9}")
    print("-" * 40)

    for index, (key, subset) in enumerate(shards.items()):
        relative = f"input/{key}.jsonl"
        hours = len(subset) * args.seconds_per_call / 3600

        if not args.dry_run:
            write_shard(subset, output_root / relative)

        manifest["shards"].append({
            "index": index,
            "key": key,
            "model": key.split("__")[0],
            "run_label": key.split("__", 1)[1],
            "n_justifications": int(len(subset)),
            "input": relative,
            "output": f"annotations/{key}.jsonl",
        })
        print(f"{index:>3}  {key:<16} {len(subset):>5}  {hours:>8.1f}h")

    total_hours = len(frame) * args.seconds_per_call / 3600
    print("-" * 40)
    print(f"     {'TOTAL':<16} {len(frame):>5}  {total_hours:>8.1f}h serial")
    print(f"     {'':<16} {'':>5}  {total_hours/len(shards):>8.1f}h with all {len(shards)} shards in parallel")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nInput  : {input_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nRun with: sbatch --array=0-{len(shards) - 1} "
          f"slurm_files/justification_annotation_full.slurm")


if __name__ == "__main__":
    main()
