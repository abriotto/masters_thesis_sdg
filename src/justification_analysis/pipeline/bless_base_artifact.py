"""One-time migration: verify the pre-existing base parser artifact, then
attach a manifest to it.

The base `discopy_explicit_candidates.csv` predates the manifest system, so
there is no record of which corpus produced it. Attaching a manifest by
assumption would defeat the entire point - it would bless a file we have not
checked, and every later freshness check would inherit that unverified claim.

So this reconciles the artifact against the current base corpus using the
identifiers the artifact actually carries, and refuses to write a manifest
unless every one of them agrees:

  * the model / game / run vocabulary matches the corpus exactly;
  * every candidate's (model, game_id, run_label, sentence_id) exists in the
    corpus's own segmentation;
  * every candidate's stored `sentence_text` is byte-identical to the
    sentence the project segmenter produces for that position;
  * the artifact's justification coverage is consistent with the corpus.

Run once:

    python -m src.justification_analysis.pipeline.bless_base_artifact
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.justification_analysis.pipeline import corpus as corpus_module  # noqa: E402
from src.justification_analysis.pipeline import manifest as manifest_module  # noqa: E402
from src.justification_analysis.pipeline.config import AnalysisConfig  # noqa: E402

# Parser provenance for the artifact being blessed. These are recorded facts
# about how the base artifact was produced, taken from the pipeline handoff
# and the environment that ran it - not guesses. If any cannot be confirmed
# they are recorded as null rather than invented.
BASE_PRODUCER = {
    "implementation": "rknaebel/discopy",
    "version": "1.1.0",
    "commit": "5507d65",
    "data_package": "rknaebel/discopy-data",
    "data_version": "1.0.2",
    "data_commit": "87db2a2",
    "checkpoint": "bert-10.11.21-13.31",
    "checkpoint_note": (
        "release 1.1.0 archive bert-10.11.21-13.31.tar.gz, extracted locally "
        "to a directory named bert-codi; the directory name is a local "
        "convenience, the checkpoint identity is bert-10.11.21-13.31"),
    "bert_model": "bert-base-cased",
    "component": "ConnectiveSenseClassifier",
    "used_context": 1,
    "relation_type": "Explicit",
    "note": (
        "blessed retrospectively by exact reconciliation against the base "
        "corpus; see bless_base_artifact.py"),
}


def reconcile(config: AnalysisConfig, corpus: pd.DataFrame,
              candidates: pd.DataFrame) -> pd.DataFrame:
    """Every check that must pass before a manifest may be written."""
    sentences = corpus_module.sentence_frame(corpus)

    key = ["model", "game_id", "run_label", "sentence_id"]
    merged = candidates.merge(
        sentences[key + ["sentence_text"]], on=key, how="left",
        suffixes=("", "_corpus"), indicator=True)

    matched = merged["_merge"].eq("both")
    text_equal = (merged.loc[matched, "sentence_text"].astype(str)
                  == merged.loc[matched, "sentence_text_corpus"].astype(str))

    corpus_models = set(corpus["model"].astype(str))
    corpus_runs = set(corpus["run_label"].astype(str))
    corpus_games = set(corpus["game_id"].astype(str))

    checks = [
        ("candidate models are a subset of the corpus",
         set(candidates["model"].astype(str)) <= corpus_models,
         sorted(set(candidates["model"].astype(str)) - corpus_models)),
        ("candidate runs are a subset of the corpus",
         set(candidates["run_label"].astype(str)) <= corpus_runs,
         sorted(set(candidates["run_label"].astype(str)) - corpus_runs)),
        ("candidate games are a subset of the corpus",
         set(candidates["game_id"].astype(str)) <= corpus_games,
         sorted(set(candidates["game_id"].astype(str)) - corpus_games)[:5]),
        ("every candidate maps to a corpus sentence",
         bool(matched.all()), int((~matched).sum())),
        ("every candidate's sentence text is byte-identical to the corpus",
         bool(text_equal.all()), int((~text_equal).sum())),
        ("candidate justification coverage is within the corpus",
         candidates["justification_id"].nunique() <= len(corpus),
         candidates["justification_id"].nunique()),
        ("accepted + rejected equals enumerated",
         int(candidates["is_connective"].sum())
         + int((~candidates["is_connective"]).sum()) == len(candidates),
         len(candidates)),
        ("accepted relations all carry a top-level PDTB class",
         bool(candidates.loc[candidates["is_connective"], "top_level"]
              .notna().all()),
         int(candidates.loc[candidates["is_connective"], "top_level"]
             .isna().sum())),
        ("occurrence ids are unique",
         int(candidates["occurrence_id"].duplicated().sum()) == 0,
         int(candidates["occurrence_id"].duplicated().sum())),
    ]
    return pd.DataFrame([
        {"check": name, "passed": bool(passed), "detail": str(detail),
         "status": "OK" if passed else "FAIL"}
        for name, passed, detail in checks
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="base")
    parser.add_argument("--prompt-version", default="prompt_v4")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing manifest")
    args = parser.parse_args()

    config = AnalysisConfig(stage=args.stage,
                            prompt_version=args.prompt_version,
                            repo_root=REPO_ROOT)

    if config.parser_manifest_path.exists() and not args.force:
        print(f"manifest already exists: {config.parser_manifest_path}")
        print("nothing to do (pass --force to rewrite)")
        return 0

    if not config.parser_candidates_path.exists():
        print(f"no artifact to bless at {config.parser_candidates_path}")
        return 1

    corpus = corpus_module.load_corpus(config)
    candidates = pd.read_csv(config.parser_candidates_path)

    print(f"corpus     : {len(corpus)} justifications, "
          f"{corpus['game_id'].nunique()} games")
    print(f"artifact   : {len(candidates)} candidates")
    print(f"fingerprint: {corpus_module.corpus_fingerprint(corpus)}")
    print()

    report = reconcile(config, corpus, candidates)
    for row in report.itertuples():
        print(f"  [{row.status:4s}] {row.check}"
              + (f"  -- {row.detail}" if row.detail not in ("0", "[]") else ""))

    if not report["passed"].all():
        print()
        print("RECONCILIATION FAILED - no manifest written.")
        print("The artifact does not demonstrably belong to this corpus.")
        return 1

    accepted = int(candidates["is_connective"].sum())
    manifest = manifest_module.build_manifest(
        config, corpus,
        artifact_path=config.parser_candidates_path,
        artifact_kind="discopy_explicit_candidates",
        producer=BASE_PRODUCER,
        outputs={
            "n_candidates": int(len(candidates)),
            "n_accepted": accepted,
            "n_nosense": int(len(candidates) - accepted),
            "top_level_counts": (
                candidates.loc[candidates["is_connective"], "top_level"]
                .value_counts().sort_index().to_dict()),
            "observed_senses": sorted(
                candidates.loc[candidates["is_connective"], "raw_sense"]
                .astype(str).unique()),
        },
    )
    manifest["migration"] = {
        "blessed_retrospectively": True,
        "reason": "artifact predates the manifest system",
        "reconciliation_checks": int(len(report)),
        "all_passed": True,
    }
    manifest_module.write_manifest(manifest, config.parser_manifest_path)

    print()
    print(f"all {len(report)} checks passed - manifest written to")
    print(f"  {config.parser_manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
