"""Deliberate tests of the failure mode the manifest system exists to prevent.

Not a unit test of happy paths. Each case below is a way the old pipeline
would have silently produced results about the wrong corpus, and each must now
stop the analysis.

    python -m src.justification_analysis.pipeline.test_stale_artifact_protection
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.justification_analysis.pipeline import corpus as corpus_module  # noqa: E402
from src.justification_analysis.pipeline import manifest as manifest_module  # noqa: E402
from src.justification_analysis.pipeline.config import AnalysisConfig  # noqa: E402

results = []


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f"\n         {detail}" if detail else ""))


def main() -> int:
    config = AnalysisConfig(stage="base", repo_root=REPO_ROOT)
    corpus = corpus_module.load_corpus(config)
    fingerprint = corpus_module.corpus_fingerprint(corpus)

    print("Stale-artifact protection tests")
    print(f"base corpus fingerprint: {fingerprint}\n")

    # -- 1. the genuine base artifact is accepted --------------------------
    print("1. the current base artifact passes the freshness gate")
    try:
        candidates, manifest = manifest_module.load_verified_candidates(
            config, corpus)
        record("base artifact accepted",
               len(candidates) == 14209,
               f"{len(candidates)} candidates, manifest fingerprint matches")
    except Exception as error:
        record("base artifact accepted", False, f"{type(error).__name__}: {error}")

    # -- 2. one altered justification must invalidate it -------------------
    print("\n2. altering ONE justification changes the fingerprint")
    tampered = corpus.copy()
    index = tampered.index[0]
    tampered.loc[index, "justification"] = (
        tampered.loc[index, "justification"] + " (edited)")
    tampered_fingerprint = corpus_module.corpus_fingerprint(tampered)
    record("fingerprint changes after a one-character edit",
           tampered_fingerprint != fingerprint,
           f"{tampered_fingerprint[:16]}... != {fingerprint[:16]}...")

    print("\n3. the analysis REFUSES the old artifact against that corpus")
    try:
        manifest_module.load_verified_candidates(config, tampered)
        record("stale artifact refused", False,
               "it was accepted - the gate does not work")
    except manifest_module.StaleArtifactError as error:
        message = str(error)
        record("stale artifact refused", True,
               "raised StaleArtifactError")
        record("refusal names the regeneration command",
               "run_discopy_on_justifications.py" in message)
        record("refusal states it will not fall back",
               "will not fall back" in message)
    except Exception as error:
        record("stale artifact refused", False,
               f"wrong exception: {type(error).__name__}: {error}")

    # -- 4. a stage with no corpus at all ----------------------------------
    print("\n4. a stage with no corpus stops at load time, with no fallback")
    finetuned = AnalysisConfig(stage="finetuned", repo_root=REPO_ROOT)
    try:
        corpus_module.load_corpus(finetuned)
        record("missing stage refused at load", False,
               "a corpus was returned for a stage that has no inputs")
    except FileNotFoundError as error:
        message = str(error)
        record("missing stage refused at load", True,
               "raised FileNotFoundError")
        record("error names the stage that was looked for",
               "finetuned" in message)
        record("error does not silently substitute base",
               "falling back" in message or "stops here" in message)

    # -- 5. and never reaches for the base CSV -----------------------------
    print("\n5. a stage with no parser artifact never reads the base CSV")
    record("finetuned artifact path is a different file from base",
           finetuned.parser_candidates_path != config.parser_candidates_path,
           f"{finetuned.parser_candidates_path.relative_to(REPO_ROOT)}")
    record("finetuned artifact does not exist",
           not finetuned.parser_candidates_path.exists())
    try:
        # Bypass the corpus load (stage 4 already proves that stops) and ask
        # the gate directly, to prove IT also refuses rather than resolving to
        # the base file.
        manifest_module.verify_artifact(
            finetuned, corpus,
            finetuned.parser_candidates_path, finetuned.parser_manifest_path,
            "discopy_explicit_candidates")
        record("missing artifact refused", False, "it was accepted")
    except manifest_module.MissingArtifactError as error:
        record("missing artifact refused", True, "raised MissingArtifactError")
        record("refusal prints the regeneration command for THIS stage",
               "--stage finetuned" in str(error))

    # -- 6. an artifact whose file changed after its manifest ---------------
    print("\n6. an artifact edited after its manifest is rejected on row count")
    manifest = manifest_module.read_manifest(config.parser_manifest_path)
    recorded = manifest["outputs"]["n_candidates"]
    record("manifest records the candidate count",
           recorded == 14209, f"n_candidates={recorded}")

    print()
    failed = [name for name, passed, _ in results if not passed]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print("Stale-artifact protection behaves as specified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
