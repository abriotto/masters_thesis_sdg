"""Run the explicit side of discopy over every prompt_v4 justification.

Executed in `discopy-env`, standalone, writing a CSV the notebook consumes.
The notebook itself runs in `sdglogs` and never imports TensorFlow.

    python src/justification_analysis/discopy_parser/run_discopy_on_justifications.py \
        --model-path <checkpoint dir> --out <csv path>

Every row is one connective CANDIDATE that discopy enumerated, including the
ones it rejected (`is_connective=False`). Keeping the rejects is the point:
they are the evidence for whether the parser discriminates connective from
non-connective uses of ambiguous forms, and they cannot be recovered from the
upstream API, which drops them.

Justification metadata (model, game, run, decoding regime, justification id)
and the project's deterministic sentence ids are preserved on every row so the
table can be joined with the DiMLex occurrence table and, later, with the
sentence-level DeepSeek annotations.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def find_repo_root(start=None, repo_name="masters_thesis_sdg"):
    current = (start or Path(__file__)).resolve()
    while current.name != repo_name:
        if current.parent == current:
            raise FileNotFoundError(f"repo root {repo_name!r} not found")
        current = current.parent
    return current


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.sentences import split_sentences  # noqa: E402
from src.justification_analysis.discopy_parser.discopy_explicit import (  # noqa: E402
    PDTB_TOP_LEVEL,
    build_document,
    load_explicit_component,
    parse_explicit,
    verify_against_upstream,
)

from src.justification_analysis.pipeline import corpus as corpus_module  # noqa: E402
from src.justification_analysis.pipeline import manifest as manifest_module  # noqa: E402
from src.justification_analysis.pipeline.config import AnalysisConfig  # noqa: E402


def _package_version(name: str):
    """Record what actually ran, rather than what the docs claim ran."""
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


def load_justifications(config: AnalysisConfig) -> pd.DataFrame:
    """The corpus for the configured stage.

    This used to be a second, hand-maintained copy of the analysis loader,
    carrying the comment "Reproduce the notebook's loading logic exactly,
    including id assignment". Two implementations of one load is exactly how a
    parser artifact silently stops matching the corpus it is used against, so
    there is now one: `pipeline.corpus.load_corpus`.
    """
    return corpus_module.load_corpus(config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--bert-model", default="bert-base-cased")
    ap.add_argument("--stage", default="base",
                    help="which corpus to parse; output goes to this stage's "
                         "own namespace and cannot overwrite another's")
    ap.add_argument("--prompt-version", default="prompt_v4")
    ap.add_argument("--out", default=None,
                    help="override the output path (default: derived from "
                         "--stage, which is what you normally want)")
    ap.add_argument("--limit", type=int, default=0,
                    help="parse only the first N justifications; produces a "
                         "PARTIAL artifact, which is marked as such in the "
                         "manifest and will be refused by the analyses")
    args = ap.parse_args()

    from discopy_data.nn.bert import get_sentence_embedder

    config = AnalysisConfig(stage=args.stage,
                            prompt_version=args.prompt_version,
                            repo_root=REPO_ROOT)
    corpus = load_justifications(config)
    fingerprint = corpus_module.corpus_fingerprint(corpus)
    print(f"stage       : {config.stage}", flush=True)
    print(f"prompt      : {config.prompt_version}", flush=True)
    print(f"corpus hash : {fingerprint}", flush=True)

    votes = corpus
    if args.limit:
        votes = votes.head(args.limit)
    print(f"justifications: {len(votes)}"
          + (f"  (LIMITED from {len(corpus)})" if args.limit else ""), flush=True)

    embedder = get_sentence_embedder(args.bert_model)
    component = load_explicit_component(args.model_path)
    print(f"component: {type(component).__name__}, "
          f"used_context={component.used_context}", flush=True)

    rows = []
    verified = False
    start = time.time()

    for position, vote in enumerate(votes.itertuples(index=False)):
        text = vote.justification
        sentences = split_sentences(text)
        if not sentences:
            continue

        doc = build_document(f"just_{vote.justification_id}", text, sentences)
        if not doc.sentences:
            continue
        for sent_i, sent in enumerate(doc.sentences):
            doc.sentences[sent_i].embeddings = embedder(sent.tokens)

        # Verify once, on the first real document, that the batched path
        # reproduces upstream's per-candidate predictions exactly.
        if not verified:
            verify_against_upstream(component, doc)
            print("batched parse verified against upstream parse()", flush=True)
            verified = True

        for hit in parse_explicit(component, doc, keep_nosense=True):
            sent_index = hit["sentence_index"]
            rows.append(
                {
                    "model": vote.model,
                    "game_id": vote.game_id,
                    "session_name": vote.session_name,
                    "run_label": vote.run_label,
                    "run_number": vote.run_number,
                    "decoding_group": vote.decoding_group,
                    "justification_id": vote.justification_id,
                    # 1-indexed, matching build_sentence_records()
                    "sentence_id": sent_index + 1,
                    "sentence_text": sentences[sent_index],
                    "candidate_surface": hit["candidate_surface"],
                    "connective_surface": hit["connective_surface"],
                    "char_spans": ";".join(f"{a}-{b}" for a, b in hit["char_spans"]),
                    "start": hit["char_spans"][0][0],
                    "end": hit["char_spans"][-1][1],
                    "n_char_spans": hit["n_char_spans"],
                    "is_discontinuous": hit["is_discontinuous"],
                    "is_connective": hit["is_connective"],
                    "raw_sense": hit["raw_sense"],
                    "top_level": hit["top_level"],
                    "confidence": hit["confidence"],
                    "relation_type": hit["type"],
                }
            )

        if (position + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (position + 1) / elapsed
            print(f"  {position + 1}/{len(votes)} "
                  f"({rate:.1f}/s, eta {(len(votes) - position - 1) / rate / 60:.1f} min)",
                  flush=True)

    occurrences = pd.DataFrame(rows)
    occurrences.insert(0, "occurrence_id", np.arange(len(occurrences)))

    # --- invariants -------------------------------------------------------
    assert occurrences["occurrence_id"].is_unique, "duplicate occurrence ids"
    assert (occurrences["relation_type"] == "Explicit").all(), \
        "a non-explicit relation entered the table"
    accepted = occurrences[occurrences["is_connective"]]
    assert accepted["top_level"].isin(PDTB_TOP_LEVEL).all() or \
        accepted["top_level"].isna().any(), "unexpected top-level label"
    assert set(occurrences["decoding_group"]) <= {"Stochastic", "Greedy"}

    out = Path(args.out) if args.out else config.parser_candidates_path
    out.parent.mkdir(parents=True, exist_ok=True)
    occurrences.to_csv(out, index=False, encoding="utf-8-sig")

    # The manifest is what makes this artifact usable. Without it the analyses
    # refuse to consume the CSV, by design: a candidate table with no record of
    # which corpus produced it is exactly the failure mode being eliminated.
    accepted_count = int(occurrences["is_connective"].sum())
    manifest = manifest_module.build_manifest(
        config, corpus,
        artifact_path=out,
        artifact_kind="discopy_explicit_candidates",
        producer={
            "implementation": "rknaebel/discopy",
            "component": "ConnectiveSenseClassifier",
            "checkpoint": str(args.model_path),
            "checkpoint_name": Path(args.model_path).name,
            "bert_model": args.bert_model,
            "used_context": getattr(component, "used_context", None),
            "relation_type": "Explicit",
            "discopy_version": _package_version("discopy"),
            "discopy_data_version": _package_version("discopy-data"),
            "transformers_version": _package_version("transformers"),
            "command": " ".join(sys.argv),
        },
        outputs={
            "n_candidates": int(len(occurrences)),
            "n_accepted": accepted_count,
            "n_nosense": int(len(occurrences) - accepted_count),
            "top_level_counts": (
                occurrences.loc[occurrences["is_connective"], "top_level"]
                .value_counts().sort_index().to_dict()),
            "observed_senses": sorted(
                occurrences.loc[occurrences["is_connective"], "raw_sense"]
                .astype(str).unique()),
            "partial": bool(args.limit),
            "n_justifications_parsed": int(len(votes)),
        },
    )
    if args.limit:
        manifest["corpus"]["fingerprint"] = (
            "PARTIAL-RUN-NOT-A-VALID-CORPUS-" + fingerprint)
        manifest["corpus"]["partial_run_note"] = (
            f"--limit {args.limit} was used, so this artifact covers only "
            f"{len(votes)} of {len(corpus)} justifications. The fingerprint is "
            f"deliberately poisoned so no analysis can consume it.")

    manifest_path = (out.parent / f"{out.stem}.manifest.json"
                     if args.out else config.parser_manifest_path)
    manifest_module.write_manifest(manifest, manifest_path)

    print(f"\nwrote {len(occurrences)} candidate rows -> {out}", flush=True)
    print(f"  manifest               : {manifest_path.name}", flush=True)
    print(f"  accepted as connectives: {accepted_count}", flush=True)
    print(f"  rejected (NoSense)     : {int((~occurrences['is_connective']).sum())}", flush=True)
    print(f"elapsed: {(time.time() - start) / 60:.1f} min", flush=True)
    print(occurrences.loc[occurrences["is_connective"], "top_level"]
          .value_counts(dropna=False).to_string(), flush=True)
    if args.limit:
        print("\nWARNING: --limit produced a PARTIAL artifact. Its manifest "
              "fingerprint is poisoned so the analyses will refuse it.",
              flush=True)


if __name__ == "__main__":
    main()
