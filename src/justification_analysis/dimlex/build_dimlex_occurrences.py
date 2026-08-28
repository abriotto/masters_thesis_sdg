"""Build the DiMLex occurrence table for a stage, with a manifest.

The DiMLex occurrence table is corpus-dependent, so it gets the same freshness
treatment as the parser artifact: a manifest recording the corpus fingerprint,
and a gate that refuses it against a different corpus. Without this a
fine-tuned run could silently reuse base occurrences in the coverage
diagnostic.

Cheap to rebuild (~5 s), unlike the parser, but the guarantee matters more
than the cost.

    python -m src.justification_analysis.dimlex.build_dimlex_occurrences \
        --stage base
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.justification_analysis.dimlex import dimlex_lexicon as dl  # noqa: E402
from src.justification_analysis.pipeline import corpus as corpus_module  # noqa: E402
from src.justification_analysis.pipeline import manifest as manifest_module  # noqa: E402
from src.justification_analysis.pipeline.config import AnalysisConfig  # noqa: E402


def build(config: AnalysisConfig):
    corpus = corpus_module.load_corpus(config)
    assignments = dl.load_marker_assignments()
    contiguous, discontinuous = dl.build_matching_lexicons(assignments)
    metrics, occurrences = dl.match_corpus(corpus, contiguous, discontinuous)
    occurrences = dl.attach_sentence_ids(occurrences, corpus)
    return corpus, assignments, metrics, occurrences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="base")
    parser.add_argument("--prompt-version", default="prompt_v4")
    args = parser.parse_args()

    config = AnalysisConfig(stage=args.stage,
                            prompt_version=args.prompt_version,
                            repo_root=REPO_ROOT)
    corpus, assignments, metrics, occurrences = build(config)

    out = config.dimlex_occurrences_path
    out.parent.mkdir(parents=True, exist_ok=True)
    # `component_spans` holds tuples; drop it from the CSV rather than writing
    # a repr that cannot be read back.
    export = occurrences.drop(columns=["component_spans"], errors="ignore")
    export.to_csv(out, index=False, encoding="utf-8-sig")

    manifest = manifest_module.build_manifest(
        config, corpus,
        artifact_path=out,
        artifact_kind="dimlex_occurrences",
        producer={
            "implementation": "src.justification_analysis.dimlex.dimlex_lexicon",
            "lexicon_entries": int(len(assignments)),
            "contiguous_entries": int(assignments["use_for_contiguous_matching"].sum()),
            "discontinuous_entries": int(assignments["use_for_discontinuous_matching"].sum()),
            "max_component_gap_tokens": dl.MAX_COMPONENT_GAP_TOKENS,
            "gap_threshold_note": (
                "FROZEN methodological choice, selected during pipeline "
                "development against the BASE corpus. Never recalibrated per "
                "stage."),
            "released_dimlex_entries": dl.RELEASED_DIMLEX_ENTRY_COUNT,
            "paper_reported_dimlex_entries": dl.PAPER_REPORTED_DIMLEX_ENTRIES,
            "command": " ".join(sys.argv),
        },
        outputs={
            "n_occurrences": int(len(occurrences)),
            "n_contiguous": int((occurrences["match_type"] == "contiguous").sum()),
            "n_discontinuous": int((occurrences["match_type"] == "discontinuous").sum()),
            "n_unplaced": int(occurrences["sentence_id"].isna().sum()),
        },
    )
    manifest_module.write_manifest(manifest, config.dimlex_manifest_path)

    print(f"stage       : {config.stage}")
    print(f"corpus hash : {manifest['corpus']['fingerprint']}")
    print(f"occurrences : {len(occurrences)}")
    print(f"  contiguous    : {int((occurrences['match_type'] == 'contiguous').sum())}")
    print(f"  discontinuous : {int((occurrences['match_type'] == 'discontinuous').sum())}")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"      {config.dimlex_manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
