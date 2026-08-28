"""EXPERIMENT: force `given` / `given that` spans through discopy's sense classifier.

    "C:/Users/annab/discopy-env/Scripts/python.exe" \
        src/justification_analysis/discopy_parser/forced_span_probe.py \
        --model-path <checkpoint dir> --out <csv>

Narrow question: when the exact span is handed to the existing
`ConnectiveSenseClassifier`, bypassing ONLY candidate enumeration, does it
return a sensible Contingency sense or `NoSense`?

WHAT IS AND IS NOT CHANGED
--------------------------
* The classifier is untouched: same loaded weights, same `get_bert_features`,
  same `used_context`, same argmax. The ONLY difference from the production
  path is that `get_connective_candidates()` is not consulted - the connective
  token indices come from the DiMLex occurrence table instead.
* Documents are built exactly as in the corpus run (`build_document`), and the
  FULL document is embedded, not just the target sentence, so the +/-1 token
  context window sees the same neighbours it would in a real run. The features
  are therefore identical to what the parser would compute if the span had
  been enumerated.
* Nothing is written into the standard discopy outputs. This writes one
  separate experimental CSV.
* This is a probe, not a hybrid. It activates nothing.

Acceptance here is NOT evidence that the hybrid works. A non-`NoSense` label
only means the classifier returned something; whether the span is a genuine
PDTB-style Explicit connective and whether the sense is right are manual
questions, answered in `5_forced_given_validation.ipynb`.
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
    build_document, collapse_pdtb_sense, load_explicit_component,
)

TARGET_FORMS = ("given", "given that")

ARTIFACTS = (
    REPO_ROOT / "analysis" / "cross_model" / "base" / "voting"
    / "prompt_v4" / "justification_analysis" / "discourse_parser"
)


def parse_spans(text: str):
    out = []
    for chunk in str(text).split(";"):
        begin, _, end = chunk.partition("-")
        try:
            out.append((int(begin), int(end)))
        except ValueError:
            pass
    return out


def load_targets(all_forms: bool = False, reuse: str = "") -> pd.DataFrame:
    """Spans to force. `all_forms` takes every NOT_A_CANDIDATE occurrence.

    `reuse` points at an earlier predictions CSV; occurrences already scored
    there are dropped, so a previous run is never recomputed. Identity is
    (justification_id, char_spans, marker), which is unique per occurrence.
    """
    gap = pd.read_csv(ARTIFACTS / "coverage_gap_triage.csv", encoding="utf-8-sig")
    targets = (gap.copy() if all_forms
               else gap.loc[gap["marker"].isin(TARGET_FORMS)].copy())

    if reuse and Path(reuse).exists():
        previous = pd.read_csv(reuse, encoding="utf-8-sig")
        done = set(zip(previous["justification_id"], previous["char_spans"],
                       previous["marker"]))
        before = len(targets)
        keys = list(zip(targets["justification_id"], targets["char_spans"],
                        targets["marker"]))
        targets = targets.loc[[k not in done for k in keys]].copy()
        print(f"reusing {before - len(targets)} predictions from "
              f"{Path(reuse).name}", flush=True)
    assert (targets["alignment_status"]
            .str.startswith("NOT_A_CANDIDATE")).all(), \
        "a target span was already a discopy candidate; probe would be circular"
    return targets


def load_justifications() -> pd.DataFrame:
    from src.justification_analysis.comparison.discourse_statistics import (
        load_justification_frame,
    )
    return load_justification_frame(REPO_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--bert-model", default="bert-base-cased")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all-forms", action="store_true",
                        help="force every NOT_A_CANDIDATE occurrence, not just "
                             "given / given that")
    parser.add_argument("--reuse", default="",
                        help="earlier predictions CSV whose occurrences are skipped")
    args = parser.parse_args()

    from discopy.components.sense.explicit.bert_conn_sense import get_bert_features
    from discopy_data.nn.bert import get_sentence_embedder

    targets = load_targets(all_forms=args.all_forms, reuse=args.reuse)
    votes = load_justifications()
    text_by_id = dict(zip(votes["justification_id"], votes["justification"]))
    meta_by_id = votes.set_index("justification_id")[
        ["model", "game_id", "run_label", "decoding_group"]
    ].to_dict("index")

    justification_ids = sorted(targets["justification_id"].unique())
    if args.limit:
        justification_ids = justification_ids[: args.limit]
    print(f"forced spans      : {len(targets):,}", flush=True)
    print(f"justifications    : {len(justification_ids):,}", flush=True)

    embedder = get_sentence_embedder(args.bert_model)
    component = load_explicit_component(args.model_path)
    print(f"component         : {type(component).__name__} "
          f"(used_context={component.used_context}, "
          f"{len(component.classes)} classes)", flush=True)

    by_justification = dict(tuple(targets.groupby("justification_id")))

    rows = []
    unmapped = 0
    start = time.time()

    for position, justification_id in enumerate(justification_ids):
        text = text_by_id[justification_id]
        sentences = split_sentences(text)
        if not sentences:
            continue

        doc = build_document(f"just_{justification_id}", text, sentences)
        if not doc.sentences:
            continue
        for sent_i, sentence in enumerate(doc.sentences):
            doc.sentences[sent_i].embeddings = embedder(sentence.tokens)

        doc_bert = doc.get_embeddings()
        tokens = doc.get_tokens()

        features, meta = [], []
        for occurrence in by_justification[justification_id].itertuples(index=False):
            spans = parse_spans(occurrence.char_spans)
            if not spans:
                unmapped += 1
                continue
            span_begin, span_end = spans[0][0], spans[-1][1]

            # Exact span -> token indices. This replaces candidate enumeration
            # and nothing else.
            selected = [
                token for token in tokens
                if token.offset_begin >= span_begin and token.offset_end <= span_end
            ]
            if not selected:
                unmapped += 1
                continue

            conn_idxs = tuple(token.idx for token in selected)
            features.append(
                get_bert_features(conn_idxs, doc_bert, component.used_context)
            )
            meta.append((occurrence, selected))

        if not features:
            continue

        probabilities = component.model.predict(
            np.stack(features), verbose=0, batch_size=256
        )
        predictions = probabilities.argmax(-1)

        for (occurrence, selected), prediction, probability in zip(
            meta, predictions, probabilities
        ):
            prediction = int(prediction)
            raw_sense = component.classes[prediction]
            info = meta_by_id[occurrence.justification_id]
            rows.append({
                "justification_id": occurrence.justification_id,
                "model": info["model"],
                "game_id": info["game_id"],
                "run_label": info["run_label"],
                "decoding_group": info["decoding_group"],
                "sentence_id": occurrence.sentence_id,
                "sentence_text": occurrence.sentence_text,
                "marker": occurrence.marker,
                "form": occurrence.marker,
                "char_spans": occurrence.char_spans,
                "start": selected[0].offset_begin,
                "end": selected[-1].offset_end,
                "surface": " ".join(t.surface for t in selected),
                "n_tokens": len(selected),
                "triage": occurrence.triage,
                "dimlex_category": occurrence.category,
                "predicted_sense": raw_sense,
                "predicted_top_level": collapse_pdtb_sense(raw_sense),
                "is_nosense": raw_sense == "NoSense",
                "accepted": raw_sense != "NoSense",
                "confidence": float(probability[prediction]),
                "p_nosense": float(probability[0]),
            })

        if (position + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (position + 1) / elapsed
            print(f"  {position + 1}/{len(justification_ids)} "
                  f"({rate:.2f}/s, eta "
                  f"{(len(justification_ids) - position - 1) / rate / 60:.1f} min)",
                  flush=True)

    frame = pd.DataFrame(rows)
    frame.insert(0, "probe_id", np.arange(len(frame)))

    assert frame["probe_id"].is_unique
    if not args.all_forms:
        assert set(frame["form"]) <= set(TARGET_FORMS)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8-sig")

    runtime = (time.time() - start) / 60
    print(f"\nforced spans scored : {len(frame):,}", flush=True)
    print(f"unmappable spans    : {unmapped}", flush=True)
    print(f"accepted            : {int(frame['accepted'].sum()):,} "
          f"({100 * frame['accepted'].mean():.1f}%)", flush=True)
    print(f"NoSense             : {int(frame['is_nosense'].sum()):,}", flush=True)
    print(f"runtime             : {runtime:.1f} min", flush=True)
    print(f"\nwrote -> {out}", flush=True)
    print("\npredicted sense distribution:", flush=True)
    print(frame["predicted_sense"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
