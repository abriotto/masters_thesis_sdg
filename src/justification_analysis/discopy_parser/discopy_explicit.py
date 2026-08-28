"""Explicit-only shallow discourse parsing of justifications with discopy.

Runs in the dedicated `discopy-env` virtualenv, NOT in the project's `sdglogs`
environment: discopy needs TensorFlow and numpy<2, which are incompatible with
sdglogs' torch 2.11 / numpy 2.2. The notebook therefore never imports this
module; it consumes the JSONL artifact this module writes.

Parser: rknaebel/discopy 1.1.0 (CODI release), pretrained checkpoint
`bert-10.11.21-13.31`, embedding backbone `bert-base-cased`.

Only the explicit side of the parser is used. The pipeline is built from the
`ConnectiveSenseClassifier` component alone, so the implicit components
(`ImplicitArgumentExtractor`, `ArgumentSenseClassifier`) never run and no
implicit relation can enter the analysis. `ConnectiveArgumentExtractor` is
also excluded: argument spans are not needed for this research question and
the component is not required by the sense classifier.

Two deliberate deviations from the upstream code, both documented and both
verified to leave predictions unchanged:

1. Documents are built by `build_document()` from the project's own
   deterministic sentence segmentation (`src.utils.sentences.split_sentences`)
   instead of `discopy_data.data.loaders.raw.load_texts_fast`. That loader is
   buggy - it never resets its `words` list between sentences, so every
   sentence accumulates all preceding tokens - and using it would also replace
   the project's segmentation with NLTK punkt, breaking the sentence-level join
   with the DeepSeek annotations.
2. `parse_explicit()` batches the per-candidate Keras calls that
   `ConnectiveSenseClassifier.parse` issues one at a time, and keeps the
   softmax vector that upstream discards with `.argmax(-1)`. This is what makes
   a confidence column possible at all. `verify_against_upstream()` asserts the
   batched path reproduces upstream's relations exactly.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDTB sense collapsing
# ---------------------------------------------------------------------------

# The four PDTB top-level classes this thesis reports.
PDTB_TOP_LEVEL = ("Comparison", "Contingency", "Expansion", "Temporal")

# The 17 labels the pretrained checkpoint can emit, read from its own
# config.json / senses.json rather than assumed. Two of them are not PDTB
# top-level relation classes and must never be silently folded into one:
#   NoSense - the candidate is not a discourse connective here. The upstream
#             component drops these before a Relation is built, so they only
#             appear via parse_explicit(keep_nosense=True).
#   EntRel  - entity-based coherence, a PDTB relation type that sits outside
#             the four-class scheme.
NON_TOP_LEVEL_SENSES = ("NoSense", "EntRel")


def collapse_pdtb_sense(sense: str) -> Optional[str]:
    """Collapse a PDTB sense string to its top-level class.

    The PDTB sense hierarchy is dot-separated and strictly nested, so the
    top-level class is the prefix before the first dot -
    `Contingency.Cause.Reason` -> `Contingency`. Returns None for labels that
    are not one of the four top-level classes (`NoSense`, `EntRel`, and
    anything unrecognised), so callers must decide explicitly what to do with
    them rather than receive a guess.
    """
    if not sense:
        return None
    top = sense.split(".")[0].strip()
    return top if top in PDTB_TOP_LEVEL else None


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def build_document(doc_id: str, text: str, sentences: Sequence[str]):
    """Build a discopy Document from pre-segmented sentences.

    `sentences` must be verbatim substrings of `text`, in order - exactly what
    `src.utils.sentences.split_sentences` returns. Sentence boundaries are
    therefore the project's, identical to those the DeepSeek annotator sees.

    Every token carries its character offsets **into the original
    justification**, so the spans discopy returns are directly comparable with
    the DiMLex matcher's spans without any re-alignment step.
    """
    from nltk.tokenize import TreebankWordTokenizer
    from discopy_data.data.doc import Document

    tokenizer = TreebankWordTokenizer()

    sentence_payloads = []
    cursor = 0
    global_token_index = 0

    for sentence in sentences:
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        if start < 0:
            continue
        cursor = start + len(sentence)

        tokens = []
        for local_idx, (tok_start, tok_end) in enumerate(
            tokenizer.span_tokenize(sentence)
        ):
            tokens.append(
                {
                    "surface": sentence[tok_start:tok_end],
                    # Offsets are shifted into the full justification.
                    "characterOffsetBegin": start + tok_start,
                    "characterOffsetEnd": start + tok_end,
                    "upos": "",
                    "xpos": "",
                    "lemma": "",
                }
            )
            global_token_index += 1

        if tokens:
            sentence_payloads.append({"tokens": tokens})

    return Document.from_json(
        {
            "docID": doc_id,
            "meta": {},
            "text": text,
            "sentences": sentence_payloads,
        },
        load_dependencies=False,
        load_relations=False,
    )


# ---------------------------------------------------------------------------
# Explicit-only parsing
# ---------------------------------------------------------------------------

def load_explicit_component(model_path: str):
    """Load ONLY the explicit connective sense classifier from a checkpoint."""
    from discopy.parsers.pipeline import ParserPipeline

    pipeline = ParserPipeline.from_config(model_path)
    pipeline.load(model_path)

    explicit = [
        component
        for component in pipeline.components
        if type(component).__name__ == "ConnectiveSenseClassifier"
    ]
    if len(explicit) != 1:
        raise RuntimeError(
            "expected exactly one ConnectiveSenseClassifier in "
            f"{model_path}, found {[type(c).__name__ for c in pipeline.components]}"
        )
    return explicit[0]


def parse_explicit(component, doc, keep_nosense: bool = False) -> List[Dict]:
    """Explicit connective candidates of `doc` with senses and confidences.

    Mirrors `ConnectiveSenseClassifier.parse` exactly - same candidate
    enumeration, same features, same argmax - but scores all candidates of a
    document in one Keras call and retains the softmax distribution.

    With `keep_nosense=True` the candidates the parser rejects are returned as
    well, marked `is_connective=False`. Upstream discards them; they are the
    evidence for how discopy handles non-connective uses of ambiguous forms.
    """
    from discopy.components.connective.base import get_connective_candidates
    from discopy.components.sense.explicit.bert_conn_sense import get_bert_features

    if not doc.sentences:
        return []

    doc_bert = doc.get_embeddings()
    global_id_map = {
        (s_i, t.local_idx): t.idx
        for s_i, s in enumerate(doc.sentences)
        for t in s.tokens
    }

    features = []
    meta = []
    for sent_i, sent in enumerate(doc.sentences):
        for candidate in get_connective_candidates(sent):
            conn_idxs = tuple(global_id_map[(sent_i, i)] for i, _ in candidate)
            features.append(
                get_bert_features(conn_idxs, doc_bert, component.used_context)
            )
            meta.append((sent_i, sent, candidate))

    if not features:
        return []

    probabilities = component.model.predict(
        np.stack(features), verbose=0, batch_size=256
    )
    predictions = probabilities.argmax(-1)

    results = []
    for (sent_i, sent, candidate), pred, prob in zip(
        meta, predictions, probabilities
    ):
        pred = int(pred)
        raw_sense = component.classes[pred]
        is_connective = pred > 0
        if not is_connective and not keep_nosense:
            continue

        conn_tokens = [sent.tokens[i] for i, _ in candidate]
        char_spans = [
            (t.offset_begin, t.offset_end) for t in conn_tokens
        ]
        # Merge adjacent token spans the way TokenSpan.get_character_spans
        # does, so a discontinuous connective keeps one span per component.
        merged: List[List[int]] = []
        previous_idx = None
        for token in conn_tokens:
            if previous_idx is not None and token.idx == previous_idx + 1:
                merged[-1][1] = token.offset_end
            else:
                merged.append([token.offset_begin, token.offset_end])
            previous_idx = token.idx

        results.append(
            {
                "sentence_index": sent_i,
                "candidate_surface": " ".join(c for _, c in candidate),
                "connective_surface": " ".join(t.surface for t in conn_tokens),
                "token_indices": [t.idx for t in conn_tokens],
                "char_spans": [tuple(span) for span in merged],
                "n_char_spans": len(merged),
                "is_discontinuous": len(merged) > 1,
                "is_connective": is_connective,
                "raw_sense": raw_sense,
                "top_level": collapse_pdtb_sense(raw_sense),
                "confidence": float(prob[pred]),
                "type": "Explicit",
            }
        )
    return results


def verify_against_upstream(component, doc) -> Tuple[int, int]:
    """Assert the batched path reproduces upstream `parse()` exactly.

    Returns (n_upstream, n_batched). Raises AssertionError on any divergence.
    """
    upstream = component.parse(doc, [])
    upstream_key = sorted(
        (tuple(t.idx for t in r.conn.tokens), tuple(r.senses), r.type)
        for r in upstream
    )
    batched = parse_explicit(component, doc, keep_nosense=False)
    batched_key = sorted(
        (tuple(r["token_indices"]), (r["raw_sense"],), r["type"])
        for r in batched
    )
    assert upstream_key == batched_key, (
        "batched explicit parse diverged from upstream ConnectiveSenseClassifier"
        f"\n  upstream: {upstream_key}\n  batched : {batched_key}"
    )
    assert all(r["type"] == "Explicit" for r in batched), "non-explicit relation produced"
    return len(upstream_key), len(batched_key)
