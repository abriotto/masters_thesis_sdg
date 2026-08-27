"""Shared alignment layer for the joint discourse x semantic analysis.

This module used to carry a full SENTENCE-LEVEL joint analysis. That analysis
was withdrawn on 2026-08-27: it defined association as same-sentence
co-occurrence, which misses connectives that relate content across a sentence
boundary and confounds models that differ in justification length. Its tables,
figures, notebook and qualitative sample have been deleted; the replacement is
`src/justification_analysis/joint_justification/`.

What remains here is the part that was never in question and is still used by
the replacement:

  * `load_layers`         - the frozen semantic frames and the ACCEPTED
                            discourse relations (`is_connective == True`);
  * `align_relations`     - attaches every accepted relation to exactly one
                            canonical sentence, keyed on
                            (model, game_id, run_label, sentence_id) and then
                            VERIFIED by byte-exact sentence-text comparison.
                            No fuzzy matching; failures are returned, not
                            silently resolved;
  * `build_joint_sentences` - one row per canonical sentence with both layers
                            as presence flags.

The sentence-level co-presence those flags express is still computed, but it
is now the LOCALIZATION DIAGNOSTIC of the justification-level analysis - a
reported property of a pairing, never its definition.

Note the two id columns `align_relations` produces: `justification_id` is the
discourse pipeline's own integer row index and `justification_id_canonical` is
the semantic string id. Anything aggregating to justifications must use the
canonical one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.justification_analysis.semantic import semantic_final as sem

# ---------------------------------------------------------------------------
# Vocabulary and design constants
# ---------------------------------------------------------------------------

CATEGORY_ORDER: List[str] = list(sem.CATEGORY_ORDER)          # 7 substantive
OTHER_CATEGORY = sem.OTHER_CATEGORY
ALL_CATEGORIES: List[str] = list(sem.ALL_CATEGORIES)

TOP_LEVEL_ORDER = ["Comparison", "Contingency", "Expansion", "Temporal"]
ANY_RELATION = "Any explicit"

# The nine senses the parser actually produced on this corpus, grouped by
# top-level class. Taken from the frozen discourse handoff, and asserted
# against the data rather than trusted.
SENSE_ORDER = [
    "Comparison.Contrast", "Comparison.Concession",
    "Contingency.Cause", "Contingency.Condition",
    "Expansion.Conjunction", "Expansion.Alternative", "Expansion.Restatement",
    "Temporal.Asynchronous", "Temporal.Synchrony",
]

MODEL_ORDER = list(sem.MODEL_ORDER)
DECODING_ORDER = list(sem.DECODING_ORDER)
SENTENCE_KEY = ["model", "game_id", "run_label", "sentence_id"]

DISCOURSE_SUBPATH = Path(
    "analysis/cross_model/base/voting/prompt_v4/justification_analysis"
    "/discourse_parser/discopy_explicit_candidates.csv"
)






def _order(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model" in out.columns:
        out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    if "decoding_group" in out.columns:
        out["decoding_group"] = pd.Categorical(
            out["decoding_group"], DECODING_ORDER, ordered=True)
    if "semantic_category" in out.columns:
        known = [c for c in ALL_CATEGORIES
                 if c in set(out["semantic_category"].astype(str))]
        out["semantic_category"] = pd.Categorical(
            out["semantic_category"], known, ordered=True)
    if "discourse_relation" in out.columns:
        vocabulary = [*TOP_LEVEL_ORDER, *SENSE_ORDER, ANY_RELATION]
        known = [r for r in vocabulary
                 if r in set(out["discourse_relation"].astype(str))]
        out["discourse_relation"] = pd.Categorical(
            out["discourse_relation"], known, ordered=True)
    sort_cols = [c for c in ("model", "decoding_group", "run_label",
                             "semantic_category", "discourse_relation")
                 if c in out.columns]
    return (out.sort_values(sort_cols).reset_index(drop=True)
            if sort_cols else out)


# ---------------------------------------------------------------------------
# Input and alignment
# ---------------------------------------------------------------------------

def load_layers(repo_root: Path) -> Dict[str, pd.DataFrame]:
    """The semantic frames and the ACCEPTED discourse relations.

    Only `is_connective == True` rows are returned as relations: the rejected
    NoSense candidates stay in the source table so the accept/reject behaviour
    is inspectable, but they are not relations and never enter this analysis.
    """
    repo_root = Path(repo_root)
    semantic = sem.load_annotations(repo_root)

    candidates = pd.read_csv(repo_root / DISCOURSE_SUBPATH)
    relations = candidates.loc[candidates["is_connective"]].copy()
    relations["model"] = relations["model"].astype(str)

    return {
        "semantic": semantic,
        "candidates": candidates,
        "relations": relations,
    }


def align_relations(layers: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Attach every accepted relation to exactly one canonical sentence.

    The join is on (model, game_id, run_label, sentence_id) - identifiers both
    pipelines already carry - and is then VERIFIED by comparing the sentence
    text the parser scored against the canonical sentence text, byte for byte.
    No fuzzy matching, no normalisation, no fallback: a relation that fails
    either step is returned in `unaligned` for explicit reporting.
    """
    semantic = layers["semantic"]
    relations = layers["relations"]

    sentences = semantic["sentences"].merge(
        semantic["justifications"][
            ["justification_id", "model", "game_id", "run_label",
             "decoding_group"]],
        on="justification_id", how="left",
    )
    sentences["model"] = sentences["model"].astype(str)
    sentences["decoding_group"] = sentences["decoding_group"].astype(str)

    merged = relations.merge(
        sentences[SENTENCE_KEY + ["justification_id", "text"]],
        on=SENTENCE_KEY, how="left", indicator=True,
        suffixes=("", "_canonical"),
    )

    no_sentence = merged.loc[merged["_merge"] != "both"].copy()
    no_sentence["alignment_failure"] = "no canonical sentence for this key"

    matched = merged.loc[merged["_merge"] == "both"].copy()
    text_differs = matched.loc[
        matched["sentence_text"].astype(str) != matched["text"].astype(str)
    ].copy()
    text_differs["alignment_failure"] = "sentence text does not match canonical"

    aligned = matched.loc[
        matched["sentence_text"].astype(str) == matched["text"].astype(str)
    ].drop(columns=["_merge"]).copy()

    unaligned = pd.concat([no_sentence, text_differs], ignore_index=True)
    if len(unaligned):
        unaligned = unaligned[[
            "occurrence_id", "model", "game_id", "run_label", "sentence_id",
            "connective_surface", "raw_sense", "alignment_failure",
        ]]

    return {"sentences": sentences, "aligned": aligned, "unaligned": unaligned}


def build_joint_sentences(layers: Dict[str, pd.DataFrame],
                          alignment: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per canonical sentence, with both layers as presence flags.

    Multi-label structure is preserved on both sides: a sentence may carry
    several semantic categories and several discourse relations, and each is
    an independent boolean. Sentences with no relation and/or no semantic
    label are kept - dropping them would silently change every denominator.
    """
    semantic = layers["semantic"]
    sentences = alignment["sentences"]
    aligned = alignment["aligned"]

    joint = sentences[
        ["justification_id", "sentence_id", "model", "game_id", "run_label",
         "decoding_group", "text", "n_labels"]
    ].copy()

    # --- semantic presence -------------------------------------------------
    labels = semantic["labels"]
    for category in ALL_CATEGORIES:
        present = set(
            map(tuple,
                labels.loc[labels["category"].astype(str).eq(category),
                           ["justification_id", "sentence_id"]].to_numpy())
        )
        joint[f"sem_{category}"] = [
            (j, s) in present
            for j, s in zip(joint["justification_id"], joint["sentence_id"])
        ]

    # --- discourse presence ------------------------------------------------
    keys = list(map(tuple, joint[SENTENCE_KEY].to_numpy()))

    for relation in TOP_LEVEL_ORDER:
        present = set(map(tuple, aligned.loc[
            aligned["top_level"].astype(str).eq(relation), SENTENCE_KEY
        ].to_numpy()))
        joint[f"disc_{relation}"] = [key in present for key in keys]

    for sense in SENSE_ORDER:
        present = set(map(tuple, aligned.loc[
            aligned["raw_sense"].astype(str).eq(sense), SENTENCE_KEY
        ].to_numpy()))
        joint[f"sense_{sense}"] = [key in present for key in keys]

    counts = (aligned.groupby(SENTENCE_KEY, observed=True).size()
              .rename("n_relations"))
    joint = joint.merge(counts, on=SENTENCE_KEY, how="left")
    joint["n_relations"] = joint["n_relations"].fillna(0).astype(int)
    joint["has_any_relation"] = joint["n_relations"] > 0

    joint["n_semantic_categories"] = joint[
        [f"sem_{c}" for c in CATEGORY_ORDER]].sum(axis=1)

    return _order(joint)
