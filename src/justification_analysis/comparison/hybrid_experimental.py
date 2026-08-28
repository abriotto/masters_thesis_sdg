"""EXPERIMENTAL hybrid relation table: standard discopy ∪ accepted DiMLex-only spans.

NOT the production pipeline. Nothing here overwrites the standard discopy
outputs; the result is written to a separate experimental directory and stays
there until the hybrid is approved.

Architecture under test:

    candidate set = discopy's own candidates
                    ∪ DiMLex candidates discopy never enumerated
    then the SAME unchanged ConnectiveSenseClassifier decides, for every
    candidate, whether it is a discourse connective and what sense it carries.

The division of labour is deliberate: DiMLex only broadens candidate coverage,
discopy alone makes every contextual decision. No per-form rule, whitelist or
blacklist is applied anywhere in this module - if a lexical form contributes
nothing after contextual filtering, that is a result, not something to patch.

Provenance is recorded on every relation (`standard_discopy` /
`dimlex_expanded`) so the two sources can always be separated again.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.justification_analysis.comparison.discourse_statistics import (
    DECODING_ORDER, MODEL_ORDER, PDTB_TOP_LEVEL, RUN_KEYS,
)

STANDARD = "standard_discopy"
EXPANDED = "dimlex_expanded"


def build_hybrid_table(
    accepted_standard: pd.DataFrame,
    forced: pd.DataFrame,
) -> pd.DataFrame:
    """Union of standard accepted relations and accepted forced spans.

    Standard predictions are copied through untouched. A forced span is added
    only if it is accepted AND its (justification_id, char_spans) is not
    already present in the standard table - so no span can be double counted
    even if the alignment bookkeeping were imperfect.
    """
    standard = accepted_standard.copy()
    standard["provenance"] = STANDARD
    standard["source_form"] = standard["candidate_surface"]

    keep = [
        "justification_id", "model", "game_id", "run_label", "decoding_group",
        "sentence_id", "sentence_text", "char_spans", "start", "end",
        "raw_sense", "top_level", "confidence", "provenance", "source_form",
    ]
    standard = standard[[c for c in keep if c in standard.columns]]

    new = forced.loc[forced["accepted"]].copy()
    new["provenance"] = EXPANDED
    new = new.rename(columns={
        "predicted_sense": "raw_sense",
        "predicted_top_level": "top_level",
        "marker": "source_form",
    })
    new = new[[c for c in keep if c in new.columns]]

    existing_spans = set(zip(standard["justification_id"], standard["char_spans"]))
    before = len(new)
    new = new.loc[[
        (j, s) not in existing_spans
        for j, s in zip(new["justification_id"], new["char_spans"])
    ]]
    n_duplicate = before - len(new)

    hybrid = pd.concat([standard, new], ignore_index=True)
    hybrid.insert(0, "relation_id", np.arange(len(hybrid)))

    # --- invariants -------------------------------------------------------
    assert hybrid["relation_id"].is_unique
    assert not hybrid.duplicated(["justification_id", "char_spans"]).any(), \
        "duplicate span in the hybrid table"
    assert hybrid["top_level"].isin(PDTB_TOP_LEVEL).all(), \
        "a relation carries a label outside the four top-level classes"
    assert set(hybrid["provenance"]) <= {STANDARD, EXPANDED}
    # Standard rows must be untouched.
    assert int((hybrid["provenance"] == STANDARD).sum()) == len(accepted_standard), \
        "the hybrid dropped or duplicated a standard discopy relation"

    hybrid.attrs["n_duplicate_spans_skipped"] = n_duplicate
    return hybrid


def relation_rates(
    relations: pd.DataFrame,
    justifications: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Run-level then across-run rates, matching the main discopy tables."""
    base = justifications[
        ["justification_id", "model", "decoding_group", "run_label", "n_words"]
    ]
    counts = relations.groupby("justification_id").size().rename("n")
    frame = base.merge(counts, left_on="justification_id",
                       right_index=True, how="left")
    frame["n"] = frame["n"].fillna(0).astype(int)
    frame["has"] = (frame["n"] > 0).astype(int)

    run = frame.groupby(RUN_KEYS, as_index=False).agg(
        n_just=("justification_id", "nunique"),
        total_words=("n_words", "sum"),
        total=("n", "sum"),
        n_with=("has", "sum"),
    )
    run["per_100w"] = 100 * run["total"] / run["total_words"]
    run["pct_just"] = 100 * run["n_with"] / run["n_just"]

    summary = run.groupby(["model", "decoding_group"], as_index=False).agg(
        total_relations=("total", "mean"),
        per_100_words=("per_100w", "mean"),
        sd_per_100_words=("per_100w", "std"),
        pct_justifications=("pct_just", "mean"),
    )
    summary["variant"] = label
    return summary


def compare_standard_hybrid(
    accepted_standard: pd.DataFrame,
    hybrid: pd.DataFrame,
    justifications: pd.DataFrame,
) -> pd.DataFrame:
    standard_rates = relation_rates(accepted_standard, justifications, "standard")
    hybrid_rates = relation_rates(hybrid, justifications, "hybrid")
    both = pd.concat([standard_rates, hybrid_rates], ignore_index=True)
    both["model"] = pd.Categorical(both["model"], MODEL_ORDER, ordered=True)
    both["decoding_group"] = pd.Categorical(both["decoding_group"],
                                            DECODING_ORDER, ordered=True)
    return both.sort_values(["model", "decoding_group", "variant"]).reset_index(drop=True)


def class_profiles(hybrid: pd.DataFrame, category_column: str = "top_level") -> pd.DataFrame:
    """Four-class proportions by model x decoding x provenance-inclusive variant."""
    rows = []
    for variant, subset in (
        ("standard", hybrid.loc[hybrid["provenance"].eq(STANDARD)]),
        ("hybrid", hybrid),
    ):
        for (model, decoding), group in subset.groupby(
            ["model", "decoding_group"], observed=True
        ):
            counts = group[category_column].value_counts()
            total = int(counts.sum())
            row = {"model": model, "decoding_group": decoding,
                   "variant": variant, "n_relations": total}
            for category in PDTB_TOP_LEVEL:
                row[f"pct_{category}"] = (
                    round(100 * int(counts.get(category, 0)) / total, 2)
                    if total else np.nan
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["model"] = pd.Categorical(frame["model"], MODEL_ORDER, ordered=True)
    frame["decoding_group"] = pd.Categorical(frame["decoding_group"],
                                             DECODING_ORDER, ordered=True)
    return frame.sort_values(["model", "decoding_group", "variant"]).reset_index(drop=True)


def gains_by_model(hybrid: pd.DataFrame) -> pd.DataFrame:
    """How many relations each model gains, and into which classes."""
    added = hybrid.loc[hybrid["provenance"].eq(EXPANDED)]
    table = (
        added.groupby(["model", "decoding_group", "top_level"], observed=True)
        .size().rename("n_added").reset_index()
        .pivot_table(index=["model", "decoding_group"], columns="top_level",
                     values="n_added", fill_value=0, observed=True)
    )
    table["total_added"] = table.sum(axis=1)
    return table


def gains_by_form(hybrid: pd.DataFrame) -> pd.DataFrame:
    """Which lexical forms actually survive contextual filtering."""
    added = hybrid.loc[hybrid["provenance"].eq(EXPANDED)]
    return (
        added.groupby("source_form", observed=True)
        .agg(n_added=("relation_id", "size"),
             mean_confidence=("confidence", "mean"))
        .join(
            added.groupby(["source_form", "top_level"], observed=True)
            .size().unstack(fill_value=0)
        )
        .sort_values("n_added", ascending=False)
    )


def sense_shift(hybrid: pd.DataFrame) -> pd.DataFrame:
    """Fine-grained sense counts, standard vs hybrid."""
    standard = (hybrid.loc[hybrid["provenance"].eq(STANDARD), "raw_sense"]
                .value_counts().rename("standard"))
    full = hybrid["raw_sense"].value_counts().rename("hybrid")
    table = pd.concat([standard, full], axis=1).fillna(0).astype(int)
    table["added"] = table["hybrid"] - table["standard"]
    return table.sort_values("hybrid", ascending=False)
