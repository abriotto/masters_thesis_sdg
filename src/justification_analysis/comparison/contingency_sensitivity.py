"""Sensitivity of the cross-model Contingency comparison to the `given` /
`given that` candidate-inventory gap.

SENSITIVITY DIAGNOSTIC ONLY. Nothing here is a corrected parser output and
nothing here replaces the discopy results. No span is forced into the parser,
the candidate inventory is untouched, and no inference is re-run.

Motivation. The manual coverage inspection found that the out-of-inventory
occurrences discopy never enumerates are, in this corpus, essentially two
forms: `given` (10/10 judged valid PDTB-style Explicit connectives) and
`given that` (6/6). Every one was labelled Contingency. `despite` (1/7),
`eventually` (0/4) and `with` (0/3) did not survive the same criterion. The
question this module answers is whether adding those occurrences back would
change the cross-model picture.

Two augmented quantities are built, both clearly labelled:

  extended_contingency_upper_bound
      discopy Contingency + EVERY lexical `given` and `given that` occurrence.
      Deliberately generous: it assumes every lexical hit is a connective,
      which the manual inspection shows is false for `given` (verbal uses like
      "would have given" exist). It is the ceiling, not an estimate.

  extended_contingency_plausible
      discopy Contingency + only those `given` / `given that` occurrences whose
      syntactic context matches the manually validated pattern, using the
      deterministic rule below. No LLM, no manual annotation.

The deterministic plausibility rule (from coverage_gap_analysis.classify_occurrence):

  * `given that`  - accepted whenever clause-initial or followed by a comma.
                    `that` forces a clausal complement, so the construction is
                    a subordinator by definition. 54/54 accepted.
  * `given`       - accepted only when BOTH clause-initial (sentence start, or
                    directly after , ; : or a coordinator) AND followed later
                    in the sentence by a comma, i.e. the "Given X, Y" frame.
                    Rejected outright when preceded by an auxiliary
                    ("was/has/have/had ... given"), which is verbal. 460/477
                    accepted.

Both counts keep the caveat that a lexical occurrence is not automatically a
discourse connective; the plausible variant only narrows the error, it does not
remove it.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.justification_analysis.comparison.discourse_statistics import (
    PDTB_TOP_LEVEL, RUN_KEYS, MODEL_ORDER, DECODING_ORDER,
)

GAP_FORMS = ("given", "given that")


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model" in out.columns:
        out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    if "decoding_group" in out.columns:
        out["decoding_group"] = pd.Categorical(out["decoding_group"],
                                               DECODING_ORDER, ordered=True)
    cols = [c for c in ("model", "decoding_group") if c in out.columns]
    return out.sort_values(cols).reset_index(drop=True) if cols else out


def per_form_rates(gap: pd.DataFrame, justifications: pd.DataFrame,
                   forms: Sequence[str] = GAP_FORMS) -> pd.DataFrame:
    """Occurrences, per-100-word rate and justification presence, per form.

    Run-level first, then mean across runs - the same aggregation the main
    discopy tables use, so the numbers are directly comparable.
    """
    base = justifications[
        ["justification_id", "model", "decoding_group", "run_label", "n_words"]
    ]
    rows = []
    for form in list(forms) + ["__combined__"]:
        subset = (gap if form == "__combined__"
                  else gap.loc[gap["marker"].eq(form)])
        if form == "__combined__":
            subset = gap.loc[gap["marker"].isin(forms)]

        counts = subset.groupby("justification_id").size().rename("n")
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
            total_occurrences=("total", "mean"),
            per_100_words=("per_100w", "mean"),
            sd_per_100_words=("per_100w", "std"),
            pct_justifications=("pct_just", "mean"),
        )
        summary["form"] = "given + given that" if form == "__combined__" else form
        rows.append(summary)

    return _order(pd.concat(rows, ignore_index=True))[
        ["model", "decoding_group", "form", "total_occurrences",
         "per_100_words", "sd_per_100_words", "pct_justifications"]
    ]


def contingency_sensitivity(
    accepted: pd.DataFrame,
    gap: pd.DataFrame,
    justifications: pd.DataFrame,
) -> pd.DataFrame:
    """Original / plausible / upper-bound Contingency, per model x decoding."""
    base = justifications[
        ["justification_id", "model", "decoding_group", "run_label", "n_words"]
    ]

    variants: Dict[str, pd.DataFrame] = {}
    variants["original"] = accepted.loc[accepted["top_level"].eq("Contingency")]

    gap_forms = gap.loc[gap["marker"].isin(GAP_FORMS)]
    variants["upper_bound"] = pd.concat(
        [variants["original"][["justification_id"]],
         gap_forms[["justification_id"]]], ignore_index=True)
    variants["plausible"] = pd.concat(
        [variants["original"][["justification_id"]],
         gap_forms.loc[gap_forms["triage"].eq("CLAUSE_INITIAL_PLAUSIBLE"),
                       ["justification_id"]]], ignore_index=True)

    rows = []
    for name, occ in variants.items():
        counts = occ.groupby("justification_id").size().rename("n")
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
            total_occurrences=("total", "mean"),
            per_100_words=("per_100w", "mean"),
            sd_per_100_words=("per_100w", "std"),
            pct_justifications=("pct_just", "mean"),
        )
        summary["variant"] = name
        rows.append(summary)

    return _order(pd.concat(rows, ignore_index=True))[
        ["model", "decoding_group", "variant", "total_occurrences",
         "per_100_words", "sd_per_100_words", "pct_justifications"]
    ]


def four_class_profiles(
    accepted: pd.DataFrame,
    gap: pd.DataFrame,
    justifications: pd.DataFrame,
) -> pd.DataFrame:
    """Top-level class proportions under each Contingency variant.

    Temporal / Comparison / Expansion are held fixed; only Contingency is
    augmented, so the proportions shift purely because the denominator grows.
    """
    # Both tables already carry model / decoding_group; only attach them if a
    # table is missing them, so no suffixed duplicate columns are created.
    def _with_ids(frame):
        if {"model", "decoding_group"} <= set(frame.columns):
            return frame
        return frame.merge(
            justifications[["justification_id", "model", "decoding_group"]],
            on="justification_id", how="left",
        )

    labelled = _with_ids(accepted)
    gap_forms = _with_ids(gap.loc[gap["marker"].isin(GAP_FORMS)])

    rows = []
    for name in ("original", "plausible", "upper_bound"):
        if name == "original":
            extra = gap_forms.iloc[0:0]
        elif name == "plausible":
            extra = gap_forms.loc[gap_forms["triage"].eq("CLAUSE_INITIAL_PLAUSIBLE")]
        else:
            extra = gap_forms

        for (model, decoding), group in labelled.groupby(
            ["model", "decoding_group"], observed=True
        ):
            counts = {c: int(group["top_level"].eq(c).sum())
                      for c in PDTB_TOP_LEVEL}
            add = extra.loc[extra["model"].eq(model)
                            & extra["decoding_group"].eq(decoding)]
            counts["Contingency"] += len(add)
            total = sum(counts.values())
            row = {"model": model, "decoding_group": decoding, "variant": name,
                   "n_occurrences": total}
            for c in PDTB_TOP_LEVEL:
                row[f"pct_{c}"] = round(100 * counts[c] / total, 2) if total else np.nan
            rows.append(row)

    return _order(pd.DataFrame(rows))


def ordering_check(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Model ordering and relative gaps under each variant."""
    rows = []
    for (decoding, variant), group in sensitivity.groupby(
        ["decoding_group", "variant"], observed=True
    ):
        ranked = group.sort_values("per_100_words", ascending=False)
        order = " > ".join(str(m) for m in ranked["model"])
        top = float(ranked["per_100_words"].iat[0])
        bottom = float(ranked["per_100_words"].iat[-1])
        rows.append({
            "decoding_group": decoding,
            "variant": variant,
            "ordering_high_to_low": order,
            "highest_rate": round(top, 3),
            "lowest_rate": round(bottom, 3),
            "spread": round(top - bottom, 3),
            "ratio_high_low": round(top / bottom, 3) if bottom else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["decoding_group", "variant"])
