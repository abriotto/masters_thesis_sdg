"""Descriptive statistics for the discopy explicit-relation analysis.

Pure pandas, runs in `sdglogs`. Everything is computed from the CSV artifacts
written by `run_discopy_on_justifications.py` and notebook 1's occurrence
export, so nothing here re-runs parser inference.

Two conventions are shared with the DiMLex notebook and must not drift:

  * the word denominator is `WORD_PATTERN`, the notebook's token pattern, NOT
    whitespace splitting. Both analyses therefore normalise per 100 words on
    the same scale.
  * statistics are computed per run first, then summarised across the three
    stochastic runs as mean and SD. The greedy run is a single run and is kept
    separate throughout; its SD is undefined and reported as NaN.

Stochastic generations from the same game are not independent games, so any
game-level aggregation averages within a game before averaging across games.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PDTB_TOP_LEVEL = ("Comparison", "Contingency", "Expansion", "Temporal")
CATEGORY_ORDER = ["Contingency", "Comparison", "Expansion", "Temporal"]
MODEL_ORDER = ["Gemma 4 2B", "Gemma 4 4B", "Gemma 4 31B"]
DECODING_ORDER = ["Stochastic", "Greedy"]

MODEL_FOLDER_PATTERNS = {
    "Gemma 4 2B": "*gemma-4-E2B*",
    "Gemma 4 4B": "*gemma-4-E4B*",
    "Gemma 4 31B": "*gemma-4-31B*",
}
CSV_RELATIVE_PATH = Path(
    "base/voting/prompt_v4/vote_stability/tables/llm_vote_file_level.csv"
)

# Identical to the DiMLex notebook's WORD_PATTERN.
WORD_PATTERN = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", flags=re.UNICODE)

RUN_KEYS = ["model", "decoding_group", "run_label"]


def load_justification_frame(repo_root: Path = None, config=None) -> pd.DataFrame:
    """One row per justification, for the ACTIVE stage.

    This used to be one of two independent implementations of the same load -
    the other lived in the parser runner, with a comment asking a maintainer
    to keep them identical by hand. Both now delegate to
    `pipeline.corpus.load_corpus`, so there is exactly one definition of what
    the corpus is and the parser can never disagree with the analysis about it.

    Kept as a named function because several notebooks and diagnostic modules
    call it; the signature accepts either a repo root (legacy, base stage) or
    an explicit config.
    """
    from src.justification_analysis.pipeline import config as pipeline_config
    from src.justification_analysis.pipeline import corpus as corpus_module

    if config is None:
        config = pipeline_config.AnalysisConfig(
            repo_root=Path(repo_root) if repo_root
            else pipeline_config.find_repo_root())
    return corpus_module.load_corpus(config)


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model" in out.columns:
        out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    if "decoding_group" in out.columns:
        out["decoding_group"] = pd.Categorical(
            out["decoding_group"], DECODING_ORDER, ordered=True
        )
    sort_cols = [c for c in ("model", "decoding_group") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


def _order_index(frame):
    """Same ordering as `_order`, for frames that carry it on the index."""
    names = list(frame.index.names)
    if "model" not in names:
        return frame
    out = frame.reset_index()
    return _order(out).set_index(names)


def compute_discopy_statistics(
    accepted_occurrences: pd.DataFrame,
    justifications: pd.DataFrame,
    category_column: str = "top_level",
) -> Dict[str, pd.DataFrame]:
    """Run-level and model-level descriptives for one occurrence table.

    Works unchanged for the DiMLex table (`category_column="category"`), which
    is what makes the sensitivity comparison a like-for-like contrast rather
    than two differently-computed numbers.
    """
    base = justifications[
        ["justification_id", "model", "game_id", "run_label", "run_number",
         "decoding_group", "n_words"]
    ].copy()

    counts = accepted_occurrences.groupby("justification_id").size().rename("n_markers")
    per_just = base.merge(counts, left_on="justification_id", right_index=True, how="left")
    per_just["n_markers"] = per_just["n_markers"].fillna(0).astype(int)
    per_just["has_marker"] = (per_just["n_markers"] > 0).astype(int)

    # --- overall, per run then across runs --------------------------------
    run_level = per_just.groupby(RUN_KEYS, as_index=False).agg(
        n_justifications=("justification_id", "nunique"),
        total_words=("n_words", "sum"),
        total_occurrences=("n_markers", "sum"),
        n_justifications_with_marker=("has_marker", "sum"),
    )
    run_level["occurrences_per_justification"] = (
        run_level["total_occurrences"] / run_level["n_justifications"]
    )
    run_level["occurrences_per_100_words"] = (
        100 * run_level["total_occurrences"] / run_level["total_words"]
    )
    run_level["pct_justifications_with_marker"] = (
        100 * run_level["n_justifications_with_marker"] / run_level["n_justifications"]
    )

    overall = run_level.groupby(["model", "decoding_group"], as_index=False).agg(
        n_runs=("run_label", "nunique"),
        mean_occurrences_per_justification=("occurrences_per_justification", "mean"),
        sd_occurrences_per_justification=("occurrences_per_justification", "std"),
        mean_occurrences_per_100_words=("occurrences_per_100_words", "mean"),
        sd_occurrences_per_100_words=("occurrences_per_100_words", "std"),
        mean_pct_justifications_with_marker=("pct_justifications_with_marker", "mean"),
        sd_pct_justifications_with_marker=("pct_justifications_with_marker", "std"),
    )

    # --- per category -----------------------------------------------------
    category_run_rows = []
    for category in PDTB_TOP_LEVEL:
        subset = accepted_occurrences.loc[
            accepted_occurrences[category_column] == category
        ]
        cat_counts = subset.groupby("justification_id").size().rename("n")
        frame = base.merge(cat_counts, left_on="justification_id",
                           right_index=True, how="left")
        frame["n"] = frame["n"].fillna(0).astype(int)
        frame["has"] = (frame["n"] > 0).astype(int)
        grouped = frame.groupby(RUN_KEYS, as_index=False).agg(
            n_justifications=("justification_id", "nunique"),
            total_words=("n_words", "sum"),
            total_occurrences=("n", "sum"),
            n_justifications_with_category=("has", "sum"),
        )
        grouped["category"] = category
        grouped["occurrences_per_100_words"] = (
            100 * grouped["total_occurrences"] / grouped["total_words"]
        )
        grouped["occurrences_per_justification"] = (
            grouped["total_occurrences"] / grouped["n_justifications"]
        )
        grouped["pct_justifications_with_category"] = (
            100 * grouped["n_justifications_with_category"]
            / grouped["n_justifications"]
        )
        category_run_rows.append(grouped)

    category_run_level = pd.concat(category_run_rows, ignore_index=True)
    category_summary = category_run_level.groupby(
        ["model", "decoding_group", "category"], as_index=False
    ).agg(
        n_runs=("run_label", "nunique"),
        mean_per_100_words=("occurrences_per_100_words", "mean"),
        sd_per_100_words=("occurrences_per_100_words", "std"),
        mean_per_justification=("occurrences_per_justification", "mean"),
        sd_per_justification=("occurrences_per_justification", "std"),
        mean_pct_justifications=("pct_justifications_with_category", "mean"),
        sd_pct_justifications=("pct_justifications_with_category", "std"),
    )

    # --- category proportions --------------------------------------------
    labelled = accepted_occurrences.merge(
        justifications[["justification_id", "model", "decoding_group"]],
        on="justification_id", how="left", suffixes=("", "_j"),
    )
    proportions = (
        labelled.groupby(["model", "decoding_group", category_column])
        .size().unstack(fill_value=0)
        .reindex(columns=list(PDTB_TOP_LEVEL), fill_value=0)
    )
    proportions = 100 * proportions.div(proportions.sum(axis=1), axis=0)

    return {
        "per_justification": per_just,
        "run_level": run_level,
        "overall_summary": _order(overall),
        "category_run_level": category_run_level,
        "category_summary": _order(category_summary),
        "category_proportions": proportions.round(2),
    }


def category_cooccurrence(
    accepted_occurrences: pd.DataFrame,
    justifications: pd.DataFrame,
    category_column: str = "top_level",
) -> Dict[str, pd.DataFrame]:
    """Directional conditional co-occurrence, per run then averaged.

    Cell (row, col) = among justifications containing the row category, the
    percentage that also contain the column category. Diagonal left empty.
    Mirrors the DiMLex notebook's construction.
    """
    presence = justifications[
        ["justification_id", "model", "decoding_group", "run_label"]
    ].copy()
    for category in PDTB_TOP_LEVEL:
        ids = set(
            accepted_occurrences.loc[
                accepted_occurrences[category_column] == category,
                "justification_id",
            ]
        )
        presence[category] = presence["justification_id"].isin(ids).astype(int)

    results = {}
    for (model, decoding), group in presence.groupby(["model", "decoding_group"]):
        run_matrices = []
        for _, run_group in group.groupby("run_label"):
            binary = run_group[list(PDTB_TOP_LEVEL)].astype(int)
            pair = binary.T.dot(binary).astype(float)
            row_totals = np.diag(pair.to_numpy()).astype(float)
            values = np.full(pair.shape, np.nan)
            for i, total in enumerate(row_totals):
                if total > 0:
                    values[i, :] = 100 * pair.iloc[i, :].to_numpy() / total
            np.fill_diagonal(values, np.nan)
            run_matrices.append(values)

        stack = np.stack(run_matrices)
        mean = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1) if len(run_matrices) > 1 else np.full_like(mean, np.nan)
        np.fill_diagonal(mean, np.nan)
        np.fill_diagonal(sd, np.nan)
        results[(model, decoding)] = {
            "mean": pd.DataFrame(mean, index=list(PDTB_TOP_LEVEL), columns=list(PDTB_TOP_LEVEL)),
            "sd": pd.DataFrame(sd, index=list(PDTB_TOP_LEVEL), columns=list(PDTB_TOP_LEVEL)),
            "n_runs": len(run_matrices),
        }
    return results


CONDITIONAL_FORMS = ("if", "then", "if then")


def conditional_marker_diagnostic(
    discopy_aligned: pd.DataFrame,
    dimlex_aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Every corpus occurrence of `if` / `then` / `if ... then`, for inspection.

    Built because these markers showed possible Contingency-vs-Temporal
    confusion. Nothing is relabelled here - the table exists so the cases can
    be read by hand.
    """
    from src.justification_analysis.comparison.discourse_comparison import normalise_surface

    discopy = discopy_aligned.copy()
    discopy["form"] = discopy["candidate_surface"].map(normalise_surface)
    subset = discopy.loc[discopy["form"].isin(CONDITIONAL_FORMS)].copy()

    dimlex = dimlex_aligned.copy()
    dimlex["form"] = dimlex["marker"].map(normalise_surface)
    dimlex_category = (
        dimlex.loc[dimlex["discopy_occurrence_id"].notna(),
                   ["discopy_occurrence_id", "category"]]
        .drop_duplicates("discopy_occurrence_id")
        .set_index("discopy_occurrence_id")["category"]
    )
    subset["dimlex_category"] = subset["occurrence_id"].map(dimlex_category)

    subset["status"] = np.where(subset["is_connective"], "accepted", "rejected_nosense")
    columns = [
        "occurrence_id", "model", "run_label", "decoding_group",
        "justification_id", "sentence_id", "form", "connective_surface",
        "status", "raw_sense", "top_level", "confidence", "dimlex_category",
        "is_discontinuous", "sentence_text",
    ]
    return subset[columns].sort_values(
        ["form", "status", "raw_sense", "confidence"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Secondary analysis: fine-grained PDTB senses
# ---------------------------------------------------------------------------

def fine_grained_sense_statistics(
    accepted_occurrences: pd.DataFrame,
    justifications: pd.DataFrame,
    sense_column: str = "raw_sense",
) -> Dict[str, pd.DataFrame]:
    """Level-2 PDTB senses per run, then mean and SD across runs.

    Secondary to the four top-level classes, and computed on the same run-first
    convention so the numbers sit on the same scale as the primary tables.

    Senses absent from a run are zero-filled before averaging: without that, a
    sense observed in a single run would report that run's rate as its mean.
    `Expansion.Restatement` has n=1 in the whole corpus, so several of its cells
    are singletons - it is kept rather than dropped, because dropping it would
    silently change the share denominator.
    """
    run_totals = justifications.groupby(RUN_KEYS, as_index=False).agg(
        n_justifications=("justification_id", "nunique"),
        total_words=("n_words", "sum"),
    )

    labelled = accepted_occurrences[["justification_id", sense_column]].merge(
        justifications[["justification_id"] + RUN_KEYS],
        on="justification_id", how="left",
    )
    assert labelled[RUN_KEYS].notna().all().all(), \
        "an accepted occurrence has no matching justification"

    counts = (
        labelled.groupby(RUN_KEYS + [sense_column])
        .size().rename("n_occurrences").reset_index()
    )
    observed = sorted(counts[sense_column].unique())
    grid = run_totals.merge(pd.DataFrame({sense_column: observed}), how="cross")

    run_level = grid.merge(counts, on=RUN_KEYS + [sense_column], how="left")
    run_level["n_occurrences"] = run_level["n_occurrences"].fillna(0).astype(int)
    run_level["per_100_words"] = (
        100 * run_level["n_occurrences"] / run_level["total_words"]
    )
    run_level["per_justification"] = (
        run_level["n_occurrences"] / run_level["n_justifications"]
    )
    run_level["top_level"] = run_level[sense_column].str.split(".").str[0]

    summary = run_level.groupby(
        ["model", "decoding_group", "top_level", sense_column], as_index=False
    ).agg(
        n_runs=("run_label", "nunique"),
        total_occurrences=("n_occurrences", "sum"),
        mean_per_100_words=("per_100_words", "mean"),
        sd_per_100_words=("per_100_words", "std"),
        mean_per_justification=("per_justification", "mean"),
        sd_per_justification=("per_justification", "std"),
    )

    counts_wide = (
        run_level.pivot_table(
            index=["model", "decoding_group"], columns=sense_column,
            values="n_occurrences", aggfunc="sum", observed=False,
        ).reindex(columns=observed, fill_value=0).astype(int)
    )
    shares_wide = (100 * counts_wide.div(counts_wide.sum(axis=1), axis=0)).round(2)

    corpus_totals = (
        run_level.groupby([sense_column, "top_level"], as_index=False)["n_occurrences"]
        .sum().rename(columns={"n_occurrences": "n"})
        .sort_values("n", ascending=False).reset_index(drop=True)
    )
    corpus_totals["pct_of_relations"] = (
        100 * corpus_totals["n"] / corpus_totals["n"].sum()
    ).round(2)

    return {
        "run_level": run_level,
        "summary": _order(summary),
        "counts_wide": _order_index(counts_wide),
        "shares_wide": _order_index(shares_wide),
        "corpus_totals": corpus_totals,
    }


# ---------------------------------------------------------------------------
# Secondary analysis: accepted connective surface forms
# ---------------------------------------------------------------------------

def connective_form_statistics(
    candidates: pd.DataFrame,
    justifications: pd.DataFrame,
    top_n: int = 15,
) -> Dict[str, pd.DataFrame]:
    """Which connective forms the models actually use, and how they are read.

    Takes the FULL candidate table, not just the accepted rows, so the
    acceptance rate per form is available - that is the quantity that shows the
    contextual NoSense filter doing work, and it cannot be recovered from the
    accepted rows alone.
    """
    from src.justification_analysis.comparison.discourse_comparison import (
        normalise_surface,
    )

    frame = candidates[
        ["justification_id", "candidate_surface", "is_connective", "raw_sense",
         "top_level"]
    ].copy()
    frame["form"] = frame["candidate_surface"].map(normalise_surface)
    frame = frame.merge(
        justifications[["justification_id"] + RUN_KEYS],
        on="justification_id", how="left",
    )

    accepted = frame.loc[frame["is_connective"]]
    order = (
        accepted.groupby("form").size().sort_values(ascending=False).index.tolist()
    )
    top_forms = order[:top_n]

    # --- acceptance rate per form ----------------------------------------
    acceptance = frame.groupby("form", as_index=False).agg(
        n_candidates=("is_connective", "size"),
        n_accepted=("is_connective", "sum"),
    )
    acceptance["n_accepted"] = acceptance["n_accepted"].astype(int)
    acceptance["n_rejected_nosense"] = (
        acceptance["n_candidates"] - acceptance["n_accepted"]
    )
    acceptance["pct_accepted"] = (
        100 * acceptance["n_accepted"] / acceptance["n_candidates"]
    ).round(2)
    acceptance = acceptance.sort_values("n_accepted", ascending=False).reset_index(
        drop=True
    )

    # --- per-form rate by model, run first then across runs ---------------
    run_totals = justifications.groupby(RUN_KEYS, as_index=False).agg(
        total_words=("n_words", "sum"),
    )
    counts = (
        accepted.loc[accepted["form"].isin(top_forms)]
        .groupby(RUN_KEYS + ["form"]).size().rename("n").reset_index()
    )
    grid = run_totals.merge(pd.DataFrame({"form": top_forms}), how="cross")
    run_level = grid.merge(counts, on=RUN_KEYS + ["form"], how="left")
    run_level["n"] = run_level["n"].fillna(0).astype(int)
    run_level["per_100_words"] = 100 * run_level["n"] / run_level["total_words"]

    by_model = run_level.groupby(
        ["model", "decoding_group", "form"], as_index=False
    ).agg(
        n_runs=("run_label", "nunique"),
        total_occurrences=("n", "sum"),
        mean_per_100_words=("per_100_words", "mean"),
        sd_per_100_words=("per_100_words", "std"),
    )
    rates_wide = by_model.pivot_table(
        index=["model", "decoding_group"], columns="form",
        values="mean_per_100_words", observed=False,
    ).reindex(columns=top_forms)

    # --- how one form is read across contexts -----------------------------
    profile = (
        accepted.loc[accepted["form"].isin(top_forms)]
        .groupby(["form", "top_level"]).size().unstack(fill_value=0)
        .reindex(index=top_forms, columns=list(PDTB_TOP_LEVEL), fill_value=0)
    )
    profile_pct = (100 * profile.div(profile.sum(axis=1), axis=0)).round(1)
    profile_pct.insert(0, "n_accepted", profile.sum(axis=1))

    return {
        "top_forms": top_forms,
        "acceptance_by_form": acceptance,
        "run_level": run_level,
        "by_model": _order(by_model),
        "rates_wide": _order_index(rates_wide),
        "sense_profile": profile_pct,
    }


def to_latex(frame: pd.DataFrame, path: Path, caption: str = "", float_format="%.3f"):
    """Write a LaTeX table next to the CSV. Formatting only, no decisions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            frame.to_latex(index=True, float_format=float_format,
                           caption=caption or None, escape=True)
        )
