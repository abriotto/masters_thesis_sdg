"""Aggregate the forced-span probe output.

Descriptive only. Acceptance by the classifier is NOT evidence that a hybrid
works: a non-`NoSense` label only means the classifier returned something. The
questions of whether the span is a genuine PDTB-style Explicit connective and
whether the sense is right are manual, and are answered in
`5_forced_given_validation.ipynb`.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

MODEL_ORDER = ["Gemma 4 2B", "Gemma 4 4B", "Gemma 4 31B"]
CONFIDENCE_BINS = [-0.01, 0.5, 0.75, 0.9, 1.01]
CONFIDENCE_LABELS = ["<0.50", "0.50-0.75", "0.75-0.90", ">=0.90"]


def add_confidence_band(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["confidence_band"] = pd.cut(
        out["confidence"], CONFIDENCE_BINS, labels=CONFIDENCE_LABELS
    )
    return out


def acceptance_by(frame: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, observed=True).agg(
        n_forced=("accepted", "size"),
        n_accepted=("accepted", "sum"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    grouped["n_nosense"] = grouped["n_forced"] - grouped["n_accepted"]
    grouped["acceptance_rate_pct"] = (
        100 * grouped["n_accepted"] / grouped["n_forced"]
    ).round(1)
    return grouped[keys + ["n_forced", "n_accepted", "n_nosense",
                           "acceptance_rate_pct", "mean_confidence"]]


def sense_distribution(frame: pd.DataFrame, form: str = None) -> pd.DataFrame:
    subset = frame if form is None else frame.loc[frame["form"].eq(form)]
    table = (
        subset["predicted_sense"].value_counts()
        .rename_axis("predicted_sense").reset_index(name="n")
    )
    table["pct"] = (100 * table["n"] / len(subset)).round(1)
    table["top_level"] = table["predicted_sense"].map(
        lambda s: s.split(".")[0] if s not in ("NoSense", "EntRel") else s
    )
    return table


def accepted_top_level_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Among ACCEPTED spans only: where does the classifier put them?

    The question that matters is whether accepted cases land on Contingency -
    the category the manual inspection assigned to every one of these forms -
    rather than being scattered.
    """
    accepted = frame.loc[frame["accepted"]]
    rows = []
    for form, group in accepted.groupby("form", observed=True):
        counts = group["predicted_top_level"].value_counts()
        row = {"form": form, "n_accepted": len(group)}
        for category in ("Comparison", "Contingency", "Expansion", "Temporal"):
            row[category] = int(counts.get(category, 0))
        row["pct_Contingency"] = (
            round(100 * counts.get("Contingency", 0) / len(group), 1)
            if len(group) else np.nan
        )
        rows.append(row)
    if len(accepted):
        counts = accepted["predicted_top_level"].value_counts()
        row = {"form": "ALL", "n_accepted": len(accepted)}
        for category in ("Comparison", "Contingency", "Expansion", "Temporal"):
            row[category] = int(counts.get(category, 0))
        row["pct_Contingency"] = round(
            100 * counts.get("Contingency", 0) / len(accepted), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_summary(frame: pd.DataFrame) -> pd.DataFrame:
    banded = add_confidence_band(frame)
    return (
        banded.groupby(["accepted", "confidence_band"], observed=True)
        .size().rename("n").reset_index()
        .pivot(index="confidence_band", columns="accepted", values="n")
        .fillna(0).astype(int)
        .rename(columns={True: "accepted", False: "NoSense"})
    )
