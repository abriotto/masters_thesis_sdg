"""Figures for the joint discourse x semantic analysis.

Pure matplotlib over the frames `joint_final` returns, so every figure is drawn
from exactly the numbers in the corresponding table - nothing is recomputed
here and nothing is smoothed, clipped or reordered for appearance.

Conventions shared with the discourse and semantic figures: models are
E2B / E4B / 31B in that order, 300 dpi, authored at the width they will be
printed at so the point sizes are the ones the reader gets, greedy always in
its own figure and never described as a mean.

One convention specific to this layer: cells whose denominator N(c) is thin
are HATCHED rather than dropped. Several model-category combinations have only
a handful of sentences - E2B Mechanical averages six per run - and a
conditional prevalence over six sentences should be visible as unreliable, not
deleted and not silently plotted as if it were solid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.justification_analysis.joint.joint_final import (
    ANY_RELATION,
    CATEGORY_ORDER,
    MODEL_ORDER,
    THIN_SUPPORT,
    TOP_LEVEL_ORDER,
    prevalence_matrix,
)

MODEL_LABELS = {"Gemma 4 2B": "E2B", "Gemma 4 4B": "E4B", "Gemma 4 31B": "31B"}

SINGLE_COLUMN_WIDTH = 6.5
CELL_FONTSIZE = 6.5
TICK_FONTSIZE = 6.5
PANEL_TITLE_FONTSIZE = 8.5
SUPTITLE_FONTSIZE = 10.0
SUBTITLE_FONTSIZE = 7.0
DPI = 300

CATEGORY_LABELS = [c.replace("SocialJudgment", "SocialJudg.")
                   .replace("ClaimComparison", "ClaimComp.")
                   for c in CATEGORY_ORDER]
RELATION_LABELS = ["Comp.", "Cont.", "Exp.", "Temp."]


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _decoding_note(decoding: str) -> str:
    if decoding == "Greedy":
        return "greedy decoding, one run per game"
    return "mean across the three stochastic runs"


def _cell_text_color(image, value) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.5 else "black"


def _thin_mask(summary: pd.DataFrame, model: str, decoding: str) -> np.ndarray:
    """Which (category, relation) cells rest on a thin denominator."""
    return prevalence_matrix(
        summary, model, decoding, "support_is_thin"
    ).to_numpy(dtype=bool)


def _panels(summary: pd.DataFrame, decoding: str, value: str,
            cmap, vmin, vmax, fmt: str, colorbar_label: str,
            suptitle: str, path: Path, footnote: str = "",
            scale: float = 1.0) -> Path:
    frame = summary.loc[
        summary["discourse_relation"].astype(str).ne(ANY_RELATION)]

    fig, axes = plt.subplots(1, 3, figsize=(SINGLE_COLUMN_WIDTH, 2.9),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03)

    image = None
    for column, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        matrix = scale * prevalence_matrix(
            frame, model, decoding, value).to_numpy(float)
        thin = _thin_mask(frame, model, decoding)
        # aspect="auto": four columns against seven rows would otherwise force
        # a very tall panel and waste the page.
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                          aspect="auto")
        ax.set_title(MODEL_LABELS[model], fontsize=PANEL_TITLE_FONTSIZE, pad=3)

        for a in range(len(CATEGORY_ORDER)):
            for b in range(len(TOP_LEVEL_ORDER)):
                value_ab = matrix[a, b]
                if not np.isfinite(value_ab):
                    continue
                ax.text(b, a, fmt.format(value_ab), ha="center", va="center",
                        fontsize=CELL_FONTSIZE,
                        color=_cell_text_color(image, value_ab))
                if thin[a, b]:
                    ax.add_patch(plt.Rectangle(
                        (b - 0.5, a - 0.5), 1, 1, fill=False,
                        edgecolor="black", linewidth=0.0,
                        hatch="////", alpha=0.55,
                    ))

        ax.set_xticks(range(len(TOP_LEVEL_ORDER)))
        ax.set_yticks(range(len(CATEGORY_ORDER)))
        ax.set_xticklabels(RELATION_LABELS, fontsize=TICK_FONTSIZE)
        ax.set_yticklabels(CATEGORY_LABELS if column == 0
                           else [""] * len(CATEGORY_ORDER),
                           fontsize=TICK_FONTSIZE)
        ax.tick_params(length=2, pad=1.5)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.012,
                            aspect=26)
    colorbar.set_label(colorbar_label, fontsize=TICK_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=2)

    fig.suptitle(suptitle, fontsize=SUPTITLE_FONTSIZE)
    note = "\n" + _decoding_note(decoding)
    if footnote:
        note += "\n" + footnote
    fig.supxlabel(note, fontsize=SUBTITLE_FONTSIZE, color="#444444")
    return _save(fig, path)


def figure_conditional_prevalence(summary: pd.DataFrame, path: Path,
                                  decoding: str = "Stochastic") -> Path:
    """F1: P(relation | semantic category), rows categories, columns classes.

    Reads: among sentences containing this semantic category, the percentage
    that also contain at least one explicit relation of this class. It is a
    sentence-level association, not evidence that the connective attaches to
    the semantic content.
    """
    return _panels(
        summary, decoding, "mean_conditional_prevalence",
        cmap="viridis", vmin=0, vmax=60, fmt="{:.0f}",
        colorbar_label="P(relation | category)  %",
        suptitle="Explicit discourse relations in sentences by semantic category",
        path=path,
        scale=100.0,
        footnote=(f"hatched: fewer than {THIN_SUPPORT} sentences per run "
                  "carry the category"),
    )


def figure_lift(summary: pd.DataFrame, path: Path,
                decoding: str = "Stochastic") -> Path:
    """F2: sentence-level lift, centred on 1.

    Above 1 the pair shares a sentence more often than sentence-level
    independence implies, below 1 less often. Descriptive only - lift is not
    bootstrapped, and on a thin denominator it swings on a couple of sentences.
    """
    # Colour scale FIXED at [0, 2], symmetric about 1, rather than stretched
    # to the data. A single thin-support cell reaches 6.56, and letting it set
    # the range flattens every other cell to white and drags the low end to a
    # negative lift, which cannot exist. Cells beyond the range saturate; the
    # printed value is always the exact one.
    return _panels(
        summary, decoding, "mean_lift",
        cmap="RdBu_r", vmin=0.0, vmax=2.0, fmt="{:.2f}",
        colorbar_label="lift",
        suptitle="Sentence-level lift: discourse class x semantic category",
        path=path,
        footnote=(f"1.00 = sentence-level independence; colour clipped to "
                  f"[0, 2] with printed values exact;\nhatched: fewer than "
                  f"{THIN_SUPPORT} sentences per run carry the category"),
    )


def build_final_figures(tables: Dict[str, pd.DataFrame],
                        directory: Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prevalence = tables["J2_conditional_prevalence_summary"]
    lift = tables["J3b_joint_prevalence_lift_summary"]
    return [
        figure_conditional_prevalence(
            prevalence, directory / "F1_joint_conditional_prevalence.png"),
        figure_lift(lift, directory / "F2_joint_lift.png"),
        figure_conditional_prevalence(
            prevalence,
            directory / "F1b_joint_conditional_prevalence_greedy.png",
            decoding="Greedy"),
        figure_lift(lift, directory / "F2b_joint_lift_greedy.png",
                    decoding="Greedy"),
    ]
