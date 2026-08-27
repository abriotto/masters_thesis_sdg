"""Provisional inspection matrices for the justification-level joint analysis.

Deliberately light. These exist so the four quantities can be read side by
side; no decision has been made about whether any of them belongs in the
thesis, and none is styled as if it had.

Cells whose denominator is small are hatched. That is a diagnostic aid, not a
filter - nothing is dropped, and the low-support threshold is a reading aid
rather than a rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.justification_analysis.joint_justification.justification_joint import (
    ANY_RELATION,
    CATEGORY_ORDER,
    LOW_SUPPORT_DIAGNOSTIC,
    MODEL_ORDER,
    TOP_LEVEL_ORDER,
    matrix,
)

MODEL_LABELS = {"Gemma 4 2B": "E2B", "Gemma 4 4B": "E4B", "Gemma 4 31B": "31B"}
CATEGORY_LABELS = [c.replace("SocialJudgment", "SocialJudg.")
                   .replace("ClaimComparison", "ClaimComp.")
                   for c in CATEGORY_ORDER]
RELATION_LABELS = ["Comp.", "Cont.", "Exp.", "Temp."]

WIDTH = 6.5
CELL_FONTSIZE = 6.5
TICK_FONTSIZE = 6.5
DPI = 300


def _decoding_note(decoding: str) -> str:
    return ("greedy decoding, one run per game" if decoding == "Greedy"
            else "mean across the three stochastic runs")


def _text_color(image, value) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    return "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"


def _panels(summary: pd.DataFrame, decoding: str, value: str, support_flag: str,
            cmap, vmin, vmax, fmt: str, colorbar_label: str, suptitle: str,
            path: Path, scale: float = 1.0, footnote: str = "") -> Path:
    frame = summary.loc[
        summary["discourse_relation"].astype(str).ne(ANY_RELATION)]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.9), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03)

    image = None
    for column, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        values = scale * matrix(frame, model, decoding, value).to_numpy(float)
        thin = matrix(frame, model, decoding, support_flag).to_numpy(dtype=bool)
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(MODEL_LABELS[model], fontsize=8.5, pad=3)

        for a in range(len(CATEGORY_ORDER)):
            for b in range(len(TOP_LEVEL_ORDER)):
                cell = values[a, b]
                if np.isfinite(cell):
                    ax.text(b, a, fmt.format(cell), ha="center", va="center",
                            fontsize=CELL_FONTSIZE,
                            color=_text_color(image, cell))
                if thin[a, b]:
                    ax.add_patch(plt.Rectangle(
                        (b - 0.5, a - 0.5), 1, 1, fill=False, linewidth=0.0,
                        edgecolor="black", hatch="////", alpha=0.55))

        ax.set_xticks(range(len(TOP_LEVEL_ORDER)))
        ax.set_yticks(range(len(CATEGORY_ORDER)))
        ax.set_xticklabels(RELATION_LABELS, fontsize=TICK_FONTSIZE)
        ax.set_yticklabels(CATEGORY_LABELS if column == 0
                           else [""] * len(CATEGORY_ORDER),
                           fontsize=TICK_FONTSIZE)
        ax.tick_params(length=2, pad=1.5)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.012, aspect=26)
    colorbar.set_label(colorbar_label, fontsize=TICK_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=2)

    fig.suptitle(suptitle, fontsize=10)
    note = "\n" + _decoding_note(decoding)
    if footnote:
        note += "\n" + footnote
    fig.supxlabel(note, fontsize=6.5, color="#444444")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


HATCH_NOTE = (f"hatched: fewer than {LOW_SUPPORT_DIAGNOSTIC} justifications "
              "per run carry the category (diagnostic, not a filter)")


def figure_conditional_prevalence(summary, path, decoding="Stochastic"):
    return _panels(
        summary, decoding, "mean_conditional_prevalence",
        "low_support_diagnostic", "viridis", 0, 100, "{:.0f}",
        "P(relation | category)  %",
        "Justification-level P(relation | semantic category)",
        path, scale=100.0, footnote=HATCH_NOTE)


def figure_conditional_density(summary, path, decoding="Stochastic"):
    return _panels(
        summary, decoding, "mean_relations_per_100_words",
        "low_support_diagnostic", "viridis", 0, 1.6, "{:.2f}",
        "relations per 100 words",
        "Discourse density within justifications carrying each category",
        path, footnote=HATCH_NOTE)


def figure_lift(summary, path, decoding="Stochastic"):
    return _panels(
        summary, decoding, "mean_lift", "low_support_diagnostic",
        "RdBu_r", 0.0, 2.0, "{:.2f}", "lift",
        "Justification-level lift: discourse class x semantic category",
        path,
        footnote="1.00 = independence; colour clipped to [0, 2], printed "
                 f"values exact.\n{HATCH_NOTE}")


def figure_localization(summary, path, decoding="Stochastic"):
    return _panels(
        summary, decoding, "mean_localization_rate", "low_support_diagnostic",
        "magma", 0, 100, "{:.0f}", "localization rate  %",
        "Localization: paired justifications with a same-sentence co-occurrence",
        path, scale=100.0,
        footnote="descriptive diagnostic only - a low rate does not make the "
                 "justification-level pairing spurious")


def build_final_figures(tables: Dict[str, pd.DataFrame],
                        directory: Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    localization = tables["K8b_localization_summary"].copy()
    # The localization table has no support flag of its own; borrow the
    # category-level one so the same cells are marked in every figure.
    flags = tables["K2b_conditional_prevalence_summary"][
        ["model", "decoding_group", "semantic_category", "discourse_relation",
         "low_support_diagnostic"]]
    localization = localization.merge(
        flags, on=["model", "decoding_group", "semantic_category",
                   "discourse_relation"], how="left")
    localization["low_support_diagnostic"] = (
        localization["low_support_diagnostic"].fillna(False))

    paths = []
    for decoding in ("Stochastic", "Greedy"):
        suffix = "" if decoding == "Stochastic" else "_greedy"
        paths += [
            figure_conditional_prevalence(
                tables["K2b_conditional_prevalence_summary"],
                directory / f"G1_conditional_prevalence{suffix}.png", decoding),
            figure_conditional_density(
                tables["K3b_conditional_density_summary"],
                directory / f"G2_conditional_density{suffix}.png", decoding),
            figure_lift(
                tables["K4b_joint_prevalence_lift_summary"],
                directory / f"G3_lift{suffix}.png", decoding),
            figure_localization(
                localization,
                directory / f"G4_localization{suffix}.png", decoding),
        ]
    return paths
