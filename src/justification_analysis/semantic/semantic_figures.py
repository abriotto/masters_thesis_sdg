"""Thesis figures for the semantic-annotation analysis.

Pure matplotlib over the frames `semantic_final` returns, so every figure is
drawn from exactly the numbers in the corresponding final table - nothing is
recomputed here and nothing is smoothed, clipped or reordered for appearance.

Conventions, shared with the discourse figures:

  * models are E2B / E4B / 31B, always in that order, viridis at 0.15/0.45/0.72;
  * stochastic bars are the mean across the three runs, error bars the SD;
  * greedy is a single run, so it never gets an error bar. It is overlaid as an
    open diamond rather than a fourth bar, to keep it visibly not-a-mean;
  * a reference line at zero on every difference/contrast panel, because the
    sign of a contrast is the thing being read.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.justification_analysis.semantic.semantic_final import (
    CATEGORY_ORDER,
    MODEL_ORDER,
    cooccurrence_matrix,
)

MODEL_LABELS = {"Gemma 4 2B": "E2B", "Gemma 4 4B": "E4B", "Gemma 4 31B": "31B"}
MODEL_COLORS = dict(zip(MODEL_ORDER, plt.cm.viridis([0.15, 0.45, 0.72])))
K_LABELS = {0: "0/3", 1: "1/3", 2: "2/3", 3: "3/3"}

DPI = 300


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _flat(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ("model", "decoding_group", "category",
                   "category_a", "category_b"):
        if column in out.columns:
            out[column] = out[column].astype(str)
    return out


def _short(category: str) -> str:
    """Category labels are long enough to collide on a 7-tick axis."""
    return {
        "Mechanical": "Mechanical",
        "Testimony": "Testimony",
        "SocialJudgment": "Social\nJudgment",
        "Behavioral": "Behavioral",
        "ClaimComparison": "Claim\nComparison",
        "Payoff": "Payoff",
        "Uncertainty": "Uncertainty",
    }.get(category, category)


# ---------------------------------------------------------------------------
# F1 - semantic profile
# ---------------------------------------------------------------------------

def figure_semantic_prevalence(model_prevalence: pd.DataFrame, path: Path) -> Path:
    """F1: justification-level prevalence of each category, by model.

    Stochastic bars are the mean of the three runs with SD error bars; greedy
    is the open diamond. The y axis is the share of justifications in which the
    category appears at least once - not a share of labels, and not a rate.
    """
    frame = _flat(model_prevalence)
    frame = frame.loc[frame["category"].isin(CATEGORY_ORDER)]

    x = np.arange(len(CATEGORY_ORDER))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for i, model in enumerate(MODEL_ORDER):
        offset = (i - 1) * width
        stochastic = frame.loc[
            frame["model"].eq(model) & frame["decoding_group"].eq("Stochastic")
        ].set_index("category").reindex(CATEGORY_ORDER)
        greedy = frame.loc[
            frame["model"].eq(model) & frame["decoding_group"].eq("Greedy")
        ].set_index("category").reindex(CATEGORY_ORDER)

        ax.bar(
            x + offset, 100 * stochastic["prevalence_mean"], width,
            yerr=100 * stochastic["prevalence_sd"], capsize=3,
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            error_kw={"elinewidth": 0.9},
        )
        ax.scatter(
            x + offset, 100 * greedy["prevalence_mean"],
            marker="D", s=22, facecolors="none", edgecolors="black",
            linewidths=0.9, zorder=3,
            label="greedy" if i == 0 else None,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_short(c) for c in CATEGORY_ORDER], fontsize=8)
    ax.set_ylabel("% of justifications\ninvoking the category")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# F2 - co-occurrence
# ---------------------------------------------------------------------------

# Sized for `\includegraphics[width=\linewidth]` in a single-column thesis.
# The figure is authored at the width it will be printed at, so the point
# sizes below are the point sizes the reader gets - no silent rescaling.
SINGLE_COLUMN_WIDTH = 6.5

MATRIX_LABELS = [c.replace("Judgment", "Judg.").replace("Comparison", "Comp.")
                 for c in CATEGORY_ORDER]

CELL_FONTSIZE = 6.0
TICK_FONTSIZE = 6.5
PANEL_TITLE_FONTSIZE = 8.5
SUPTITLE_FONTSIZE = 10.0
SUBTITLE_FONTSIZE = 7.5
UNDEFINED_COLOR = "#e4e4e4"


def _decoding_subtitle(decoding: str) -> str:
    """Greedy is ONE deterministic run per game and must never be described as
    a mean - the previous combined figure said 'mean of 3 runs' for both."""
    if decoding == "Greedy":
        return "greedy decoding, one run per game"
    return "mean across the three stochastic runs"


def _cell_text_color(image, value) -> str:
    """Black or white, whichever the cell's own colour can carry.

    Read off the colormap rather than guessed from a value threshold, so it
    stays correct for both the sequential and the diverging scale.
    """
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 0.5 else "black"


def _matrix_axis(ax, show_y_labels: bool) -> None:
    n = len(CATEGORY_ORDER)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(MATRIX_LABELS, rotation=45, ha="right",
                       fontsize=TICK_FONTSIZE)
    ax.set_yticklabels(MATRIX_LABELS if show_y_labels else [""] * n,
                       fontsize=TICK_FONTSIZE)
    ax.tick_params(length=2, pad=1.5)


def figure_cooccurrence_prevalence(joint: pd.DataFrame, path: Path,
                                   decoding: str = "Stochastic") -> Path:
    """Three joint-prevalence matrices, one per model, on one row.

    Joint prevalence is the share of justifications carrying both categories.
    The diagonal is a category with itself, which is its marginal prevalence -
    kept, and outlined, because it is the reference the off-diagonal cells are
    read against.
    """
    n = len(CATEGORY_ORDER)
    # constrained layout, not a hand-tuned wspace: the panels carry an equal
    # aspect, so their true height is only known after the labels are laid
    # out. Letting matplotlib solve it is what removes the dead space.
    fig, axes = plt.subplots(1, 3, figsize=(SINGLE_COLUMN_WIDTH, 2.6),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02)

    image = None
    for column, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        matrix = 100 * cooccurrence_matrix(
            joint, model, decoding, "joint_prevalence_mean"
        ).to_numpy(dtype=float)
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=100)
        ax.set_title(MODEL_LABELS[model], fontsize=PANEL_TITLE_FONTSIZE,
                     pad=3)

        for a in range(n):
            for b in range(n):
                ax.text(b, a, f"{matrix[a, b]:.0f}", ha="center", va="center",
                        fontsize=CELL_FONTSIZE,
                        color=_cell_text_color(image, matrix[a, b]))
            ax.add_patch(plt.Rectangle(
                (a - 0.5, a - 0.5), 1, 1, fill=False,
                edgecolor="white", linewidth=0.7,
            ))
        _matrix_axis(ax, show_y_labels=column == 0)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.012,
                            aspect=28)
    colorbar.set_label("joint prevalence (%)", fontsize=TICK_FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=2)

    fig.suptitle("Justification-level semantic co-occurrence",
                 fontsize=SUPTITLE_FONTSIZE)
    # The decoding note sits at the bottom rather than under the title: as a
    # second heading it collided with the title once the tight bounding box
    # was applied, and it reads as a source line anyway.
    fig.supxlabel("\n" + _decoding_subtitle(decoding),
                  fontsize=SUBTITLE_FONTSIZE, color="#444444")
    return _save(fig, path)


def figure_cooccurrence_lift(lift: pd.DataFrame, path: Path,
                             decoding: str = "Stochastic") -> Path:
    """Three lift matrices, one per model, on one row.

    Lift is joint prevalence divided by the product of the marginals, so the
    scale is centred on 1 and symmetric: red is more co-occurrence than the
    marginals imply, blue less, white about as expected. The diagonal is
    undefined and is drawn in grey so it cannot be misread as lift = 1, which
    is very nearly the same white.

    Secondary and diagnostic. A pair resting on a handful of justifications
    can carry a large lift, which is why the support counts stay in
    `S6b_cooccurrence_ranked_pairs.csv` and the footnote points at them.
    """
    n = len(CATEGORY_ORDER)
    matrices = [
        cooccurrence_matrix(lift, model, decoding, "lift_mean")
        .to_numpy(dtype=float)
        for model in MODEL_ORDER
    ]

    # One symmetric scale across all three models, so the panels are
    # comparable and white falls exactly on lift = 1.
    #
    # Symmetric means the low end is 1 - span, which dips below zero whenever
    # some pair reaches a lift above 2 (it does, for greedy). A lift cannot be
    # negative, so that stretch of blue is simply unreachable - the tick list
    # below starts at 0 so no impossible value is ever printed. A TwoSlopeNorm
    # would remove the unreachable tail, but it renders the colorbar blank
    # under constrained layout in matplotlib 3.10, so this is the trade.
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    span = max(abs(1 - finite.min()), abs(finite.max() - 1))
    vmin, vmax = 1 - span, 1 + span

    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad(UNDEFINED_COLOR)

    fig, axes = plt.subplots(1, 3, figsize=(SINGLE_COLUMN_WIDTH, 2.6),
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02)

    image = None
    for column, (ax, model, matrix) in enumerate(zip(axes, MODEL_ORDER,
                                                     matrices)):
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=colormap,
                          vmin=vmin, vmax=vmax)
        ax.set_title(MODEL_LABELS[model], fontsize=PANEL_TITLE_FONTSIZE,
                     pad=3)

        for a in range(n):
            for b in range(n):
                value = matrix[a, b]
                if np.isfinite(value):
                    ax.text(b, a, f"{value:.2f}", ha="center", va="center",
                            fontsize=CELL_FONTSIZE,
                            color=_cell_text_color(image, value))
        _matrix_axis(ax, show_y_labels=column == 0)

    colorbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.012,
                            aspect=28)
    colorbar.set_label("lift", fontsize=TICK_FONTSIZE)
    low = max(vmin, 0.0)
    colorbar.set_ticks([round(v, 2) for v in
                        (low, (1 + low) / 2, 1.0, (1 + vmax) / 2, vmax)])
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=2)
    colorbar.ax.axhline(1.0, color="black", linewidth=0.8)

    fig.suptitle("Semantic co-occurrence: lift", fontsize=SUPTITLE_FONTSIZE)
    # Leading blank line is padding: the rotated tick labels reach a long way
    # down and the note otherwise sits right on top of them.
    fig.supxlabel(
        "\n" + _decoding_subtitle(decoding)
        + "\nGrey diagonal: undefined. Lift on thin-support pairs is volatile"
        " - see support counts in S6b_cooccurrence_ranked_pairs.csv.",
        fontsize=SUBTITLE_FONTSIZE - 1, color="#444444",
    )
    return _save(fig, path)


# ---------------------------------------------------------------------------
# F3 - correctness, present vs absent
# ---------------------------------------------------------------------------

def figure_correctness_presence(association: pd.DataFrame, path: Path,
                                decoding: str = "Stochastic") -> Path:
    """F3: P(correct | present) - P(correct | absent), with bootstrap CIs.

    Associational: the panel says where correctness and stated content move
    together, not that one produces the other.
    """
    frame = _flat(association)
    frame = frame.loc[frame["decoding_group"].eq(decoding)]

    y = np.arange(len(CATEGORY_ORDER))
    offsets = [0.25, 0.0, -0.25]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for model, offset in zip(MODEL_ORDER, offsets):
        subset = frame.loc[frame["model"].eq(model)].set_index(
            "category").reindex(CATEGORY_ORDER)
        centre = 100 * subset["delta"].to_numpy(dtype=float)
        low = 100 * subset["ci_low"].to_numpy(dtype=float)
        high = 100 * subset["ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            centre, y + offset,
            xerr=[centre - low, high - centre],
            fmt="o", markersize=4.5, capsize=2.5, elinewidth=1.0,
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
        )

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("Judgment", " Judgment")
                        .replace("Comparison", " Comparison")
                        for c in CATEGORY_ORDER], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(
        "percentage points: P(correct | category present)\n"
        "- P(correct | category absent)"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# F4 - correctness stability
# ---------------------------------------------------------------------------

def figure_correctness_stability(stability: pd.DataFrame, path: Path) -> Path:
    """F4: mean Q per category across the 0/3 - 3/3 stability groups.

    One panel per model. Q is the share of a game's three stochastic runs whose
    justification invokes the category, so a rising line means the category is
    more prevalent in games the model gets right more consistently - a property
    of the games as much as of the model.
    """
    frame = _flat(stability)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(CATEGORY_ORDER)))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        subset = frame.loc[frame["model"].eq(model)]
        group_sizes = (
            subset.drop_duplicates("k_correct_runs")
            .set_index("k_correct_runs")["n_games_in_group"].to_dict()
        )
        for color, category in zip(colors, CATEGORY_ORDER):
            line = subset.loc[subset["category"].eq(category)].sort_values(
                "k_correct_runs")
            ax.plot(line["k_correct_runs"], 100 * line["mean_q"],
                    marker="o", markersize=3.5, linewidth=1.3, color=color,
                    label=category)
        ax.set_title(MODEL_LABELS[model], fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(
            [f"{K_LABELS[k]}\nn={group_sizes.get(k, 0)}" for k in range(4)],
            fontsize=7,
        )
        ax.set_xlabel("correct stochastic runs", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("mean % of a game's runs\ninvoking the category")
    axes[-1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1.0),
                    loc="upper left")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# F5 - within-game contrasts
# ---------------------------------------------------------------------------

def figure_within_game(contrasts: pd.DataFrame, path: Path) -> Path:
    """F5: within mixed games, category presence in correct minus incorrect runs.

    Model and transcript are held fixed here, so this is the panel that says
    whether the aggregate association in F3 survives when game difficulty
    cannot explain it.
    """
    frame = _flat(contrasts)
    y = np.arange(len(CATEGORY_ORDER))
    offsets = [0.25, 0.0, -0.25]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for model, offset in zip(MODEL_ORDER, offsets):
        subset = frame.loc[frame["model"].eq(model)].set_index(
            "category").reindex(CATEGORY_ORDER)
        centre = 100 * subset["delta_within"].to_numpy(dtype=float)
        low = 100 * subset["ci_low"].to_numpy(dtype=float)
        high = 100 * subset["ci_high"].to_numpy(dtype=float)
        n_mixed = int(subset["n_mixed_games"].iloc[0])
        ax.errorbar(
            centre, y + offset,
            xerr=[centre - low, high - centre],
            fmt="o", markersize=4.5, capsize=2.5, elinewidth=1.0,
            color=MODEL_COLORS[model],
            label=f"{MODEL_LABELS[model]} (n={n_mixed} mixed games)",
        )

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("Judgment", " Judgment")
                        .replace("Comparison", " Comparison")
                        for c in CATEGORY_ORDER], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(
        "percentage points: category presence in correct runs\n"
        "- presence in incorrect runs, same model and game"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_final_figures(tables: Dict[str, pd.DataFrame],
                        directory: Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return [
        figure_semantic_prevalence(
            tables["S3_model_semantic_prevalence"],
            directory / "F1_semantic_prevalence.png"),
        figure_cooccurrence_prevalence(
            tables["S5_cooccurrence_joint_prevalence"],
            directory / "F2_semantic_cooccurrence_prevalence.png"),
        figure_cooccurrence_lift(
            tables["S6_cooccurrence_lift"],
            directory / "F2b_semantic_cooccurrence_lift.png"),
        figure_correctness_presence(
            tables["S7_correctness_presence_association"],
            directory / "F3_correctness_presence_association.png"),
        figure_correctness_stability(
            tables["S9_correctness_stability_semantics"],
            directory / "F4_correctness_stability.png"),
        figure_within_game(
            tables["S10_within_game_correctness_contrasts"],
            directory / "F5_within_game_correctness.png"),
        figure_cooccurrence_prevalence(
            tables["S5_cooccurrence_joint_prevalence"],
            directory / "F2c_semantic_cooccurrence_prevalence_greedy.png",
            decoding="Greedy"),
        figure_cooccurrence_lift(
            tables["S6_cooccurrence_lift"],
            directory / "F2d_semantic_cooccurrence_lift_greedy.png",
            decoding="Greedy"),
        figure_correctness_presence(
            tables["S7_correctness_presence_association"],
            directory / "F3b_correctness_presence_greedy.png",
            decoding="Greedy"),
    ]
