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

def figure_cooccurrence(joint: pd.DataFrame, lift: pd.DataFrame,
                        path: Path, decoding: str = "Stochastic") -> Path:
    """F2: joint prevalence (top row) and lift (bottom row) per model.

    Two rows because they answer different questions and are read differently:
    joint prevalence is how often a pair actually co-occurs, lift is whether it
    does so more than its marginals imply. The lift diagonal is blank by
    construction - see `semantic_final.cooccurrence`.
    """
    # hspace is generous on purpose: the rotated tick labels of the top row
    # otherwise run into the titles of the bottom row.
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.4),
                             gridspec_kw={"hspace": 0.45})
    n = len(CATEGORY_ORDER)
    labels = [c.replace("Judgment", "Judg.").replace("Comparison", "Comp.")
              for c in CATEGORY_ORDER]

    joint_images = []
    for j, model in enumerate(MODEL_ORDER):
        matrix = 100 * cooccurrence_matrix(
            joint, model, decoding, "joint_prevalence_mean"
        ).to_numpy(dtype=float)
        image = axes[0, j].imshow(matrix, cmap="viridis", vmin=0, vmax=100)
        joint_images.append(image)
        axes[0, j].set_title(f"{MODEL_LABELS[model]} - joint prevalence (%)",
                             fontsize=9)
        for a in range(n):
            for b in range(n):
                value = matrix[a, b]
                axes[0, j].text(
                    b, a, f"{value:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if value < 55 else "black",
                )

    lift_matrices = [
        cooccurrence_matrix(lift, model, decoding, "lift_mean").to_numpy(dtype=float)
        for model in MODEL_ORDER
    ]
    finite = np.concatenate([m[np.isfinite(m)] for m in lift_matrices])
    span = max(abs(1 - finite.min()), abs(finite.max() - 1))
    vmin, vmax = 1 - span, 1 + span

    lift_image = None
    for j, model in enumerate(MODEL_ORDER):
        matrix = lift_matrices[j]
        lift_image = axes[1, j].imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[1, j].set_title(f"{MODEL_LABELS[model]} - lift", fontsize=9)
        for a in range(n):
            for b in range(n):
                value = matrix[a, b]
                if np.isfinite(value):
                    axes[1, j].text(b, a, f"{value:.2f}", ha="center",
                                    va="center", fontsize=6, color="black")

    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(labels if col == 0 else [""] * n, fontsize=6)

    fig.colorbar(joint_images[-1], ax=axes[0, :], fraction=0.020, pad=0.01)
    fig.colorbar(lift_image, ax=axes[1, :], fraction=0.020, pad=0.01)
    fig.suptitle(
        f"Justification-level semantic co-occurrence ({decoding.lower()}, "
        "mean of 3 runs)", fontsize=10,
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
        figure_cooccurrence(
            tables["S5_cooccurrence_joint_prevalence"],
            tables["S6_cooccurrence_lift"],
            directory / "F2_semantic_cooccurrence.png"),
        figure_correctness_presence(
            tables["S7_correctness_presence_association"],
            directory / "F3_correctness_presence_association.png"),
        figure_correctness_stability(
            tables["S9_correctness_stability_semantics"],
            directory / "F4_correctness_stability.png"),
        figure_within_game(
            tables["S10_within_game_correctness_contrasts"],
            directory / "F5_within_game_correctness.png"),
        figure_cooccurrence(
            tables["S5_cooccurrence_joint_prevalence"],
            tables["S6_cooccurrence_lift"],
            directory / "F2b_semantic_cooccurrence_greedy.png",
            decoding="Greedy"),
        figure_correctness_presence(
            tables["S7_correctness_presence_association"],
            directory / "F3b_correctness_presence_greedy.png",
            decoding="Greedy"),
    ]
