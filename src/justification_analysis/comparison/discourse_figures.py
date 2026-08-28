"""Thesis figures for the explicit-discourse-relation analysis (RQ2).

Pure matplotlib over the frames `discourse_final` returns, so every figure is
drawn from exactly the numbers in the corresponding final table - nothing is
recomputed here and nothing is smoothed, clipped or reordered for appearance.

Conventions, shared with the other cross-model figures in this thesis:

  * models are E2B / E4B / 31B, always in that order, viridis at 0.15/0.45/0.72;
  * stochastic bars are the mean across the three runs, error bars the SD;
  * greedy is a single run, so it never gets an error bar. It is overlaid as a
    separate marker rather than a fourth bar, to keep it visibly not-a-mean.

The final figure set is deliberately small: density, composition, and the
level-2 senses. There is no discourse-only co-occurrence figure - that analysis
belongs to the later joint discourse x semantic work.

`figure_contextual_filtering` is a METHOD figure, not a result, and is written
outside the final-results directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.justification_analysis.comparison.discourse_statistics import (
    CATEGORY_ORDER,
    MODEL_ORDER,
)

MODEL_LABELS = {"Gemma 4 2B": "E2B", "Gemma 4 4B": "E4B", "Gemma 4 31B": "31B"}
MODEL_COLORS = dict(zip(MODEL_ORDER, plt.cm.viridis([0.15, 0.45, 0.72])))
CATEGORY_COLORS = dict(zip(CATEGORY_ORDER, plt.cm.viridis([0.12, 0.40, 0.65, 0.88])))

DPI = 300


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _flat(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept a table either indexed by model/decoding or with them as columns."""
    if "model" in (frame.index.names or []):
        frame = frame.reset_index()
    out = frame.copy()
    out["model"] = out["model"].astype(str)
    out["decoding_group"] = out["decoding_group"].astype(str)
    return out


def _rows(frame: pd.DataFrame, decoding: str, model: str) -> pd.Series:
    match = frame.loc[frame["decoding_group"].eq(decoding) & frame["model"].eq(model)]
    assert len(match) == 1, f"expected one row for {model}/{decoding}, got {len(match)}"
    return match.iloc[0]


def figure_top_level_density(top_level_density: pd.DataFrame, path: Path) -> Path:
    """F2: relations per 100 words in each top-level class, by model.

    Stochastic is the comparison; greedy sits on top as an open diamond so the
    single-run value is visible without pretending to be a mean.
    """
    frame = _flat(top_level_density)
    x = np.arange(len(CATEGORY_ORDER))
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, model in enumerate(MODEL_ORDER):
        offset = (i - 1) * width
        stochastic = _rows(frame, "Stochastic", model)
        greedy = _rows(frame, "Greedy", model)
        ax.bar(
            x + offset,
            [stochastic[f"{c}_per_100_words_mean"] for c in CATEGORY_ORDER], width,
            yerr=[stochastic[f"{c}_per_100_words_sd"] for c in CATEGORY_ORDER],
            capsize=3, color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            error_kw={"elinewidth": 0.9},
        )
        ax.scatter(
            x + offset, [greedy[f"{c}_per_100_words_mean"] for c in CATEGORY_ORDER],
            marker="D", s=16, facecolor="white", edgecolor="black",
            linewidth=0.8, zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.set_ylabel("Relations per 100 words")
    ax.set_xlabel("PDTB top-level class")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        plt.Line2D([], [], marker="D", linestyle="none", markersize=5,
                   markerfacecolor="white", markeredgecolor="black")
    )
    labels.append("greedy (single run)")
    ax.legend(handles, labels, frameon=False, fontsize=8, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def figure_composition(composition: pd.DataFrame, path: Path) -> Path:
    """F3: what each model's relations consist of, shares summing to 100.

    Shares were computed per run and then averaged, so this is independent of
    how many relations a model produced.
    """
    frame = _flat(composition)
    rows = [(model, decoding) for model in MODEL_ORDER
            for decoding in ("Stochastic", "Greedy")]
    labels = [f"{MODEL_LABELS[m]}  {d.lower()}" for m, d in rows]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for category in CATEGORY_ORDER:
        values = np.array([
            _rows(frame, decoding, model)[f"{category}_pct_of_relations_mean"]
            for model, decoding in rows
        ], dtype=float)
        ax.barh(y, values, left=left, color=CATEGORY_COLORS[category],
                label=category, height=0.62)
        for yi, (start, value) in enumerate(zip(left, values)):
            if value >= 6:
                ax.text(start + value / 2, y[yi], f"{value:.0f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if category == "Temporal" else "white")
        left = left + values

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of accepted explicit relations (%)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    ax.spines[["top", "right", "left"]].set_visible(False)
    return _save(fig, path)


def figure_fine_grained_senses(fine_grained: pd.DataFrame, path: Path) -> Path:
    """F4: level-2 PDTB senses, stochastic runs only.

    Senses are ordered by corpus frequency. Only senses the corpus actually
    contains are drawn. `Expansion.Restatement` (n=1 corpus-wide) stays in the
    table; an all-but-invisible bar is the honest rendering of a singleton.
    """
    frame = _flat(fine_grained)
    order = (
        frame.groupby("raw_sense", observed=True)["total_count"].sum()
        .sort_values(ascending=False).index.tolist()
    )
    stochastic = frame.loc[frame["decoding_group"].eq("Stochastic")]
    y = np.arange(len(order))[::-1]
    height = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(order) + 1.4))
    for i, model in enumerate(MODEL_ORDER):
        rows = (
            stochastic.loc[stochastic["model"].eq(model)]
            .set_index("raw_sense").reindex(order)
        )
        ax.barh(
            y + (1 - i) * height, rows["mean_per_100_words"].fillna(0), height,
            xerr=rows["sd_per_100_words"].fillna(0), capsize=2,
            color=MODEL_COLORS[model], label=MODEL_LABELS[model],
            error_kw={"elinewidth": 0.8},
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Relations per 100 words (stochastic mean $\\pm$ SD)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def figure_bootstrap_forest(
    bootstrap: pd.DataFrame, path: Path, decoding: str = "Stochastic"
) -> Path:
    """F5: pairwise model differences with 95% percentile bootstrap intervals.

    One row per (metric, model pair), grouped by metric, with a zero reference
    line. A marker is filled when the interval excludes zero and hollow grey
    when it covers zero - the only inferential claim the bootstrap supports,
    read straight off the figure.

    Stochastic and greedy are drawn as separate figures: greedy is a single run,
    so its interval reflects between-game variation only and must not be read as
    the same quantity.
    """
    from matplotlib.transforms import blended_transform_factory

    frame = bootstrap.copy()
    if "metric" in (frame.index.names or []):
        frame = frame.reset_index()
    frame = frame.loc[frame["decoding_group"].astype(str).eq(decoding)]

    metric_order = ["All relations", *CATEGORY_ORDER]
    pairs = [(MODEL_ORDER[i], MODEL_ORDER[j])
             for i in range(len(MODEL_ORDER))
             for j in range(i + 1, len(MODEL_ORDER))]

    lookup = frame.set_index(
        [frame["metric"].astype(str), frame["model_a"].astype(str),
         frame["model_b"].astype(str)]
    )

    rows = []
    for metric in metric_order:
        for model_a, model_b in pairs:
            record = lookup.loc[(metric, model_a, model_b)]
            rows.append({
                "metric": metric,
                "label": f"{MODEL_LABELS[model_a]} − {MODEL_LABELS[model_b]}",
                "difference": float(record["difference"]),
                "ci_low": float(record["ci_low"]),
                "ci_high": float(record["ci_high"]),
                "excludes_zero": bool(record["ci_excludes_zero"]),
            })
    table = pd.DataFrame(rows)
    y = np.arange(len(table))[::-1]

    significant = plt.cm.viridis(0.30)
    inconclusive = "0.55"

    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(table) + 1.5))
    ax.axvline(0, color="black", linewidth=0.9, zorder=1)

    for yi, row in zip(y, table.itertuples(index=False)):
        colour = significant if row.excludes_zero else inconclusive
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=colour,
                linewidth=1.6, solid_capstyle="round", zorder=2)
        ax.plot(row.difference, yi, marker="o", markersize=5.5, color=colour,
                markerfacecolor=colour if row.excludes_zero else "white",
                markeredgewidth=1.2, zorder=3)

    # Group separators and the metric name for each block of three pairs.
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for index, metric in enumerate(metric_order):
        block = y[index * len(pairs):(index + 1) * len(pairs)]
        ax.text(-0.30, float(np.mean(block)), metric, transform=transform,
                ha="left", va="center", fontsize=9, fontweight="bold")
        if index:
            ax.axhline(float(block[0]) + 0.5, color="0.85", linewidth=0.7,
                       zorder=0)

    def fmt(value: float) -> str:
        # A bound that rounds to 0.00 without being zero gets a third decimal,
        # so "excludes zero" stays readable as a fact rather than a rounding.
        return f"{value:+.3f}" if abs(value) < 0.005 else f"{value:+.2f}"

    span = float(table["ci_high"].max() - table["ci_low"].min())
    left = float(table["ci_low"].min()) - 0.05 * span
    right = float(table["ci_high"].max()) + 0.45 * span
    for yi, row in zip(y, table.itertuples(index=False)):
        ax.text(right, yi,
                f"{fmt(row.difference)}  [{fmt(row.ci_low)}, {fmt(row.ci_high)}]",
                ha="right", va="center", fontsize=7.4, color="0.25",
                family="DejaVu Sans Mono")

    ax.set_yticks(y)
    ax.set_yticklabels(table["label"], fontsize=8.5)
    ax.set_xlim(left, right)
    ax.set_ylim(-0.8, len(table) - 0.2)
    ax.set_xlabel("Difference in relations per 100 words "
                  "(95% percentile bootstrap CI)")
    ax.set_title(f"{decoding} decoding", fontsize=10, loc="left")

    handles = [
        plt.Line2D([], [], marker="o", color=significant, markersize=5.5,
                   linewidth=1.6),
        plt.Line2D([], [], marker="o", color=inconclusive, markersize=5.5,
                   markerfacecolor="white", markeredgewidth=1.2, linewidth=1.6),
    ]
    ax.legend(handles, ["interval excludes zero", "interval covers zero"],
              frameon=False, fontsize=8, ncol=2, loc="lower right",
              bbox_to_anchor=(1.0, 1.005))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return _save(fig, path)


def build_final_figures(tables: Dict[str, pd.DataFrame],
                        figure_dir: Path) -> List[Path]:
    """The final figures, in the order the thesis references them."""
    figure_dir = Path(figure_dir)
    return [
        figure_top_level_density(
            tables["F2_top_level_density"],
            figure_dir / "F1_top_level_density.png",
        ),
        figure_composition(
            tables["F3_top_level_composition"],
            figure_dir / "F2_composition.png",
        ),
        figure_fine_grained_senses(
            tables["F4_fine_grained_senses"],
            figure_dir / "F3_fine_grained_senses.png",
        ),
        figure_bootstrap_forest(
            tables["F5_bootstrap_pairwise"],
            figure_dir / "F4_bootstrap_forest_stochastic.png",
            decoding="Stochastic",
        ),
        figure_bootstrap_forest(
            tables["F5_bootstrap_pairwise"],
            figure_dir / "F5_bootstrap_forest_greedy.png",
            decoding="Greedy",
        ),
    ]


# ---------------------------------------------------------------------------
# Method figure - not a result
# ---------------------------------------------------------------------------

def figure_contextual_filtering(
    acceptance_by_form: pd.DataFrame, path: Path, top_n: int = 15
) -> Path:
    """How often each candidate form survives the NoSense filter.

    Evidence that the contextual classifier does real work (`for`: 1,632
    candidates, 0 accepted), not a finding about the models. Ordered by
    candidate count, so the forms driving the corpus-level rejection rate are at
    the top.
    """
    frame = (
        acceptance_by_form.sort_values("n_candidates", ascending=False)
        .head(top_n).iloc[::-1]
    )
    y = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(frame) + 1.2))
    ax.barh(y, frame["pct_accepted"], color=plt.cm.viridis(0.45), height=0.66)
    for yi, (pct, n) in enumerate(zip(frame["pct_accepted"], frame["n_candidates"])):
        # A near-full bar leaves no room outside, so the count moves inside it.
        inside = pct > 88
        ax.text(
            pct - 1.5 if inside else pct + 1.5, yi, f"n={int(n):,}",
            va="center", ha="right" if inside else "left", fontsize=7.5,
            color="white" if inside else "black",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(frame["form"], fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Candidates accepted as connectives (%)")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)
