"""Canonical FINAL discourse results for RQ2. Standard discopy only.

The single input is

    discourse_parser/discopy_explicit_candidates.csv   filtered to is_connective

Nothing in this module reads DiMLex, the forced-span probe or the rejected
hybrid, and no artifact it writes may be derived from them. Outputs go to their
own directories - `thesis_tables/final_discourse/` and `figures/final_discourse/`
- so the final results are never confused with the exploratory tables that share
the parent folders.

Scope, deliberately small: overall explicit-relation density, the four top-level
PDTB densities, the relative four-class composition, the level-2 senses as a
secondary descriptive, and one inferential artifact - a paired game-level
bootstrap of pairwise model differences. No co-occurrence, no lexical counts as
results, no stratification, no significance battery.

Two aggregation rules hold everywhere and are not negotiable:

  * descriptive tables compute each run independently, then report mean and SD
    across the three stochastic runs. Greedy is one run, kept separate, SD
    undefined (NaN);
  * the bootstrap resamples GAMES, not justifications, and uses the same
    resampled game ids for every model - the models answered the same 191
    games, so the comparison is paired and the game-level variation cancels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.justification_analysis.comparison import discourse_comparison as dc
from src.justification_analysis.comparison import discourse_statistics as ds
from src.justification_analysis.pipeline import config as pipeline_config
from src.justification_analysis.pipeline import corpus as corpus_module
from src.justification_analysis.pipeline import manifest as manifest_module
from src.justification_analysis.comparison.discourse_statistics import (
    CATEGORY_ORDER,
    DECODING_ORDER,
    MODEL_ORDER,
    PDTB_TOP_LEVEL,
    RUN_KEYS,
)

# Output locations derive from the ACTIVE configuration, so a fine-tuned run
# writes into its own namespace and cannot overwrite a base artifact. There is
# no module-level base path any more - asking for a directory requires saying
# which stage you mean, or accepting the configured default.
def final_tables_dir(config=None) -> Path:
    config = config or pipeline_config.default_config()
    return config.final_discourse_tables


def final_figures_dir(config=None) -> Path:
    config = config or pipeline_config.default_config()
    return config.final_discourse_figures

BOOTSTRAP_SEED = 20260826
BOOTSTRAP_REPLICATES = 10_000

# Frozen at the pipeline freeze point.
INVARIANTS = {
    "candidates": 14209,
    "accepted": 5504,
    "justifications": 2292,
    "justifications_per_run": 191,
    "games": 191,
    "word_pattern_tokens": 169748,
}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_production_data(config=None, repo_root: Path = None):
    """The accepted relations and the justification frame, for the ACTIVE stage.

    Two things changed here, neither of them statistical:

      * the candidate table is no longer read by path. It goes through the
        manifest freshness gate, so an artifact built from a different corpus
        is refused rather than silently consumed;
      * the frozen base counts are no longer asserted unconditionally. They
        are properties of the base corpus, not of the pipeline, and a
        fine-tuned corpus is a different size. Structural checks that must
        hold for ANY corpus stay; the frozen values moved to
        `base_regression_checks`, applied only when the active stage is base.

    Every metric definition below this function is untouched.
    """
    if config is None:
        config = pipeline_config.AnalysisConfig(
            repo_root=Path(repo_root) if repo_root
            else pipeline_config.find_repo_root())

    justifications = corpus_module.load_corpus(config)
    candidates, manifest = manifest_module.load_verified_candidates(
        config, justifications)
    candidates = dc.normalise_candidates(candidates)
    accepted = candidates.loc[candidates["is_connective"]].copy()

    # Structural invariants - true of any corpus, in any stage.
    assert len(candidates) == len(accepted) + int((~candidates["is_connective"]).sum()), \
        "accepted + rejected does not equal enumerated"
    assert int(candidates["occurrence_id"].duplicated().sum()) == 0, \
        "duplicate occurrence ids"
    assert accepted["top_level"].isin(PDTB_TOP_LEVEL).all(), \
        "an accepted relation carries no top-level class"
    assert not accepted["raw_sense"].isin(["NoSense", "EntRel"]).any(), \
        "NoSense or EntRel leaked into the accepted relations"
    assert int(justifications["n_words"].sum()) > 0, "empty word denominator"

    if config.is_base:
        failures = base_regression_checks(candidates, accepted, justifications)
        assert not failures, (
            "base regression failed - the base corpus or artifact changed:\n"
            + "\n".join(failures))

    accepted.attrs["manifest"] = manifest
    justifications.attrs["manifest"] = manifest
    return accepted, justifications


def base_regression_checks(candidates, accepted, justifications) -> List[str]:
    """The frozen BASE values, kept as a regression rather than a pipeline rule.

    These describe the base corpus. They must never be applied to another
    stage - a fine-tuned corpus that reproduced 2,292 justifications and
    169,748 tokens exactly would be suspicious, not reassuring.
    """
    observed = {
        "candidates": len(candidates),
        "accepted": len(accepted),
        "justifications": len(justifications),
        "word_pattern_tokens": int(justifications["n_words"].sum()),
        "games": int(justifications["game_id"].nunique()),
        "justifications_per_run": int(
            justifications.groupby(RUN_KEYS, observed=True).size().unique()[0]),
    }
    return [
        f"{key}: expected {INVARIANTS[key]}, got {value}"
        for key, value in observed.items()
        if key in INVARIANTS and value != INVARIANTS[key]
    ]


def _order(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "model" in out.columns:
        out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    if "decoding_group" in out.columns:
        out["decoding_group"] = pd.Categorical(out["decoding_group"],
                                               DECODING_ORDER, ordered=True)
    cols = [c for c in ("model", "decoding_group") if c in out.columns]
    return out.sort_values(cols).reset_index(drop=True) if cols else out


# ---------------------------------------------------------------------------
# F0 - the per-run base every other table is built from
# ---------------------------------------------------------------------------

def run_level_table(accepted: pd.DataFrame,
                    justifications: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, decoding, run): counts, words and densities.

    Everything downstream aggregates this table, so the un-aggregated numbers
    stay inspectable and no statistic is computed twice by two routes.
    """
    base = justifications[
        ["justification_id", "model", "decoding_group", "run_label", "n_words"]
    ]
    counts = accepted.groupby("justification_id").size().rename("n")
    per_just = base.merge(counts, left_on="justification_id",
                          right_index=True, how="left")
    per_just["n"] = per_just["n"].fillna(0).astype(int)
    per_just["has"] = (per_just["n"] > 0).astype(int)

    run = per_just.groupby(RUN_KEYS, as_index=False).agg(
        n_justifications=("justification_id", "nunique"),
        total_words=("n_words", "sum"),
        total_relations=("n", "sum"),
        n_justifications_with_relation=("has", "sum"),
    )

    for category in PDTB_TOP_LEVEL:
        subset = accepted.loc[accepted["top_level"].eq(category)]
        cat_counts = subset.groupby("justification_id").size().rename("n_cat")
        frame = base.merge(cat_counts, left_on="justification_id",
                           right_index=True, how="left")
        frame["n_cat"] = frame["n_cat"].fillna(0).astype(int)
        totals = frame.groupby(RUN_KEYS, as_index=False)["n_cat"].sum()
        run = run.merge(totals.rename(columns={"n_cat": f"n_{category}"}),
                        on=RUN_KEYS)

    run["relations_per_100_words"] = 100 * run["total_relations"] / run["total_words"]
    run["relations_per_justification"] = (
        run["total_relations"] / run["n_justifications"]
    )
    run["pct_justifications_with_relation"] = (
        100 * run["n_justifications_with_relation"] / run["n_justifications"]
    )
    for category in PDTB_TOP_LEVEL:
        run[f"{category}_per_100_words"] = (
            100 * run[f"n_{category}"] / run["total_words"]
        )
        run[f"{category}_pct_of_relations"] = (
            100 * run[f"n_{category}"] / run["total_relations"]
        )

    assert (run["n_justifications"] == INVARIANTS["justifications_per_run"]).all()
    assert run[[f"n_{c}" for c in PDTB_TOP_LEVEL]].sum(axis=1).eq(
        run["total_relations"]).all(), "class counts do not sum to the total"
    return _order(run)


def _summarise(run_level: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Mean and SD across runs. One stochastic group of 3, one greedy of 1."""
    aggregation = {"n_runs": ("run_label", "nunique")}
    for column in columns:
        aggregation[f"{column}_mean"] = (column, "mean")
        aggregation[f"{column}_sd"] = (column, "std")
    summary = run_level.groupby(["model", "decoding_group"], as_index=False,
                                observed=True).agg(**aggregation)
    return _order(summary)


# ---------------------------------------------------------------------------
# F1 / F2 / F3 - the primary descriptives
# ---------------------------------------------------------------------------

def overall_density(run_level: pd.DataFrame) -> pd.DataFrame:
    """A. How much explicit discourse marking each model produces."""
    return _summarise(run_level, [
        "relations_per_100_words",
        "relations_per_justification",
        "pct_justifications_with_relation",
    ])


def top_level_density(run_level: pd.DataFrame) -> pd.DataFrame:
    """B. Density of each of the four top-level PDTB classes."""
    summary = _summarise(
        run_level, [f"{c}_per_100_words" for c in CATEGORY_ORDER]
    )
    return summary


def top_level_composition(run_level: pd.DataFrame) -> pd.DataFrame:
    """C. What a model's relations consist of, independent of how many there are.

    Shares are computed within each run and then averaged, so a run that
    happened to produce more relations does not dominate the mean.
    """
    summary = _summarise(
        run_level, [f"{c}_pct_of_relations" for c in CATEGORY_ORDER]
    )
    means = summary[[f"{c}_pct_of_relations_mean" for c in CATEGORY_ORDER]].sum(axis=1)
    assert np.allclose(means, 100.0), "class shares do not sum to 100"
    return summary


# ---------------------------------------------------------------------------
# F4 - level-2 senses, secondary descriptive
# ---------------------------------------------------------------------------

def fine_grained_senses(accepted: pd.DataFrame,
                        justifications: pd.DataFrame) -> pd.DataFrame:
    """D. The level-2 PDTB senses actually produced, per model and decoding.

    Only senses observed in the corpus appear: the checkpoint supports 16
    non-NoSense labels but the corpus uses 9, and printing seven structural
    zeros would suggest the models were measured against something they were
    never scored on. `Expansion.Restatement` (n=1 corpus-wide) is kept.
    """
    senses = ds.fine_grained_sense_statistics(accepted, justifications)
    run_level = senses["run_level"]

    shares = run_level.copy()
    totals = (
        run_level.groupby(RUN_KEYS, observed=True)["n_occurrences"]
        .sum().rename("run_total")
    )
    shares = shares.merge(totals, on=RUN_KEYS, how="left")
    shares["pct_of_relations"] = 100 * shares["n_occurrences"] / shares["run_total"]

    table = shares.groupby(
        ["model", "decoding_group", "top_level", "raw_sense"],
        as_index=False, observed=True,
    ).agg(
        n_runs=("run_label", "nunique"),
        total_count=("n_occurrences", "sum"),
        mean_count_per_run=("n_occurrences", "mean"),
        mean_per_100_words=("per_100_words", "mean"),
        sd_per_100_words=("per_100_words", "std"),
        mean_pct_of_relations=("pct_of_relations", "mean"),
        sd_pct_of_relations=("pct_of_relations", "std"),
    )
    assert int(table["total_count"].sum()) == INVARIANTS["accepted"], \
        "level-2 senses do not sum to the accepted relations"
    return _order(table)


# ---------------------------------------------------------------------------
# F5 - paired game-level bootstrap
# ---------------------------------------------------------------------------

def _metric_matrices(
    accepted: pd.DataFrame,
    justifications: pd.DataFrame,
    decoding: str,
) -> Tuple[List[str], List[str], np.ndarray, Dict[str, np.ndarray]]:
    """Per (model, run, game) word counts and relation counts.

    Returns the game order, the run labels, the words array and one counts
    array per metric, all shaped (n_models, n_runs, n_games) and aligned on the
    same game order - which is what makes the resampling paired.
    """
    subset = justifications.loc[justifications["decoding_group"].eq(decoding)]
    games = sorted(subset["game_id"].unique())
    runs = sorted(subset["run_label"].unique())
    game_index = {game: i for i, game in enumerate(games)}

    shape = (len(MODEL_ORDER), len(runs), len(games))
    words = np.zeros(shape)
    metrics = {name: np.zeros(shape) for name in ["overall", *PDTB_TOP_LEVEL]}

    counts = accepted.groupby(["justification_id", "top_level"], observed=True).size()
    per_just_total = accepted.groupby("justification_id").size()

    for m, model in enumerate(MODEL_ORDER):
        for r, run in enumerate(runs):
            rows = subset.loc[subset["model"].eq(model) & subset["run_label"].eq(run)]
            assert len(rows) == len(games), \
                f"{model}/{run}: {len(rows)} justifications for {len(games)} games"
            for justification_id, game, n_words in zip(
                rows["justification_id"], rows["game_id"], rows["n_words"]
            ):
                g = game_index[game]
                words[m, r, g] = n_words
                metrics["overall"][m, r, g] = per_just_total.get(justification_id, 0)
                for category in PDTB_TOP_LEVEL:
                    metrics[category][m, r, g] = counts.get(
                        (justification_id, category), 0
                    )
    return games, runs, words, metrics


def paired_game_bootstrap(
    accepted: pd.DataFrame,
    justifications: pd.DataFrame,
    n_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """95% percentile CIs for pairwise model differences in relation density.

    One replicate: resample the 191 game ids with replacement, use the SAME
    resampled ids for every model, keep all stochastic realisations of each
    sampled game, compute the density separately per run, average the run
    values into one model value, then difference the models. Stochastic and
    greedy are bootstrapped separately; greedy has a single run, so its
    interval reflects between-game variation only.

    The resample is expressed as a multinomial multiplicity vector, which is
    exactly equivalent to sampling ids with replacement and lets every model,
    run and metric reuse the identical replicate weights.
    """
    rows = []
    for decoding in DECODING_ORDER:
        games, runs, words, metrics = _metric_matrices(
            accepted, justifications, decoding
        )
        n_games = len(games)
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(
            n_games, np.full(n_games, 1 / n_games), size=n_replicates
        ).astype(np.float64)                                   # (B, G)

        flat_words = words.reshape(-1, n_games)                # (M*R, G)
        boot_words = flat_words @ weights.T                    # (M*R, B)

        for metric, counts in metrics.items():
            flat_counts = counts.reshape(-1, n_games)
            boot_counts = flat_counts @ weights.T
            rates = 100 * boot_counts / boot_words             # (M*R, B)
            rates = rates.reshape(len(MODEL_ORDER), len(runs), n_replicates)
            model_values = rates.mean(axis=1)                  # (M, B)

            observed = 100 * counts.sum(axis=2) / words.sum(axis=2)  # (M, R)
            observed = observed.mean(axis=1)                   # (M,)

            for i in range(len(MODEL_ORDER)):
                for j in range(i + 1, len(MODEL_ORDER)):
                    differences = model_values[i] - model_values[j]
                    low, high = np.percentile(differences, [2.5, 97.5])
                    rows.append({
                        "decoding_group": decoding,
                        "metric": "All relations" if metric == "overall" else metric,
                        "model_a": MODEL_ORDER[i],
                        "model_b": MODEL_ORDER[j],
                        "rate_a": observed[i],
                        "rate_b": observed[j],
                        "difference": observed[i] - observed[j],
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                        "n_runs_averaged": len(runs),
                        "n_games": n_games,
                        "n_replicates": n_replicates,
                        "seed": seed,
                    })

    metric_order = ["All relations", *CATEGORY_ORDER]
    table = pd.DataFrame(rows)
    table["metric"] = pd.Categorical(table["metric"], metric_order, ordered=True)
    table["decoding_group"] = pd.Categorical(table["decoding_group"],
                                             DECODING_ORDER, ordered=True)
    return table.sort_values(
        ["decoding_group", "metric", "model_a", "model_b"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

FINAL_TABLE_CAPTIONS = {
    "F0_run_level": "Per-run explicit relation counts, words and densities",
    "F1_overall_density": "Overall explicit relation density by model and decoding",
    "F2_top_level_density": "Top-level PDTB relation density per 100 words",
    "F3_top_level_composition": "Relative composition of explicit relations (\\%)",
    "F4_fine_grained_senses": "Level-2 PDTB senses by model and decoding",
    "F5_bootstrap_pairwise": (
        "Paired game-level bootstrap, 95\\% percentile CIs for pairwise "
        "model differences in relation density"
    ),
}


def build_final_tables(accepted: pd.DataFrame,
                       justifications: pd.DataFrame,
                       n_replicates: int = BOOTSTRAP_REPLICATES,
                       seed: int = BOOTSTRAP_SEED) -> Dict[str, pd.DataFrame]:
    run_level = run_level_table(accepted, justifications)
    return {
        "F0_run_level": run_level.set_index(["model", "decoding_group", "run_label"]),
        "F1_overall_density": overall_density(run_level).set_index(
            ["model", "decoding_group"]),
        "F2_top_level_density": top_level_density(run_level).set_index(
            ["model", "decoding_group"]),
        "F3_top_level_composition": top_level_composition(run_level).set_index(
            ["model", "decoding_group"]),
        "F4_fine_grained_senses": fine_grained_senses(
            accepted, justifications).set_index(
            ["model", "decoding_group", "top_level", "raw_sense"]),
        "F5_bootstrap_pairwise": paired_game_bootstrap(
            accepted, justifications, n_replicates, seed).set_index(
            ["decoding_group", "metric", "model_a", "model_b"]),
    }


def write_final_tables(tables: Dict[str, pd.DataFrame], table_dir: Path) -> List[Path]:
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in tables.items():
        csv_path = table_dir / f"{name}.csv"
        frame.to_csv(csv_path, encoding="utf-8-sig")
        try:
            ds.to_latex(frame, table_dir / f"{name}.tex",
                        caption=FINAL_TABLE_CAPTIONS.get(name, name))
        except Exception as error:  # pragma: no cover - formatting only
            print(f"  (latex skipped for {name}: {error})")
        written.append(csv_path)
    return written
