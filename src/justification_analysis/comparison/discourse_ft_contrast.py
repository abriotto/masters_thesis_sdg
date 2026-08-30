"""BASE vs fine-tuned discourse contrast (RQ3), on matched games.

This is NOT a second characterisation of the fine-tuned models. It answers one
question per model: what changed in the stated justification structure after
familiarisation? So every quantity here is a within-model, paired BASE-FT
difference, never a standalone FT profile.

## What is reused unchanged

The statistical definitions are the frozen RQ2 ones, reused verbatim:

* density = 100 * (relations summed over the games) / (words summed over the
  same games) - a ratio of sums per run, not a mean of per-justification rates;
* each run is computed independently, then the three stochastic runs are
  averaged into one value;
* the word denominator is `pipeline.corpus.WORD_PATTERN`;
* the bootstrap resamples GAMES with replacement and reuses one multiplicity
  vector everywhere it has to be paired.

`discourse_final.run_level_table` could not simply be called: it asserts 191
justifications per run, which is false on a matched subset by construction. The
arithmetic below is the same; only the assertion differs.

## Why matched games

Greedy is not analysed - the fine-tuned models were never run greedily.

Three fine-tuned generations are unusable, and they are not evenly spread:

* E2B - one generation was deleted outright, so its game has 2 runs, not 3;
* 31B - three generations survive as rows with EMPTY justification text. Left
  in, they would contribute 0 words and 0 relations to their run and quietly
  depress that model's fine-tuned density.

A game is retained for a model only if BOTH conditions have all three
stochastic runs AND no justification is empty in either condition. The
excluded games are dropped from BOTH conditions, so the BASE side of every
contrast is recomputed on exactly the games the FT side has. Comparing a
fine-tuned value on 188 games against the frozen BASE value on 191 would
confound the change with the subset.

The retained sets are DERIVED, not hard-coded, and asserted against the
expected sizes at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.justification_analysis.pipeline import manifest as manifest_module
from src.justification_analysis.pipeline.config import AnalysisConfig, default_config
from src.justification_analysis.pipeline.corpus import load_corpus

PDTB_TOP_LEVEL = ("Comparison", "Contingency", "Expansion", "Temporal")
CATEGORY_ORDER = ["Contingency", "Comparison", "Expansion", "Temporal"]

# Same seed and replicate count as the frozen RQ2 bootstrap.
BOOTSTRAP_SEED = 20260826
BOOTSTRAP_REPLICATES = 10_000

BASE_STAGE = "base"
FT_STAGE = "ft"

# The level-2 senses that characterised the BASE models. Others are screened
# for a substantial change rather than catalogued.
FOCUS_SENSES = (
    "Contingency.Cause",
    "Contingency.Condition",
    "Comparison.Contrast",
    "Expansion.Conjunction",
    "Temporal.Asynchronous",
    "Temporal.Synchrony",
)


@dataclass
class ConditionData:
    """One condition's manifest-verified relations and its corpus."""
    stage: str
    config: AnalysisConfig
    corpus: pd.DataFrame
    accepted: pd.DataFrame
    manifest: dict


def load_condition(stage: str, decoding: str = "Stochastic") -> ConditionData:
    """Load one stage, refusing any artifact that is not this corpus's.

    Goes through `manifest.load_verified_candidates`, so a stale or missing
    parser artifact stops the analysis here rather than producing a plausible
    contrast against the wrong corpus.
    """
    config = default_config(stage=stage)
    corpus = load_corpus(config)
    candidates, manifest = manifest_module.load_verified_candidates(config, corpus)

    corpus = corpus.loc[corpus["decoding_group"].eq(decoding)].copy()
    accepted = candidates.loc[
        candidates["is_connective"]
        & candidates["justification_id"].isin(corpus["justification_id"])
    ].copy()

    assert (candidates["relation_type"] == "Explicit").all(), \
        f"{stage}: a non-explicit relation is present"
    assert not accepted["raw_sense"].isin(["NoSense", "EntRel"]).any(), \
        f"{stage}: NoSense or EntRel leaked into the accepted relations"
    assert accepted["top_level"].isin(PDTB_TOP_LEVEL).all(), \
        f"{stage}: an accepted relation carries no top-level class"
    return ConditionData(stage, config, corpus, accepted, manifest)


def matched_games(base: ConditionData, ft: ConditionData,
                  model: str, n_runs: int = 3) -> List[str]:
    """Games usable for this model in BOTH conditions.

    Usable means: three stochastic runs present, and no empty justification.
    An empty justification is a failed generation, not a short one - it would
    otherwise contribute zero words and zero relations to the density.
    """
    def usable(condition: ConditionData) -> set:
        rows = condition.corpus.loc[condition.corpus["model"].eq(model)]
        complete = {
            game for game, group in rows.groupby("game_id")
            if len(group) == n_runs
        }
        empty = set(
            rows.loc[rows["justification"].str.strip().eq(""), "game_id"]
        )
        return complete - empty

    return sorted(usable(base) & usable(ft))


def excluded_games(base: ConditionData, ft: ConditionData,
                   model: str) -> pd.DataFrame:
    """Which games were dropped for this model, and why. For the audit."""
    games = set(base.corpus.loc[base.corpus["model"].eq(model), "game_id"]) | \
        set(ft.corpus.loc[ft.corpus["model"].eq(model), "game_id"])
    keep = set(matched_games(base, ft, model))
    rows = []
    for game in sorted(games - keep):
        reasons = []
        for condition in (base, ft):
            rows_c = condition.corpus.loc[
                condition.corpus["model"].eq(model)
                & condition.corpus["game_id"].eq(game)
            ]
            if len(rows_c) != 3:
                missing = 3 - len(rows_c)
                reasons.append(
                    f"{condition.stage}: {len(rows_c)} runs "
                    f"({missing} generation{'s' if missing != 1 else ''} absent)"
                )
            n_empty = int(rows_c["justification"].str.strip().eq("").sum())
            if n_empty:
                reasons.append(f"{condition.stage}: {n_empty} empty justification")
        rows.append({"model": model, "game_id": game,
                     "reason": "; ".join(reasons)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per (condition, run, game) matrices - the basis for both the point estimates
# and the bootstrap, so the two can never diverge.
# ---------------------------------------------------------------------------

def _sense_metrics(accepted: pd.DataFrame) -> List[str]:
    return sorted(accepted["raw_sense"].dropna().unique())


def build_matrices(condition: ConditionData, model: str, games: Sequence[str],
                   senses: Sequence[str]) -> Tuple[List[str], np.ndarray, Dict]:
    """Words and per-metric counts, shaped (n_runs, n_games).

    Aligned on the given game order so BASE and FT index the same games, which
    is what makes the resampling paired.
    """
    rows = condition.corpus.loc[
        condition.corpus["model"].eq(model)
        & condition.corpus["game_id"].isin(games)
    ]
    runs = sorted(rows["run_label"].unique())
    index = {game: i for i, game in enumerate(games)}

    words = np.zeros((len(runs), len(games)))
    metrics = {name: np.zeros((len(runs), len(games)))
               for name in ["overall", *PDTB_TOP_LEVEL, *senses]}

    accepted = condition.accepted.loc[
        condition.accepted["justification_id"].isin(rows["justification_id"])
    ]
    total = accepted.groupby("justification_id").size()
    by_class = accepted.groupby(["justification_id", "top_level"],
                                observed=True).size()
    by_sense = accepted.groupby(["justification_id", "raw_sense"],
                                observed=True).size()

    for r, run in enumerate(runs):
        run_rows = rows.loc[rows["run_label"].eq(run)]
        assert len(run_rows) == len(games), (
            f"{condition.stage}/{model}/{run}: {len(run_rows)} justifications "
            f"for {len(games)} matched games")
        for justification_id, game, n_words in zip(
            run_rows["justification_id"], run_rows["game_id"], run_rows["n_words"]
        ):
            g = index[game]
            words[r, g] = n_words
            metrics["overall"][r, g] = total.get(justification_id, 0)
            for category in PDTB_TOP_LEVEL:
                metrics[category][r, g] = by_class.get(
                    (justification_id, category), 0)
            for sense in senses:
                metrics[sense][r, g] = by_sense.get((justification_id, sense), 0)

    assert words.sum() > 0, f"{condition.stage}/{model}: no words in the matched set"
    return runs, words, metrics


def _density(counts: np.ndarray, words: np.ndarray) -> float:
    """Per run, then averaged: the frozen definition."""
    per_run = 100 * counts.sum(axis=1) / words.sum(axis=1)
    return float(per_run.mean())


# ---------------------------------------------------------------------------
# The paired bootstrap
# ---------------------------------------------------------------------------

def contrast_model(base: ConditionData, ft: ConditionData, model: str,
                   games: Sequence[str], senses: Sequence[str],
                   n_replicates: int = BOOTSTRAP_REPLICATES,
                   seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """FT - BASE density differences with paired 95% percentile CIs.

    One replicate resamples the matched games with replacement and applies the
    SAME resampled games to both conditions. The pairing is what removes
    between-game variation from the difference: a game that is verbose in BASE
    is verbose in FT too, and resampling it affects both sides together.
    """
    _, base_words, base_metrics = build_matrices(base, model, games, senses)
    _, ft_words, ft_metrics = build_matrices(ft, model, games, senses)

    n_games = len(games)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_games, np.full(n_games, 1 / n_games), size=n_replicates
    ).astype(np.float64)                                       # (B, G)

    base_w = base_words @ weights.T                            # (R, B)
    ft_w = ft_words @ weights.T

    rows = []
    for metric in ["overall", *PDTB_TOP_LEVEL, *senses]:
        base_point = _density(base_metrics[metric], base_words)
        ft_point = _density(ft_metrics[metric], ft_words)

        base_boot = (100 * (base_metrics[metric] @ weights.T) / base_w).mean(axis=0)
        ft_boot = (100 * (ft_metrics[metric] @ weights.T) / ft_w).mean(axis=0)
        differences = ft_boot - base_boot
        low, high = np.percentile(differences, [2.5, 97.5])

        rows.append({
            "model": model,
            "metric": "All relations" if metric == "overall" else metric,
            "level": ("overall" if metric == "overall"
                      else "top_level" if metric in PDTB_TOP_LEVEL
                      else "sense"),
            "n_games": n_games,
            "base": base_point,
            "ft": ft_point,
            "delta": ft_point - base_point,
            "ci_low": low,
            "ci_high": high,
            "ci_excludes_zero": bool(low > 0 or high < 0),
        })
    return pd.DataFrame(rows)


def descriptive_profile(condition: ConditionData, model: str,
                        games: Sequence[str]) -> Dict[str, float]:
    """Per-run then averaged: coverage, length, density. Matched games only."""
    rows = condition.corpus.loc[
        condition.corpus["model"].eq(model)
        & condition.corpus["game_id"].isin(games)
    ]
    counts = condition.accepted.groupby("justification_id").size()
    per_just = rows.assign(
        n=rows["justification_id"].map(counts).fillna(0).astype(int)
    )
    per_just["has"] = (per_just["n"] > 0).astype(int)

    per_run = per_just.groupby("run_label").agg(
        pct_with=("has", lambda values: 100 * values.mean()),
        mean_words=("n_words", "mean"),
        total_relations=("n", "sum"),
        total_words=("n_words", "sum"),
    )
    per_run["density"] = 100 * per_run["total_relations"] / per_run["total_words"]
    return {
        "pct_justifications_with_relation": float(per_run["pct_with"].mean()),
        "mean_words_per_justification": float(per_run["mean_words"].mean()),
        "relations_per_100_words": float(per_run["density"].mean()),
        "n_runs": int(len(per_run)),
        "n_justifications": int(len(per_just)),
    }


def composition_contrast(base: ConditionData, ft: ConditionData, model: str,
                         games: Sequence[str], parts: Sequence[str],
                         whole: str,
                         n_replicates: int = BOOTSTRAP_REPLICATES,
                         seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Share of a top-level class taken by each of its level-2 senses.

    Answers "did the balance inside Contingency shift", which a density
    contrast alone cannot: both senses can fall while the balance between them
    moves. The denominator is the class, not all relations.
    """
    senses = list(parts)
    _, base_words, base_metrics = build_matrices(base, model, games, senses)
    _, ft_words, ft_metrics = build_matrices(ft, model, games, senses)

    n_games = len(games)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_games, np.full(n_games, 1 / n_games), size=n_replicates
    ).astype(np.float64)

    def share(metrics, part):
        per_run = 100 * metrics[part].sum(axis=1) / metrics[whole].sum(axis=1)
        return float(per_run.mean())

    def share_boot(metrics, part):
        numerator = metrics[part] @ weights.T
        denominator = metrics[whole] @ weights.T
        with np.errstate(invalid="ignore", divide="ignore"):
            per_run = 100 * numerator / denominator
        return np.nanmean(per_run, axis=0)

    rows = []
    for part in parts:
        base_point, ft_point = share(base_metrics, part), share(ft_metrics, part)
        differences = share_boot(ft_metrics, part) - share_boot(base_metrics, part)
        low, high = np.percentile(differences, [2.5, 97.5])
        rows.append({
            "model": model, "whole": whole, "part": part, "n_games": n_games,
            "base_pct": base_point, "ft_pct": ft_point,
            "delta_pct": ft_point - base_point,
            "ci_low": low, "ci_high": high,
            "ci_excludes_zero": bool(low > 0 or high < 0),
        })
    return pd.DataFrame(rows)
