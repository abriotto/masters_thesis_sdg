"""BASE vs fine-tuned SEMANTIC contrast (RQ3), on matched games.

This is NOT a second characterisation of the fine-tuned models, and not a
repeat of the RQ2 semantic analysis. It answers one question per model: did
familiarisation change WHAT KINDS OF INFORMATION the model states in its
justification? So every quantity here is a within-model, paired BASE-FT
difference, never a standalone FT profile.

## What is reused unchanged

The statistical definitions are the frozen RQ2 ones (`semantic_final`), reused
verbatim:

* PRIMARY UNIT IS THE JUSTIFICATION. prevalence = share of a run's
  justifications in which the category appears at least once, counted once per
  justification however many sentences carry it;
* each run is computed independently, then the three stochastic runs are
  averaged into one value. Stochastic and greedy are never pooled - and the
  fine-tuned models were never run greedily, so only Stochastic exists here;
* the sentence-normalised metric is 100 * (assignments summed over the games)
  / (sentences summed over the same games) per run, then averaged - a ratio of
  sums, not a mean of per-justification rates. It is a SENSITIVITY CHECK, never
  a primary result;
* the bootstrap resamples GAMES with replacement, keeping a game's three
  stochastic runs together, and reuses ONE multiplicity vector across both
  conditions so the difference is paired.

`semantic_final.run_level_prevalence` and `density_sensitivity` could not
simply be called: they aggregate whatever rows they are handed, with no way to
express "the same 188 games on both sides", and `presence_tensor` asserts a
hole-free model x run x game grid across all three models at once, which a
per-model matched subset is not. The arithmetic below is the same; only the
subsetting and the pairing differ. `test_matched_semantic_equivalence` pins
that claim by running both on the one input where both are valid - the full
191-game BASE stochastic set.

## Why matched games

Three fine-tuned generations are unusable, and they are not evenly spread:

* E2B - one generation was deleted outright, so its game has 2 runs, not 3;
* 31B - three generations survive as rows with EMPTY justification text. They
  were never sent to the annotator, so they carry no labels at all; left in,
  they would count as justifications with zero categories present and quietly
  depress that model's fine-tuned prevalence.

A game is retained for a model only if BOTH conditions have all three
stochastic runs AND no justification is empty in either condition. The
excluded games are dropped from BOTH conditions, so the BASE side of every
contrast is recomputed on exactly the games the FT side has. Comparing a
fine-tuned prevalence on 188 games against the frozen BASE value on 191 would
confound the change with the subset.

That rule is not restated here. It is imported from `discourse_ft_contrast`,
so the RQ3 semantic contrast and the RQ3 discourse contrast are guaranteed to
speak about the same games; two copies of an exclusion rule is exactly how the
two halves of RQ3 would silently stop being comparable.

All EIGHT categories are carried, `Other` included. RQ2 dropped `Other` from
its bootstrap because it is a 0.3% residual and added noise to a between-model
comparison. Here it is kept visible: if familiarisation pushed content out of
the taxonomy, that has to be able to show up rather than being absorbed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.justification_analysis.comparison.discourse_ft_contrast import (
    excluded_games, matched_games,
)
from src.justification_analysis.pipeline.config import AnalysisConfig, default_config
from src.justification_analysis.pipeline.corpus import load_corpus
from src.justification_analysis.semantic import semantic_final as sf

# Eight categories, in the frozen schema's own order. Taken from the schema
# rather than restated so the contrast cannot drift from the prompt that
# produced the labels.
CATEGORIES: List[str] = list(sf.ALL_CATEGORIES)

# Same seed and replicate count as the frozen RQ2 bootstrap.
BOOTSTRAP_SEED = sf.BOOTSTRAP_SEED
BOOTSTRAP_REPLICATES = sf.BOOTSTRAP_REPLICATES

BASE_STAGE = "base"
FT_STAGE = "ft"

# The primary RQ3 semantic outcome. Everything else is read in support of it.
PRIMARY_CATEGORY = "Mechanical"

STOCHASTIC = "Stochastic"


@dataclass
class Condition:
    """One condition's annotations, plus the corpus that defines its games."""
    stage: str
    config: AnalysisConfig
    corpus: pd.DataFrame
    justifications: pd.DataFrame
    sentences: pd.DataFrame
    labels: pd.DataFrame


def load_condition(stage: str, decoding: str = STOCHASTIC) -> Condition:
    """Load one stage's annotations, refusing another stage's.

    `require_semantic_inputs` (called inside `load_annotations`) raises if the
    stage's own annotation run is absent, so a missing fine-tuned run stops the
    analysis here rather than producing a plausible contrast against base.
    """
    config = default_config(stage=stage)
    data = sf.load_annotations(config=config)

    justifications = data["justifications"]
    justifications = justifications.loc[
        justifications["decoding_group"].astype(str).eq(decoding)
    ].copy()

    keep = set(justifications["justification_id"])
    sentences = data["sentences"].loc[
        data["sentences"]["justification_id"].isin(keep)].copy()
    labels = data["labels"].loc[
        data["labels"]["justification_id"].isin(keep)].copy()

    unknown = sorted(set(labels["category"].astype(str)) - set(CATEGORIES))
    assert not unknown, f"{stage}: labels outside the frozen schema: {unknown}"

    corpus = load_corpus(config)
    corpus = corpus.loc[corpus["decoding_group"].eq(decoding)].copy()

    return Condition(stage, config, corpus, justifications, sentences, labels)


def annotation_coverage(condition: Condition, model: str,
                        games: Sequence[str]) -> pd.DataFrame:
    """Corpus rows for these games that carry NO annotation, and why.

    The annotation input was built from the corpus, so a gap between the two is
    either an empty generation that was never sent or a shard that came back
    short. Both must be visible before any prevalence is quoted.
    """
    rows = condition.corpus.loc[
        condition.corpus["model"].eq(model)
        & condition.corpus["game_id"].isin(games)
    ]
    annotated = set(condition.justifications.loc[
        condition.justifications["model"].astype(str).eq(model),
        ["game_id", "run_label"]].itertuples(index=False, name=None))

    missing = []
    for row in rows.itertuples(index=False):
        if (row.game_id, row.run_label) not in annotated:
            missing.append({
                "stage": condition.stage,
                "model": model,
                "game_id": row.game_id,
                "run_label": row.run_label,
                "n_words": int(row.n_words),
                "reason": ("empty justification text"
                           if not str(row.justification).strip()
                           else "not present in the annotation output"),
            })
    return pd.DataFrame(missing)


# ---------------------------------------------------------------------------
# Per (condition, run, game) matrices - the basis for both the point estimates
# and the bootstrap, so the two can never diverge.
# ---------------------------------------------------------------------------

def build_matrices(condition: Condition, model: str,
                   games: Sequence[str]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Presence, assignment counts and sentences, shaped (n_runs, n_games).

    Aligned on the given game order so BASE and FT index the same games, which
    is what makes the resampling paired.

    `presence` is the 0/1 justification-level indicator that feeds the PRIMARY
    metric; `assignments` counts every label occurrence and feeds the
    sentence-normalised sensitivity check only.
    """
    frame = condition.justifications.loc[
        condition.justifications["model"].astype(str).eq(model)
        & condition.justifications["game_id"].isin(games)
    ]
    runs = sorted(frame["run_label"].unique())
    index = {game: i for i, game in enumerate(games)}

    shape = (len(runs), len(games))
    presence = {c: np.zeros(shape) for c in CATEGORIES}
    assignments = {c: np.zeros(shape) for c in CATEGORIES}
    sentences = np.zeros(shape)

    labels = condition.labels.loc[
        condition.labels["justification_id"].isin(frame["justification_id"])]
    by_category = labels.groupby(["justification_id", "category"],
                                 observed=True).size()

    for r, run in enumerate(runs):
        run_rows = frame.loc[frame["run_label"].eq(run)]
        assert len(run_rows) == len(games), (
            f"{condition.stage}/{model}/{run}: {len(run_rows)} annotated "
            f"justifications for {len(games)} matched games")
        for row in run_rows.itertuples(index=False):
            g = index[row.game_id]
            sentences[r, g] = row.n_sentences
            for category in CATEGORIES:
                presence[category][r, g] = float(getattr(row, f"has_{category}"))
                assignments[category][r, g] = by_category.get(
                    (row.justification_id, category), 0)

    assert sentences.sum() > 0, \
        f"{condition.stage}/{model}: no sentences in the matched set"
    return runs, {"presence": presence, "assignments": assignments,
                  "sentences": sentences}


def _weights(n_games: int, n_replicates: int, seed: int) -> np.ndarray:
    """Multinomial multiplicities over games - the frozen resampling scheme.

    One vector per replicate, applied to BOTH conditions and every category, so
    the difference is paired and the categories share a resample.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        n_games, np.full(n_games, 1 / n_games), size=n_replicates
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Justification-level prevalence - the PRIMARY outcome
# ---------------------------------------------------------------------------

def prevalence_contrast(base: Condition, ft: Condition, model: str,
                        games: Sequence[str],
                        n_replicates: int = BOOTSTRAP_REPLICATES,
                        seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """FT - BASE prevalence differences, in points, with paired 95% CIs.

    One replicate resamples the matched games with replacement and applies the
    SAME resampled games to both conditions. The pairing removes between-game
    variation from the difference: a game that invites mechanical reasoning
    invites it in both conditions, and resampling it moves both sides together.

    No p-values. The interval is a descriptive uncertainty statement about the
    difference, and `ci_excludes_zero` is recorded as a fact about the
    interval, not applied as a decision rule.
    """
    _, base_m = build_matrices(base, model, games)
    _, ft_m = build_matrices(ft, model, games)

    n_games = len(games)
    weights = _weights(n_games, n_replicates, seed)            # (B, G)

    rows = []
    for category in CATEGORIES:
        base_grid, ft_grid = base_m["presence"][category], ft_m["presence"][category]

        # Per run, then averaged across runs - the frozen aggregation order.
        base_point = float((base_grid.mean(axis=1)).mean())
        ft_point = float((ft_grid.mean(axis=1)).mean())

        base_boot = ((base_grid @ weights.T) / n_games).mean(axis=0)   # (B,)
        ft_boot = ((ft_grid @ weights.T) / n_games).mean(axis=0)
        differences = 100 * (ft_boot - base_boot)
        low, high = np.percentile(differences, [2.5, 97.5])

        rows.append({
            "model": model,
            "category": category,
            "is_substantive": category != sf.OTHER_CATEGORY,
            "n_games": n_games,
            "n_justifications_per_condition": n_games * base_grid.shape[0],
            "base_pct": 100 * base_point,
            "ft_pct": 100 * ft_point,
            "delta_pp": 100 * (ft_point - base_point),
            "ci_low": low,
            "ci_high": high,
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "n_replicates": n_replicates,
            "seed": seed,
        })
    return pd.DataFrame(rows)


def run_level_prevalence(condition: Condition, model: str,
                         games: Sequence[str]) -> pd.DataFrame:
    """Per-run prevalence on the matched games - the spread behind the mean.

    A three-run mean can hide a category that only one run ever produces, which
    for a rare category like Mechanical at E2B is the difference between a real
    shift and one lucky run.
    """
    runs, matrices = build_matrices(condition, model, games)
    rows = []
    for r, run in enumerate(runs):
        for category in CATEGORIES:
            present = matrices["presence"][category][r]
            rows.append({
                "stage": condition.stage,
                "model": model,
                "run_label": run,
                "category": category,
                "n_justifications": len(games),
                "n_present": int(present.sum()),
                "prevalence_pct": 100 * float(present.mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Sentence-normalised sensitivity - NEVER a primary result
# ---------------------------------------------------------------------------

def density_contrast(base: Condition, ft: Condition, model: str,
                     games: Sequence[str],
                     n_replicates: int = BOOTSTRAP_REPLICATES,
                     seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """FT - BASE category assignments per 100 sentences, paired 95% CIs.

    The metric the primary analysis deliberately does not use, computed so that
    "would normalising for justification length change the story?" is answered
    with numbers rather than assertion. A prevalence change that survives here
    is not a length artefact; one that disappears here was carried by the
    models writing more, or fewer, sentences after familiarisation.
    """
    _, base_m = build_matrices(base, model, games)
    _, ft_m = build_matrices(ft, model, games)

    n_games = len(games)
    weights = _weights(n_games, n_replicates, seed)
    base_sentences = base_m["sentences"] @ weights.T           # (R, B)
    ft_sentences = ft_m["sentences"] @ weights.T

    def point(matrices, category):
        per_run = (100 * matrices["assignments"][category].sum(axis=1)
                   / matrices["sentences"].sum(axis=1))
        return float(per_run.mean())

    rows = []
    for category in CATEGORIES:
        base_point, ft_point = point(base_m, category), point(ft_m, category)

        base_boot = (100 * (base_m["assignments"][category] @ weights.T)
                     / base_sentences).mean(axis=0)
        ft_boot = (100 * (ft_m["assignments"][category] @ weights.T)
                   / ft_sentences).mean(axis=0)
        differences = ft_boot - base_boot
        low, high = np.percentile(differences, [2.5, 97.5])

        rows.append({
            "model": model,
            "category": category,
            "is_substantive": category != sf.OTHER_CATEGORY,
            "n_games": n_games,
            "base_per_100_sentences": base_point,
            "ft_per_100_sentences": ft_point,
            "delta": ft_point - base_point,
            "ci_low": low,
            "ci_high": high,
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "n_replicates": n_replicates,
            "seed": seed,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------

def descriptive_profile(condition: Condition, model: str,
                        games: Sequence[str]) -> Dict[str, float]:
    """Per-run then averaged length and labelling volume, matched games only.

    Length is the precondition for reading the sensitivity check: if BASE and
    FT write justifications of the same length, prevalence and density can only
    disagree through where the labels fall, not how many sentences carry them.
    """
    runs, matrices = build_matrices(condition, model, games)
    sentences = matrices["sentences"]
    total_labels = sum(matrices["assignments"][c] for c in CATEGORIES)
    distinct = sum(matrices["presence"][c] for c in CATEGORIES
                   if c != sf.OTHER_CATEGORY)

    return {
        "stage": condition.stage,
        "model": model,
        "n_runs": len(runs),
        "n_justifications": int(sentences.size),
        "mean_sentences_per_justification": float(sentences.mean(axis=1).mean()),
        "mean_labels_per_justification": float(total_labels.mean(axis=1).mean()),
        "mean_distinct_categories": float(distinct.mean(axis=1).mean()),
        "labels_per_100_sentences": float(
            (100 * total_labels.sum(axis=1) / sentences.sum(axis=1)).mean()),
    }
