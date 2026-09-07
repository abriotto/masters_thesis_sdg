"""Shared vote-statistic definitions for the RQ3 voting arms.

Notebook 2 and :mod:`src.voting.ft_vote_contrast` need the same bootstrap,
agreement and arm-resolution logic. They used to carry private copies of it,
which is how two arms drift apart without anyone noticing: the finetuned arm
ran at 5,000 replicates through a script while the base arm ran at 5,000
through a notebook, and nothing in the repo tied the two numbers together.
The definitions now live here once and both importers are thin.

`N_BOOT` is the design chapter's declared default for accuracy and for the
base-vs-finetuned contrasts. The surrogate model (notebook 3) deliberately
keeps its own, smaller budgets in `utils_choice.rq1_pipeline`, because those
bootstrap a cross-validated loss rather than a sample mean.
"""
from __future__ import annotations

from itertools import combinations
import json

import numpy as np

N_BOOT = 10_000
BOOT_SEED = 0
LIFT_BOOT_SEED = 20260822
SWAP_BOOT_SEED = 20260823

CIRCLE_VOTE_STRING = "No Werewolf"

# One results/voting directory per (arm, model). The arm name is also the
# analysis stage directory notebook 1 writes to, so every downstream consumer
# selects an arm with a single string.
ARM_RESULTS_DIRS: dict[str, dict[str, str]] = {
    "base": {
        "2B": "unsloth_gemma-4-E2B-it-unsloth-bnb-4bit",
        "4B": "unsloth_gemma-4-E4B-it-unsloth-bnb-4bit",
        "31B": "unsloth_gemma-4-31B-it-unsloth-bnb-4bit",
    },
    "ft": {
        "2B": "unsloth_gemma-4-E2B-it-unsloth-bnb-4bit__ft_gemma-4-E2B-role-inference-traced-final_adapter",
        "4B": "unsloth_gemma-4-E4B-it-unsloth-bnb-4bit__ft_gemma-4-E4B-role-inference-traced-final_adapter",
        "31B": "unsloth_gemma-4-31B-it-unsloth-bnb-4bit__ft_gemma-4-31B-role-inference-traced-final_adapter",
    },
    "derivation": {
        "2B": "unsloth_gemma-4-E2B-it-unsloth-bnb-4bit__ft_gemma-4-E2B-derivation-v1-checkpoint-23",
        "4B": "unsloth_gemma-4-E4B-it-unsloth-bnb-4bit__ft_gemma-4-E4B-derivation-v1-final_adapter",
    },
}

# Only the base arm was run at T=0; the finetuned arms have stochastic runs only.
ARM_HAS_GREEDY = {"base": True, "ft": False, "derivation": False}

# Games dropped because some run of some model in the arm produced no usable
# vote. The set is derived from the data (see `excluded_game_ids`); these are
# the expected sizes, asserted so a silent growth cannot pass unnoticed.
ARM_EXPECTED_EXCLUSIONS = {"base": 0, "ft": 4, "derivation": 2}
# ft = 4, not the 1 the superseded ft_vote_contrast.py excluded by hand: it
# dropped only E2B's missing result file and silently kept three 31B games
# whose run_3 (or run_2) failed to parse, scoring them on 2 draws instead
# of 3. Applying the paired rule to every model in the arm removes all four.

N_GAMES_TOTAL = 191

VALID_STATUSES = ("player_vote", "circle_vote")


def arm_models(arm: str) -> list[str]:
    """Model labels present in an arm, in the canonical thesis order."""
    present = ARM_RESULTS_DIRS[arm]
    return [m for m in ("2B", "4B", "31B") if m in present]


def n_games(arm: str) -> int:
    return N_GAMES_TOTAL - ARM_EXPECTED_EXCLUSIONS[arm]


def bootstrap_mean_ci(values, n_boot: int = N_BOOT, alpha: float = 0.05,
                      seed: int = BOOT_SEED):
    """95% percentile bootstrap CI for a mean, resampling games.

    This is the *only* inference tool used in the vote comparison. The game is
    the sampling unit throughout. Where two models, or a model and its
    finetuned counterpart, are compared they are scored on the same games, so
    the per-game difference is resampled (a paired bootstrap): game difficulty
    cancels and the CI is about the models, not about which games happened to
    fall in the corpus. A difference is quotable only when its CI excludes 0.

    Each call seeds its own generator, so a reported CI depends only on the
    values passed in -- never on how many bootstraps ran before it. Adding,
    removing or reordering an analysis therefore cannot shift the CIs of the
    others. (Two calls on samples of equal length share the same resampling
    index matrix; each CI is still valid on its own, but their bootstrap noise
    is correlated, so do not read small differences *between* CIs as
    independent evidence.)
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def within_model_agreement(labels) -> float:
    """P(two DISTINCT runs of one model agree). Self-pairs i = j excluded."""
    pairs = list(combinations(range(len(labels)), 2))
    if not pairs:
        return np.nan
    return sum(labels[i] == labels[j] for i, j in pairs) / len(pairs)


def between_model_agreement(labels_a, labels_b) -> float:
    """P(one run of A and one run of B agree), over all n_A x n_B pairs."""
    if not labels_a or not labels_b:
        return np.nan
    return sum(a == b for a in labels_a for b in labels_b) / (len(labels_a) * len(labels_b))


def parse_json(value, default):
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def excluded_game_ids(votes, arm: str, n_models: int | None = None) -> set:
    """Games an arm drops, derived from the arm's own file-level vote table.

    A game is dropped from EVERY model in the arm as soon as one model's run
    failed to produce a usable vote for it, so each arm stays a paired
    comparison on one game set. Notebook 1 emits a row per (run, game), so a
    game whose result file is missing entirely shows up as a short group
    rather than a bad status; both cases are caught here.
    """
    # `n_models` lets a caller pass a subset of the arm's models -- the
    # base-vs-arm contrast reads the BASE tables for only the models the
    # finetuned arm has, so the row count per game must follow the frame it was
    # actually given, not the full registry.
    n_models = len(arm_models(arm)) if n_models is None else n_models
    expected_rows = n_models * (3 + (1 if ARM_HAS_GREEDY[arm] else 0))
    bad = votes[~votes["status"].isin(VALID_STATUSES)]
    dropped = set(bad["game_id"])
    per_game = votes.groupby("game_id").size()
    dropped |= set(per_game[per_game < expected_rows].index)
    n_expected = ARM_EXPECTED_EXCLUSIONS[arm]
    if len(dropped) != n_expected:
        raise AssertionError(
            f"arm {arm!r}: expected {n_expected} excluded game(s), found "
            f"{len(dropped)}: {sorted(dropped)}")
    return dropped
