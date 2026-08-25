"""Grouped nested cross-validation for the RQ1 surrogate.

Outer CV groups by the exact unordered player-group composition (36 of them
across the 191 games), not by individual game -- a game-level split leaks
players (and their idiosyncratic phrasing / behaviour) between train and
test. Repeated 5-fold, seeds 0..4, one fold assignment per composition per
repeat, reused identically across every model / representation / block /
estimator so all comparisons share the same held-out games.

This does NOT give participant-level independence: 22 of the 60 players
appear in more than one composition (Phase 0 finding), so a player can still
appear in both an outer-train and an outer-test fold via a different game.
That limitation is inherent to the corpus and is not solved here.

Inner CV (3-fold, same grouping variable) is generated fresh inside each
outer-training partition at fit time (see rq1_model.py), since it must only
ever see that partition's compositions.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

N_OUTER_FOLDS = 5
OUTER_SEEDS = (0, 1, 2, 3, 4)
N_INNER_FOLDS = 3


def composition_of(roster):
    """{game key: frozenset(players)} -- the grouping variable."""
    return {k: frozenset(v["players"]) for k, v in roster.items()}


def _balanced_fold_map(compositions, games_per_comp, n_folds, seed):
    """Greedy first-fit assignment of compositions into folds by game count:
    each composition, in a seeded random order, joins whichever fold
    currently holds the fewest games.

    Deliberately *not* sorted by descending size first (that would make the
    placement of the handful of large compositions -- 18, 15, 12, 12, 11
    games -- essentially seed-independent, since a first-fit-decreasing pass
    on fixed sizes is nearly deterministic). Randomising the order instead
    means different repeats route the large compositions into different
    folds, while the greedy min-fill rule still keeps totals reasonably even.
    """
    rng = np.random.default_rng(seed)
    order = list(compositions)
    rng.shuffle(order)
    fold_totals = np.zeros(n_folds, dtype=int)
    comp_fold = {}
    for c in order:
        f = int(np.argmin(fold_totals))
        comp_fold[c] = f
        fold_totals[f] += games_per_comp[c]
    return comp_fold, fold_totals


def build_outer_folds(roster, n_folds=N_OUTER_FOLDS, seeds=OUTER_SEEDS):
    """Returns a long DataFrame: repeat, fold, key (game), composition_id.

    ``composition_id`` is a stable small integer, assigned by descending
    corpus-wide game count so it means the same thing across repeats.
    """
    comp = composition_of(roster)
    by_comp = defaultdict(list)
    for k, s in comp.items():
        by_comp[s].append(k)
    comps_sorted = sorted(by_comp, key=lambda s: (-len(by_comp[s]), sorted(s)))
    comp_id = {s: i for i, s in enumerate(comps_sorted)}
    games_per_comp = {s: len(v) for s, v in by_comp.items()}

    rows = []
    for rep, seed in enumerate(seeds):
        comp_fold, _ = _balanced_fold_map(comps_sorted, games_per_comp, n_folds, seed)
        for s, keys in by_comp.items():
            f = comp_fold[s]
            for k in keys:
                rows.append({"repeat": rep, "fold": f, "source": k[0], "session": k[1],
                            "game": k[2], "key": k, "composition_id": comp_id[s]})
    df = pd.DataFrame(rows)
    _assert_outer_folds(df, comp)
    return df


def _assert_outer_folds(df, comp):
    n_games = len(comp)
    for rep, g in df.groupby("repeat"):
        assert g["key"].nunique() == n_games, \
            f"repeat {rep}: {g['key'].nunique()} games covered, expected {n_games}"
        assert (g.groupby("key").size() == 1).all(), \
            f"repeat {rep}: a game appears in more than one fold"
        # no composition split across folds within this repeat
        cf = g.groupby("composition_id")["fold"].nunique()
        assert (cf == 1).all(), f"repeat {rep}: a composition spans multiple folds"
    assert df["composition_id"].nunique() == 36, \
        f"expected 36 compositions, found {df['composition_id'].nunique()}"
    assert df["key"].nunique() == 191, f"expected 191 games, found {df['key'].nunique()}"


def outer_splits(fold_df, repeat):
    """(fold -> test keys) for one repeat, and a key -> composition_id map."""
    g = fold_df[fold_df.repeat == repeat]
    comp_of_key = dict(zip(g["key"], g["composition_id"]))
    test_keys = {f: set(gg["key"]) for f, gg in g.groupby("fold")}
    return test_keys, comp_of_key


def inner_fold_map(train_keys, comp_of_key, n_folds=N_INNER_FOLDS, seed=0):
    """Grouped 3-fold split of an outer-training set, by the same composition
    variable, restricted to compositions present in ``train_keys``.

    ``train_keys`` is iterated in sorted order rather than as-given: it is
    typically a Python ``set`` of tuples, whose iteration order depends on
    the process's string-hash seed (randomised by default), which would
    otherwise make the seeded shuffle in :func:`_balanced_fold_map`
    non-reproducible across runs despite the fixed seed.
    """
    comps = defaultdict(list)
    for k in sorted(train_keys):
        comps[comp_of_key[k]].append(k)
    games_per_comp = {c: len(v) for c, v in comps.items()}
    comp_fold, _ = _balanced_fold_map(sorted(comps), games_per_comp, n_folds, seed)
    key_fold = {}
    for c, keys in comps.items():
        for k in keys:
            key_fold[k] = comp_fold[c]
    return key_fold
