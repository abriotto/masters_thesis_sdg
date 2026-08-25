"""Thesis human-vote protocol: the ordered voter-candidate pairwise dataset.

This is the *human comparison* for RQ1, not a second surrogate pipeline. It
keeps Lai et al.'s binary outcome -- did human voter A vote for candidate B --
but replaces everything else with the machinery already validated for the LLM
surrogate:

* the same 191-game cleaned corpus and the same roster
  (:func:`utils_choice.io.load_vote_tables`);
* the same 13 persuasion features, built by
  :func:`utils_choice.rq1_features.build_feature_table` -- reused, never
  reimplemented, so speaker normalisation, the dialogue midpoint and the
  unresolved-name behaviour are identical by construction;
* the same per-100-turn rate normalisation
  (:func:`utils_choice.rq1_model.add_rate_columns`) and the same column
  selection (:func:`utils_choice.rq1_model.feature_columns`);
* the same 36-composition grouped fold assignments
  (:func:`utils_choice.rq1_cv.build_outer_folds`).

Ground-truth roles never enter: the LLM observer cannot see them, so the
thesis models cannot either. The Lai replication's role-inclusive model lives
in :mod:`utils_choice.human_lai` and is not used here.

Differences from the replication, all deliberate:

* self-pairs are dropped -- ``voter != candidate`` by construction;
* a voter is included only if their ``votingOutcome`` entry is a valid index
  of another roster player, so every included voter contributes exactly one
  positive among their ``n - 1`` rows. Abstainers (``"NA"``), the three
  moderators, two out-of-range entries and the 34 self-votes are excluded as
  voters and reported in the diagnostics;
* every roster player stays available as a *candidate*, including a moderator
  or an abstainer, because that is exactly the alternative set the LLM was
  shown for the same game.
"""

from collections import Counter

import numpy as np
import pandas as pd

from utils_choice.features import ckey
from utils_choice.rq1_features import ALL_FEATURES_13
from utils_choice.rq1_model import feature_columns
from utils_logreg.lai2023_loading import (get_game_id, get_outcome_record,
                                          get_session_key)

#: Lai's dataset keys vs the source folder names the RQ1 game key uses.
DATASET_SOURCE = {"yt": "Youtube", "ego4d": "Ego4D"}

#: The three thesis feature sets. Values are ``(representation, block)`` pairs
#: passed straight to :func:`utils_choice.rq1_model.feature_columns`, so the
#: human models can never drift from the LLM ones.
FEATURE_SETS = {
    "Strategy": ("rate_overall", "strategy"),
    "Strategy Temporal": ("rate_temporal", "strategy"),
    "Combined": ("rate_overall", "combined"),
}

VOTER_PREFIX, CANDIDATE_PREFIX = "voter", "candidate"


def _is_index(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def build_pair_rows(annot_splits, outcome_index, roster):
    """Ordered (voter, candidate) rows for the whole 191-game corpus.

    Returns ``(pairs, excluded_voters)``. ``pairs`` has one row per ordered
    pair of distinct roster players for every included voter, with columns
    ``key, source, session, game, voter, candidate, label, n_players,
    lai_split``. ``lai_split`` is carried only as provenance; the thesis
    protocol pools all three splits and re-partitions by composition.
    """
    pair_rows, excluded = [], []
    for dataset, by_split in annot_splits.items():
        for split, games in by_split.items():
            for game in games:
                key = ckey(DATASET_SOURCE[dataset], get_session_key(game, dataset),
                           get_game_id(game))
                if key not in roster:
                    continue
                outcome = get_outcome_record(game, dataset, outcome_index)
                names = list(outcome["playerNames"])
                start_roles = list(outcome.get("startRoles", []))
                votes = list(outcome["votingOutcome"])
                assert set(names) == set(roster[key]["players"]), \
                    f"roster mismatch for {key}"
                n = len(names)
                for i, voter in enumerate(names):
                    v = votes[i] if i < len(votes) else None
                    role = start_roles[i] if i < len(start_roles) else None
                    if role == "Moderator":
                        reason = "moderator"
                    elif not _is_index(v):
                        reason = "no_vote_recorded"
                    elif not 0 <= v < n:
                        reason = "vote_index_out_of_range"
                    elif v == i:
                        reason = "self_vote"
                    else:
                        reason = None
                    if reason is not None:
                        excluded.append({"key": key, "source": key[0],
                                         "session": key[1], "game": key[2],
                                         "voter": voter, "raw_vote": str(v),
                                         "start_role": role, "reason": reason})
                        continue
                    chosen = names[v]
                    for j, candidate in enumerate(names):
                        if j == i:
                            continue
                        pair_rows.append({
                            "key": key, "source": key[0], "session": key[1],
                            "game": key[2], "voter": voter, "candidate": candidate,
                            "label": int(candidate == chosen), "n_players": n,
                            "lai_split": split})
    pairs = pd.DataFrame(pair_rows)
    pairs = pairs.sort_values(["key", "voter", "candidate"]).reset_index(drop=True)
    return pairs, pd.DataFrame(excluded)


def assert_pairwise_structure(pairs, roster):
    """Every structural guarantee the thesis protocol claims."""
    assert (pairs["voter"] != pairs["candidate"]).all(), "self-pair leaked in"
    per_voter = pairs.groupby(["key", "voter"]).agg(
        n_rows=("label", "size"), n_pos=("label", "sum"),
        n_players=("n_players", "first"))
    assert (per_voter["n_rows"] == per_voter["n_players"] - 1).all(), \
        "a voter does not have exactly n-1 candidate rows"
    assert (per_voter["n_pos"] == 1).all(), \
        "a voter does not have exactly one positive label"
    rosters = {k: set(v["players"]) for k, v in roster.items()}
    for key, g in pairs.groupby("key"):
        assert set(g["voter"]) <= rosters[key], f"off-roster voter in {key}"
        assert set(g["candidate"]) <= rosters[key], f"off-roster candidate in {key}"
    chosen = pairs[pairs["label"] == 1]
    assert len(chosen) == len(per_voter), "positives do not match voter count"
    return {"pair_rows": len(pairs), "positives": int(pairs["label"].sum()),
            "prevalence": float(pairs["label"].mean()),
            "voters": int(len(per_voter)), "games": int(pairs["key"].nunique())}


def player_vectors(feat_df, cols):
    """``(index, matrix)`` of one feature vector per (game key, player).

    ``index`` is a ``{(key, player): row}`` map into ``matrix``. Building the
    player matrix once, before the pair design, is what lets the scaler be fit
    on unique player-games rather than on duplicated pair rows.
    """
    sub = feat_df[["key", "player"] + list(cols)].reset_index(drop=True)
    index = {(k, p): i for i, (k, p) in enumerate(zip(sub["key"], sub["player"]))}
    assert len(index) == len(sub), "duplicate (key, player) in the feature table"
    return index, sub[list(cols)].to_numpy(dtype=float)


def pair_player_indices(pairs, index):
    """Row indices into the player matrix for the voter and candidate sides."""
    voter_ix = np.fromiter((index[(k, p)] for k, p in
                            zip(pairs["key"], pairs["voter"])), int, len(pairs))
    cand_ix = np.fromiter((index[(k, p)] for k, p in
                           zip(pairs["key"], pairs["candidate"])), int, len(pairs))
    return voter_ix, cand_ix


def pair_feature_names(cols):
    """``voter_*`` then ``candidate_*``, in the order the design matrix uses."""
    return ([f"{VOTER_PREFIX}_{c}" for c in cols]
            + [f"{CANDIDATE_PREFIX}_{c}" for c in cols])


def feature_set_columns(name):
    representation, block = FEATURE_SETS[name]
    return feature_columns(representation, block)


def rate_column_base(col):
    """The underlying 13-feature name behind a rate column, for labelling."""
    base = col
    for suffix in ("_early_rate", "_late_rate", "_rate"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def unresolved_name_summary(diagnostics):
    """Counts of every name the RQ1 feature builder could not place."""
    unresolved = diagnostics["unresolved_names"]
    if unresolved.empty:
        return pd.DataFrame(columns=["source", "raw_name", "n"])
    return (unresolved.groupby(["source", "raw_name"]).size()
            .reset_index(name="n").sort_values("n", ascending=False))


def assert_matches_rq1_support(feat_df, rq1_support_csv):
    """Every one of the 13 feature totals must equal the saved RQ1 table."""
    ref = pd.read_csv(rq1_support_csv).set_index("feature")
    rows = []
    for feat in ALL_FEATURES_13:
        ours = float(feat_df[feat].sum())
        theirs = float(ref.loc[feat, "total"])
        assert ours == theirs, f"{feat}: human {ours} vs RQ1 {theirs}"
        assert int((feat_df[feat] > 0).sum()) == int(ref.loc[feat, "player_games_gt0"]), \
            f"{feat}: player-game support differs from RQ1"
        rows.append({"feature": feat, "total": ours,
                     "rq1_total": theirs, "matches_rq1": True})
    return pd.DataFrame(rows)


def conservation_report(feat_df):
    """Overall == early + late for every feature, and the turn denominators."""
    rows = []
    for feat in ALL_FEATURES_13 + ["turns"]:
        early, late = feat_df[f"{feat}_early"], feat_df[f"{feat}_late"]
        ok = bool(np.allclose(early + late, feat_df[feat]))
        assert ok, f"early + late != overall for {feat}"
        rows.append({"feature": feat, "total": float(feat_df[feat].sum()),
                     "early": float(early.sum()), "late": float(late.sum()),
                     "conserved": ok})
    return pd.DataFrame(rows)


def corpus_counts(roster):
    """The three corpus integrity numbers the thesis states."""
    players = Counter()
    compositions = set()
    for info in roster.values():
        compositions.add(frozenset(info["players"]))
        players.update(info["players"])
    return {"games": len(roster), "unique_players": len(players),
            "unique_compositions": len(compositions)}
