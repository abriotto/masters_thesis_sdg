\
from .lai2023_loading import get_session_key, get_game_id, get_outcome_record
from .strategy_features import (
    STRATEGIES,
    one_hot_role,
    compute_player_strategy_dists,
    compute_player_early_late_dists,
    candidate_aggregate_feature,
    candidate_temporal_feature,
    pairwise_aggregate_feature,
    pairwise_temporal_feature,
)


# ---------------------------------------------------------------------
# Human vote rows
# ---------------------------------------------------------------------

def build_human_pairwise_rows(game, dataset, outcome_index, include_roles=True):
    """
    Human binary target, pairwise features.

    One row per (voter, candidate) pair:
        label = 1 if the human voter voted for that candidate else 0

    Features:
        voter strategy distribution + candidate strategy distribution
        + optional voter start-role one-hot
    """
    outcome = get_outcome_record(game, dataset, outcome_index)

    players = outcome["playerNames"]
    start_roles = outcome.get("startRoles", [])
    voting_outcome = outcome["votingOutcome"]
    dialogue = game["Dialogue"]

    role_map = dict(zip(players, start_roles))
    dists = compute_player_strategy_dists(dialogue, players)

    rows = []

    for i, voter in enumerate(players):
        for j, candidate in enumerate(players):
            label = 1 if voting_outcome[i] == j else 0

            feature = pairwise_aggregate_feature(voter, candidate, dists)

            if include_roles:
                feature += one_hot_role(role_map.get(voter))

            rows.append({
                "dataset": dataset,
                "session_key": get_session_key(game, dataset),
                "game_id": get_game_id(game),
                "voter": voter,
                "candidate": candidate,
                "label": label,
                "feature": feature,
            })

    return rows


def build_human_candidate_only_rows(game, dataset, outcome_index):
    """
    Human binary target, candidate-only features.

    One row per (voter, candidate) pair:
        label = 1 if the human voter voted for that candidate else 0

    Features:
        candidate strategy distribution only

    This preserves the human binary vote-vs-not-vote setup while removing
    voter-side features.
    """
    outcome = get_outcome_record(game, dataset, outcome_index)

    players = outcome["playerNames"]
    voting_outcome = outcome["votingOutcome"]
    dialogue = game["Dialogue"]

    dists = compute_player_strategy_dists(dialogue, players)

    rows = []

    for i, voter in enumerate(players):
        for j, candidate in enumerate(players):
            label = 1 if voting_outcome[i] == j else 0

            rows.append({
                "dataset": dataset,
                "session_key": get_session_key(game, dataset),
                "game_id": get_game_id(game),
                "voter": voter,
                "candidate": candidate,
                "label": label,
                "feature": candidate_aggregate_feature(candidate, dists),
            })

    return rows


def build_human_temporal_pairwise_rows(game, dataset, outcome_index):
    """
    Human binary target with early/late voter and candidate features.
    """
    outcome = get_outcome_record(game, dataset, outcome_index)

    players = outcome["playerNames"]
    voting_outcome = outcome["votingOutcome"]
    dialogue = game["Dialogue"]

    early_dists, late_dists = compute_player_early_late_dists(dialogue, players)

    rows = []

    for i, voter in enumerate(players):
        for j, candidate in enumerate(players):
            label = 1 if voting_outcome[i] == j else 0

            rows.append({
                "dataset": dataset,
                "session_key": get_session_key(game, dataset),
                "game_id": get_game_id(game),
                "voter": voter,
                "candidate": candidate,
                "label": label,
                "feature": pairwise_temporal_feature(
                    voter, candidate, early_dists, late_dists
                ),
            })

    return rows


def build_human_temporal_candidate_only_rows(game, dataset, outcome_index):
    """
    Human binary target with early/late candidate-only features.
    """
    outcome = get_outcome_record(game, dataset, outcome_index)

    players = outcome["playerNames"]
    voting_outcome = outcome["votingOutcome"]
    dialogue = game["Dialogue"]

    early_dists, late_dists = compute_player_early_late_dists(dialogue, players)

    rows = []

    for i, voter in enumerate(players):
        for j, candidate in enumerate(players):
            label = 1 if voting_outcome[i] == j else 0

            rows.append({
                "dataset": dataset,
                "session_key": get_session_key(game, dataset),
                "game_id": get_game_id(game),
                "voter": voter,
                "candidate": candidate,
                "label": label,
                "feature": candidate_temporal_feature(
                    candidate, early_dists, late_dists
                ),
            })

    return rows


def build_human_split_rows(games, dataset, outcome_index, row_builder, **kwargs):
    all_rows = []
    for game in games:
        all_rows.extend(row_builder(game, dataset, outcome_index, **kwargs))
    return all_rows


def build_human_rows_by_split(annot_splits, outcome_index, row_builder, **kwargs):
    """
    Builds combined YouTube + Ego4D rows for train/val/test.
    """
    split_rows = {"train": [], "val": [], "test": []}

    for dataset, split_map in annot_splits.items():
        for split, games in split_map.items():
            split_rows[split].extend(
                build_human_split_rows(
                    games,
                    dataset,
                    outcome_index,
                    row_builder=row_builder,
                    **kwargs,
                )
            )

    return split_rows


# ---------------------------------------------------------------------
# LLM vote rows
# ---------------------------------------------------------------------

def build_llm_binary_rows_for_game(game, vote_row, feature_kind="candidate_aggregate"):
    """
    LLM binary vote-vs-not-vote target.

    One row per candidate in the game:
        label = 1 if candidate == LLM chosen_vote else 0
        label = 0 otherwise

    The LLM is an external observer, so there is no true human voter. For
    compatibility with human pairwise rows, the returned rows include:
        voter = "LLM"
    """
    dataset = vote_row["dataset"]
    session_key = vote_row["session_key"]
    game_id = vote_row["game_id"]
    chosen_vote = vote_row["chosen_vote"]
    players = list(vote_row["player_names"])
    dialogue = game["Dialogue"]

    if feature_kind == "candidate_aggregate":
        dists = compute_player_strategy_dists(dialogue, players)

        def make_feature(candidate):
            return candidate_aggregate_feature(candidate, dists)

    elif feature_kind == "candidate_temporal":
        early_dists, late_dists = compute_player_early_late_dists(dialogue, players)

        def make_feature(candidate):
            return candidate_temporal_feature(candidate, early_dists, late_dists)

    else:
        raise ValueError(
            "LLM binary rows support feature_kind='candidate_aggregate' "
            "or feature_kind='candidate_temporal'."
        )

    rows = []

    for candidate in players:
        rows.append({
            "dataset": dataset,
            "session_key": session_key,
            "game_id": game_id,
            "voter": "LLM",
            "candidate": candidate,
            "chosen_vote": chosen_vote,
            "label": 1 if candidate == chosen_vote else 0,
            "feature": make_feature(candidate),
            "result_path": vote_row.get("path"),
        })

    return rows


def build_llm_binary_rows(llm_votes_df, game_lookup, split_lookup, feature_kind="candidate_aggregate"):
    """
    Builds train/val/test LLM binary rows.

    Returns:
        split_rows, unmatched_df

    split_rows:
        {"train": rows, "val": rows, "test": rows}

    unmatched_df:
        LLM result files that could not be aligned with annotation split games.
    """
    import pandas as pd

    split_rows = {"train": [], "val": [], "test": []}
    unmatched = []

    for _, vote_row in llm_votes_df.iterrows():
        key = (
            vote_row["dataset"],
            vote_row["session_key"],
            vote_row["game_id"],
        )

        if key not in game_lookup:
            unmatched.append({
                "dataset": vote_row["dataset"],
                "session_key": vote_row["session_key"],
                "game_id": vote_row["game_id"],
                "path": vote_row.get("path"),
            })
            continue

        split = split_lookup[key]
        game = game_lookup[key]

        split_rows[split].extend(
            build_llm_binary_rows_for_game(
                game,
                vote_row,
                feature_kind=feature_kind,
            )
        )

    return split_rows, pd.DataFrame(unmatched)
