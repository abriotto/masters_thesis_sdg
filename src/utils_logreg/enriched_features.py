"""Extending Lai's pairwise vote model with the resolved persuasion features.

Lai's design gives the voter and the candidate one vector each, holding that
player's distribution over the annotated techniques. This module keeps that
design and the same target -- did this voter vote for this candidate -- and
appends to each side the same acts *resolved* into their target and content:

* accusations **received** (werewolf-type, deception-type), which resolve
  "Accusation" into who it landed on;
* what the player claimed about their own role (Werewolf, an information role,
  how many distinct roles), which resolves "Identity Declaration" into what it
  said.

Nothing dyadic is added: the voter vector and the candidate vector stay two
independent descriptions of two players, exactly as in the paper.

The enriched columns are counts while the technique columns are proportions, so
the caller should standardise the design matrix (using training-split
statistics) before fitting a penalised model, or the penalty will fall on the
two groups very unevenly.
"""

from .lai2023_loading import get_game_id, get_outcome_record, get_session_key
from .strategy_features import (STRATEGIES, compute_player_strategy_dists,
                                one_hot_role)

# per-player features added to each side of the pair
ENRICHED_FEATURES = ["werewolf_count", "deception_count", "claims_werewolf",
                     "claims_info_role", "n_distinct_roles_claimed_self"]

# Lai's dataset keys vs the source folder names used by the choice-model loaders
DATASET_SOURCE = {"yt": "Youtube", "ego4d": "Ego4D"}


def build_player_feature_lookup(annot_root, acc_root, ic_csv):
    """``{(source, session, game, player): {feature: value}}`` for the enriched
    features, built with the same loaders the choice models use so both
    analyses see identical numbers."""
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from utils_choice.features import (load_accusation_features,
                                       load_identity_claim_features,
                                       load_technique_features)

    _, game_length, rec_speaker, _ = load_technique_features(annot_root)
    acc_counts, _ = load_accusation_features(acc_root, game_length)
    ic_feats = load_identity_claim_features(ic_csv, game_length, rec_speaker)

    lookup = {}
    for df, cols in [(acc_counts, ["werewolf_count", "deception_count"]),
                     (ic_feats, ["claims_werewolf", "claims_info_role",
                                 "n_distinct_roles_claimed_self"])]:
        for _, r in df.iterrows():
            entry = lookup.setdefault((r["key"], str(r["player"]).strip().lower()), {})
            for c in cols:
                entry[c] = float(r[c])
    return lookup


def _enriched_for(lookup, key, player):
    entry = lookup.get((key, str(player).strip().lower()), {})
    return [entry.get(f, 0.0) for f in ENRICHED_FEATURES]


def build_human_pairwise_rows_enriched(game, dataset, outcome_index, lookup,
                                       include_roles=False):
    """One row per (voter, candidate), label = 1 if the voter voted for them.

    Feature vector: voter techniques, voter enriched, candidate techniques,
    candidate enriched -- matching :func:`enriched_feature_names`.
    """
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from utils_choice.features import ckey

    outcome = get_outcome_record(game, dataset, outcome_index)
    players = outcome["playerNames"]
    start_roles = outcome.get("startRoles", [])
    voting_outcome = outcome["votingOutcome"]

    role_map = dict(zip(players, start_roles))
    dists = compute_player_strategy_dists(game["Dialogue"], players)
    key = ckey(DATASET_SOURCE[dataset], get_session_key(game, dataset),
               get_game_id(game))

    rows = []
    for i, voter in enumerate(players):
        for j, candidate in enumerate(players):
            feature = ([dists[voter][s] for s in STRATEGIES]
                       + _enriched_for(lookup, key, voter)
                       + [dists[candidate][s] for s in STRATEGIES]
                       + _enriched_for(lookup, key, candidate))
            if include_roles:
                feature += one_hot_role(role_map.get(voter))
            rows.append({"dataset": dataset,
                         "session_key": get_session_key(game, dataset),
                         "game_id": get_game_id(game),
                         "voter": voter, "candidate": candidate,
                         "label": 1 if voting_outcome[i] == j else 0,
                         "matched_enrichment": int((key, str(candidate).strip().lower())
                                                   in lookup),
                         "feature": feature})
    return rows


def enriched_feature_names(include_roles=False, role_list=None):
    names = ([f"voter_{s}" for s in STRATEGIES]
             + [f"voter_{f}" for f in ENRICHED_FEATURES]
             + [f"candidate_{s}" for s in STRATEGIES]
             + [f"candidate_{f}" for f in ENRICHED_FEATURES])
    if include_roles:
        from .strategy_features import ROLE_LIST
        names += [f"role_{r}" for r in (role_list or ROLE_LIST)]
    return names
