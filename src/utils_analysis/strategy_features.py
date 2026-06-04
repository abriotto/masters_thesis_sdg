from collections import Counter, defaultdict
import numpy as np


STRATEGIES = [
    "No Strategy",
    "Identity Declaration",
    "Accusation",
    "Interrogation",
    "Call for Action",
    "Defense",
    "Evidence",
]

ROLE_LIST = [
    "Villager",
    "Werewolf",
    "Seer",
    "Robber",
    "Troublemaker",
    "Tanner",
    "Drunk",
    "Hunter",
    "Mason",
    "Insomniac",
    "Minion",
    "Doppelganger",
]


def one_hot_role(role, role_list=ROLE_LIST):
    return [1 if role == r else 0 for r in role_list]


def compute_player_strategy_counts(dialogue, players, strategies=STRATEGIES):
    """
    Counts annotated persuasion/communication strategies for each player.
    """
    counts = {player: Counter({s: 0 for s in strategies}) for player in players}

    for rec in dialogue:
        speaker = rec.get("speaker")
        if speaker not in counts:
            continue

        for label in rec.get("annotation", []):
            if label in strategies:
                counts[speaker][label] += 1

    return counts


def normalize_strategy_counts(counts, strategies=STRATEGIES):
    """
    Converts player strategy counts into per-player distributions.
    Players with no counted strategy receive all-zero distributions.
    """
    dists = {}

    for player, cnt in counts.items():
        total = sum(cnt[s] for s in strategies)

        if total == 0:
            dists[player] = {s: 0.0 for s in strategies}
        else:
            dists[player] = {s: cnt[s] / total for s in strategies}

    return dists


def compute_player_strategy_dists(dialogue, players, strategies=STRATEGIES):
    counts = compute_player_strategy_counts(dialogue, players, strategies=strategies)
    return normalize_strategy_counts(counts, strategies=strategies)


def compute_player_early_late_dists(dialogue, players, strategies=STRATEGIES):
    """
    Splits each player's own utterances into early/late halves, then computes
    normalized strategy distributions for each half.
    """
    player_utts = defaultdict(list)

    for rec in dialogue:
        speaker = rec.get("speaker")
        if speaker in players:
            player_utts[speaker].append(rec)

    def zero_dist():
        return {s: 0.0 for s in strategies}

    def normalize_counter(counter):
        total = sum(counter[s] for s in strategies)
        if total == 0:
            return zero_dist()
        return {s: counter[s] / total for s in strategies}

    early_dists = {}
    late_dists = {}

    for player in players:
        utts = player_utts[player]
        split_idx = len(utts) // 2

        early_counter = Counter({s: 0 for s in strategies})
        late_counter = Counter({s: 0 for s in strategies})

        for rec in utts[:split_idx]:
            for label in rec.get("annotation", []):
                if label in strategies:
                    early_counter[label] += 1

        for rec in utts[split_idx:]:
            for label in rec.get("annotation", []):
                if label in strategies:
                    late_counter[label] += 1

        early_dists[player] = normalize_counter(early_counter)
        late_dists[player] = normalize_counter(late_counter)

    return early_dists, late_dists


def candidate_aggregate_feature(candidate, dists, strategies=STRATEGIES):
    return [dists[candidate][s] for s in strategies]


def candidate_temporal_feature(candidate, early_dists, late_dists, strategies=STRATEGIES):
    return (
        [early_dists[candidate][s] for s in strategies]
        + [late_dists[candidate][s] for s in strategies]
    )


def pairwise_aggregate_feature(voter, candidate, dists, strategies=STRATEGIES):
    return (
        [dists[voter][s] for s in strategies]
        + [dists[candidate][s] for s in strategies]
    )


def pairwise_temporal_feature(voter, candidate, early_dists, late_dists, strategies=STRATEGIES):
    return (
        [early_dists[voter][s] for s in strategies]
        + [late_dists[voter][s] for s in strategies]
        + [early_dists[candidate][s] for s in strategies]
        + [late_dists[candidate][s] for s in strategies]
    )


def rows_to_xy(rows):
    X = np.array([r["feature"] for r in rows], dtype=np.float32)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    return X, y


def get_feature_names(kind, strategies=STRATEGIES, role_list=ROLE_LIST, include_roles=False):
    """
    kind:
      - "candidate_aggregate"
      - "candidate_temporal"
      - "pairwise_aggregate"
      - "pairwise_temporal"
    """
    if kind == "candidate_aggregate":
        names = [f"candidate_{s}" for s in strategies]

    elif kind == "candidate_temporal":
        names = (
            [f"candidate_early_{s}" for s in strategies]
            + [f"candidate_late_{s}" for s in strategies]
        )

    elif kind == "pairwise_aggregate":
        names = (
            [f"voter_{s}" for s in strategies]
            + [f"candidate_{s}" for s in strategies]
        )

    elif kind == "pairwise_temporal":
        names = (
            [f"voter_early_{s}" for s in strategies]
            + [f"voter_late_{s}" for s in strategies]
            + [f"candidate_early_{s}" for s in strategies]
            + [f"candidate_late_{s}" for s in strategies]
        )

    else:
        raise ValueError(f"Unknown feature-name kind: {kind}")

    if include_roles:
        names += [f"role_{r}" for r in role_list]

    return names
