"""
Who gets accused, by the target's actual role.

Joins the enriched accusation annotations to the ground-truth role assignments
released with the corpus (vote_outcome_*), and reports how often a player is the
target of a Werewolf attribution or a deception accusation, broken down by the
player's starting and ending role.

The question this answers is methodological rather than descriptive: if actual
Werewolves are accused substantially more than other players, then any feature
derived from accusations carries information about a player's hidden role, and a
correlation involving such a feature cannot be read as purely behavioural.

Unit of exposure is the *player-game*: one row per player per game, since being
accused depends on the game rather than on how much the player speaks. Rates are
therefore accusations received per player-game, not per turn.

The Moderator is excluded: it is a table role rather than a game role, and the
moderator does not participate as an accusable player.

The Werewolf-versus-rest contrast is estimated with a bootstrap clustered on
games, matching the approach used for the strategy contrasts: whole games are
resampled with replacement, so that the interval reflects variation between
games rather than between individual player-games.

Usage:
    python src/pt_annotation/accusation_target_by_role.py
    python src/pt_annotation/accusation_target_by_role.py --latex
    python src/pt_annotation/accusation_target_by_role.py --role-frame end
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE = Path(r"C:\Users\annab\Documents\GitHub\masters_thesis_sdg")

ACC_ROOT = BASE / "data" / "processed" / "lai2023" / "accusation_transcripts" / "acc_targets"
EGO4D_TRUTH = BASE / "data" / "raw" / "lai2023" / "Ego4D" / "vote_outcome_ego4d"
YOUTUBE_TRUTH = BASE / "data" / "raw" / "lai2023" / "Youtube" / "vote_outcome_youtube_released"

UNKNOWN_TARGET = "UNKNOWN"

# Moderator is a table role, not a game role; NA marks a player-game with no
# recorded assignment. Neither is an accusable player role.
EXCLUDED_ROLES = {"moderator", "na", ""}

# Spelling variants in the released ground truth.
ROLE_ALIASES = {"doppleganger": "Doppelganger"}

# Player names spelled differently in the transcripts than in the released
# rosters. Keyed by (session, transcript spelling) -> roster spelling.
#
# Not included, deliberately: the two games of the "Drinking Play Through"
# session, whose roster holds *two* players called Jordan (Jordan1, Jordan2)
# while the transcript says only "Jordan". Those 22 accusations are genuinely
# ambiguous as to which player was meant and no name mapping can resolve them;
# they are left unmatched and excluded from the role join.
NAME_ALIASES = {
    "danieal": "Daniel",
}

SUBTYPES = ("werewolf", "deception")

# Roles below this many player-game instances are omitted from the table, as in
# the strategy-by-role tables.
MIN_INSTANCES = 20

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


# ============================================================
# Ground truth
# ============================================================

def load_truth():
    """{(source, session, game): {player: {'start': role, 'end': role}}}"""
    truth = {}
    for source, root in (("Ego4D", EGO4D_TRUTH), ("Youtube", YOUTUBE_TRUTH)):
        if not root.is_dir():
            print(f"  warning: ground-truth folder missing: {root}")
            continue
        for path in sorted(root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            # our session ids replace spaces with '#'
            session = path.stem.replace(" ", "#")
            for game, payload in record.items():
                names = payload.get("playerNames") or []
                starts = payload.get("startRoles") or []
                ends = payload.get("endRoles") or []
                if not (len(names) == len(starts) == len(ends)):
                    continue
                truth[(source, session, game)] = {
                    name: {"start": start, "end": end}
                    for name, start, end in zip(names, starts, ends)
                }
    return truth


# ============================================================
# Accusations received
# ============================================================

def load_accusations():
    """({(source, session, game): Counter[(target, subtype)]}, {annotated game keys})

    The second value matters: a game that was annotated but yielded no accusation
    is a genuine zero, whereas a game that was never annotated must be excluded
    from the denominator entirely rather than counted as a game in which nobody
    was accused.
    """
    received = defaultdict(Counter)
    annotated = set()
    for path in sorted(ACC_ROOT.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = record.get("metadata", {})
        key = (meta.get("source"), meta.get("session"), meta.get("game"))
        annotated.add(key)
        for item in record.get("items", []):
            for relation in item.get("relations") or []:
                accused = relation.get("accused") or []
                if UNKNOWN_TARGET in accused:
                    continue
                subtype = relation.get("type")
                for target in accused:
                    received[key][(target, subtype)] += 1
    return received, annotated


# ============================================================
# Join
# ============================================================

def normalise_name(name):
    """Match key for a player name, applying known transcript/roster variants."""
    key = (name or "").strip().casefold()
    return NAME_ALIASES.get(key, key).casefold()


def normalise_role(role):
    key = (role or "").strip().casefold()
    if key in EXCLUDED_ROLES:
        return None
    return ROLE_ALIASES.get(key, (role or "").strip())


def build_rows(truth, received, annotated, role_frame):
    """One row per player-game: (game_key, role, {subtype: n_received}).

    Restricted to games that were both annotated for accusations and have a
    ground-truth role assignment.
    """
    rows = []
    unmatched_targets = Counter()
    games_without_truth = set()
    games_joined = set()

    for key in annotated:
        if key not in truth:
            games_without_truth.add(key)
            continue
        known = {normalise_name(name) for name in truth[key]}
        for (target, _subtype), n in received.get(key, Counter()).items():
            if normalise_name(target) not in known:
                unmatched_targets[(key, target)] += n

    for key in sorted(annotated & set(truth)):
        counts = received.get(key, Counter())
        lowered = Counter()
        for (target, subtype), n in counts.items():
            lowered[(normalise_name(target), subtype)] += n
        for name, roles in truth[key].items():
            role = normalise_role(roles.get(role_frame))
            if role is None:
                continue
            per_subtype = {
                subtype: lowered.get((normalise_name(name), subtype), 0)
                for subtype in SUBTYPES
            }
            rows.append((key, role, per_subtype))
            games_joined.add(key)

    return rows, unmatched_targets, games_without_truth, games_joined


# ============================================================
# Aggregation and bootstrap
# ============================================================

def summarise(rows):
    per_role = defaultdict(lambda: {"n": 0, **{s: 0 for s in SUBTYPES},
                                    **{f"any_{s}": 0 for s in SUBTYPES}})
    for _key, role, counts in rows:
        bucket = per_role[role]
        bucket["n"] += 1
        for subtype in SUBTYPES:
            bucket[subtype] += counts[subtype]
            if counts[subtype] > 0:
                bucket[f"any_{subtype}"] += 1
    return per_role


def werewolf_contrast(rows, subtype, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """Bootstrap the (Werewolf - others) difference in accusations per player-game,
    resampling whole games with replacement."""
    by_game = defaultdict(list)
    for key, role, counts in rows:
        by_game[key].append((role, counts[subtype]))
    games = list(by_game)

    def diff(sample_games):
        ww_n = ww_sum = other_n = other_sum = 0
        for g in sample_games:
            for role, n in by_game[g]:
                if role.casefold() == "werewolf":
                    ww_n += 1
                    ww_sum += n
                else:
                    other_n += 1
                    other_sum += n
        if not ww_n or not other_n:
            return None
        return ww_sum / ww_n - other_sum / other_n

    point = diff(games)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        resampled = [rng.choice(games) for _ in games]
        d = diff(resampled)
        if d is not None:
            draws.append(d)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return point, lo, hi


# ============================================================
# Reporting
# ============================================================

def print_summary(rows, per_role, unmatched, no_truth, joined, role_frame):
    print(f"role frame                 : {role_frame}ing role")
    print(f"player-games joined        : {len(rows)} across {len(joined)} games")
    if no_truth:
        print(f"games with no ground truth : {len(no_truth)}")
        for key in sorted(no_truth)[:5]:
            print(f"    {key}")
    if unmatched:
        total = sum(unmatched.values())
        print(f"accusations at unmatched names: {total} "
              f"({len(unmatched)} distinct name/game pairs)")
        for (key, name), n in unmatched.most_common(5):
            print(f"    {name!r} in {key[1]}/{key[2]} ({n})")
    print()

    header = (f"{'Role':<14}{'N':>6}{'WW acc':>9}{'per PG':>9}{'% any':>8}"
              f"{'Dec acc':>10}{'per PG':>9}{'% any':>8}")
    print(header)
    print("-" * len(header))
    ordered = sorted(per_role.items(), key=lambda kv: kv[1]["n"], reverse=True)
    for role, b in ordered:
        if b["n"] < MIN_INSTANCES:
            continue
        print(f"{role:<14}{b['n']:>6}{b['werewolf']:>9}{b['werewolf']/b['n']:>9.2f}"
              f"{100*b['any_werewolf']/b['n']:>7.0f}%"
              f"{b['deception']:>10}{b['deception']/b['n']:>9.2f}"
              f"{100*b['any_deception']/b['n']:>7.0f}%")
    omitted = [(r, b['n']) for r, b in ordered if b["n"] < MIN_INSTANCES]
    if omitted:
        print(f"\nomitted (<{MIN_INSTANCES} player-games): "
              + ", ".join(f"{r} ({n})" for r, n in omitted))

    print()
    print("Werewolf vs. all other roles, accusations received per player-game")
    print("(game-clustered bootstrap, 95% percentile interval)")
    for subtype in SUBTYPES:
        point, lo, hi = werewolf_contrast(rows, subtype)
        flag = "" if lo <= 0 <= hi else "   <-- interval excludes zero"
        print(f"  {subtype:<10} {point:+.2f}   [{lo:+.2f}, {hi:+.2f}]{flag}")


def print_latex(per_role, role_frame):
    print(r"\begin{tabular}{lrrrrr}")
    print(r"  \toprule")
    print(r"  \textbf{%s role} & \textbf{N} & \textbf{WW} & \textbf{per PG} & "
          r"\textbf{Dec.} & \textbf{per PG} \\" % role_frame.capitalize())
    print(r"  \midrule")
    ordered = sorted(per_role.items(), key=lambda kv: kv[1]["n"], reverse=True)
    for role, b in ordered:
        if b["n"] < MIN_INSTANCES:
            continue
        print(f"  {role:<14} & {b['n']:>4} & {b['werewolf']:>4} & {b['werewolf']/b['n']:>5.2f} "
              f"& {b['deception']:>4} & {b['deception']/b['n']:>5.2f} \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description="Accusations received, by target's actual role.")
    parser.add_argument("--role-frame", choices=("start", "end"), default="start")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    truth = load_truth()
    received, annotated = load_accusations()
    rows, unmatched, no_truth, joined = build_rows(truth, received, annotated, args.role_frame)
    per_role = summarise(rows)

    if args.latex:
        print_latex(per_role, args.role_frame)
    else:
        print_summary(rows, per_role, unmatched, no_truth, joined, args.role_frame)


if __name__ == "__main__":
    main()
