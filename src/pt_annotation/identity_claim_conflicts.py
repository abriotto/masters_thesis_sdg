"""
Detects two kinds of identity-claim inconsistency within each ONUW game,
from the identity-claim annotation output (ic_targets/*.json):

1. Same-role collision: two or more DIFFERENT speakers claiming the same
   role at some point in the same game. Purely structural -- no attempt to
   reason about legitimate role swaps (Robber/Troublemaker/Drunk) or which
   claim came "later" -- intentionally kept simple, matching the
   identity-claim annotation prompt's own simplicity choice.

2. Self-contradiction: a single player's OWN claims across the game are
   mutually irreconcilable. Each claim-line is treated as a set of
   possibilities (a line claiming "Hunter or Tanner" is {Hunter, Tanner}),
   and a player contradicts themselves only if the intersection across ALL
   their claim-lines in the game is empty -- i.e. no single role could
   satisfy everything they said. Narrowing a disjunction down later (e.g.
   {Hunter,Tanner} then just {Tanner}) is NOT a contradiction, since the
   intersection is non-empty. This is computed as one running intersection
   over the whole game, not just adjacent pairs -- a 3-claim chain can be
   contradictory even when every adjacent pair overlaps.

Outputs three CSVs under the ic_targets root:
    role_conflicts.csv           -- one row per (game, role) with >=2 claimants
    self_contradictions.csv      -- one row per (game, player) with >=2 claim-lines
    player_conflict_features.csv -- one row per (game, player) with both kinds
                                     of conflict feature, ready to merge into
                                     a predictive model

Usage:
    python identity_claim_conflicts.py
    python identity_claim_conflicts.py --ic-targets-root ... --ready-root ...
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional


def find_repo_root(start: Optional[Path] = None, repo_name: str = "masters_thesis_sdg") -> Path:
    if start is None:
        start = Path.cwd().resolve()
    current = start
    while True:
        if current.name == repo_name:
            return current
        if current.parent == current:
            raise FileNotFoundError(f"Could not find repo root {repo_name!r} starting from {start}")
        current = current.parent


DEFAULT_IC_TARGETS_ROOT = None  # filled in via CLI default below
DEFAULT_READY_ROOT = None


def parse_transcript_speakers(txt_path: Path) -> dict:
    """line_number -> speaker, from a ready_for_annotation transcript .txt file."""
    speakers = {}
    if not txt_path.exists():
        return speakers
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*\[(\d+)\]\s*([^:]+):", line)
        if match:
            speakers[int(match.group(1))] = match.group(2).strip()
    return speakers


_speaker_cache = {}


def get_speakers(ready_root: Path, source_file: str) -> dict:
    if source_file not in _speaker_cache:
        _speaker_cache[source_file] = parse_transcript_speakers(ready_root / source_file)
    return _speaker_cache[source_file]


def normalize_role(role: str) -> str:
    """
    Normalizes role-name casing so 'Seer', 'seer', and 'SEER' are recognized
    as the same role for matching purposes. Roles aren't drawn from a fixed
    enum (the annotation prompt allows expansion roles named as stated in
    the transcript), so this just title-cases each word rather than
    validating against a known list -- 'village idiot' -> 'Village Idiot'.
    """
    return " ".join(word.capitalize() for word in role.strip().split())


def load_claim_events(ic_targets_root: Path, ready_root: Path) -> list:
    """
    One dict per (source, session, game, line_number, speaker, role) --
    every resolved (non-UNKNOWN, non-empty) claimed role, across all games.
    Flattened to one row per role -- used for same-role collision detection,
    where co-occurrence on a single line doesn't matter.
    """
    import pandas as pd

    rows = []
    for json_path in ic_targets_root.rglob("*.json"):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        metadata = record.get("metadata", {})
        source, session, game = metadata.get("source"), metadata.get("session"), metadata.get("game")
        source_file = metadata.get("source_file")
        if not (source and session and game):
            continue

        speakers = get_speakers(ready_root, source_file) if source_file else {}

        for item in record.get("items", []):
            line_number = item.get("line_number")
            speaker = speakers.get(line_number)
            if not speaker:
                continue
            for role in item.get("claimed_roles", []):
                if role == "UNKNOWN":
                    continue
                rows.append({
                    "source": source, "session": session, "game": game,
                    "line_number": line_number, "speaker": speaker, "role": normalize_role(role),
                })

    return pd.DataFrame(rows)


def load_claim_line_groups(ic_targets_root: Path, ready_root: Path) -> "pd.DataFrame":
    """
    One row per (source, session, game, speaker, line_number, roles) -- roles
    is the FULL set of roles claimed together on that one line (NOT flattened
    per-role). This preserves disjunctive claims ("I was the Hunter or
    Tanner" -> {"Hunter", "Tanner"} as one group) -- needed for
    self-contradiction detection, where co-occurrence on a line matters:
    narrowing a disjunction down later is consistent, not a contradiction.
    """
    import pandas as pd

    rows = []
    for json_path in ic_targets_root.rglob("*.json"):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        metadata = record.get("metadata", {})
        source, session, game = metadata.get("source"), metadata.get("session"), metadata.get("game")
        source_file = metadata.get("source_file")
        if not (source and session and game):
            continue

        speakers = get_speakers(ready_root, source_file) if source_file else {}

        for item in record.get("items", []):
            line_number = item.get("line_number")
            speaker = speakers.get(line_number)
            if not speaker:
                continue
            roles = [normalize_role(r) for r in item.get("claimed_roles", []) if r != "UNKNOWN"]
            if not roles:
                continue
            rows.append({
                "source": source, "session": session, "game": game,
                "speaker": speaker, "line_number": line_number, "roles": frozenset(roles),
            })

    return pd.DataFrame(rows)


def find_self_contradictions(claim_line_groups: "pd.DataFrame") -> "pd.DataFrame":
    """
    One row per (source, session, game, player) whose claim-lines are
    mutually inconsistent: the running intersection of ALL their claim-line
    role-sets (across the whole game, not just adjacent pairs) is empty --
    i.e. there is no single role that could satisfy everything they said.

    A disjunctive claim ("Hunter or Tanner") followed later by a narrowing
    claim ("Tanner") is NOT a contradiction: {Hunter,Tanner} & {Tanner} =
    {Tanner}, non-empty. Only genuinely irreconcilable claims are flagged --
    e.g. {Villager} & {Werewolf} = {}, or a 3-claim chain like
    {Hunter,Tanner} -> {Tanner,Villager} -> {Villager} where no adjacent
    pair conflicts but the full intersection across all three is empty.

    Requires >= 2 claim-lines from the same player to be eligible at all.
    """
    import pandas as pd

    if claim_line_groups.empty:
        return pd.DataFrame(columns=[
            "source", "session", "game", "player",
            "is_self_contradiction", "claim_line_numbers", "claim_sets",
        ])

    rows = []
    for (source, session, game, speaker), group in claim_line_groups.groupby(
        ["source", "session", "game", "speaker"]
    ):
        role_sets = list(group["roles"])
        if len(role_sets) < 2:
            continue

        running_intersection = role_sets[0]
        for role_set in role_sets[1:]:
            running_intersection = running_intersection & role_set

        rows.append({
            "source": source, "session": session, "game": game, "player": speaker,
            "is_self_contradiction": len(running_intersection) == 0,
            "claim_line_numbers": sorted(group["line_number"].tolist()),
            "claim_sets": [sorted(s) for s in role_sets],
        })

    return pd.DataFrame(rows)


def find_role_conflicts(events: "pd.DataFrame") -> "pd.DataFrame":
    """One row per (source, session, game, role) claimed by >=2 distinct speakers."""
    import pandas as pd

    if events.empty:
        return pd.DataFrame(columns=["source", "session", "game", "role", "players", "line_numbers", "n_players"])

    rows = []
    for (source, session, game, role), group in events.groupby(["source", "session", "game", "role"]):
        speakers = sorted(group["speaker"].unique())
        if len(speakers) < 2:
            continue
        rows.append({
            "source": source, "session": session, "game": game, "role": role,
            "players": speakers, "line_numbers": sorted(group["line_number"].unique().tolist()),
            "n_players": len(speakers),
        })
    return pd.DataFrame(rows)


def build_player_conflict_features(
    events: "pd.DataFrame", conflicts: "pd.DataFrame", contradictions: "pd.DataFrame"
) -> "pd.DataFrame":
    """
    One row per (source, session, game, player) who made at least one
    resolved claim, with:
      - n_distinct_roles_claimed_self: how many different roles this player
        themselves claimed (purely descriptive -- NOT necessarily deceptive;
        legitimate mid-game swaps can cause this too)
      - is_in_role_conflict: whether any role they claimed was also claimed
        by someone else
      - n_conflicting_claimants: how many OTHER distinct players also
        claimed at least one of this player's roles (0 if none)
      - conflicting_roles: which of their own claimed roles are contested
      - is_self_contradiction: whether their own claims (across the whole
        game) are mutually irreconcilable -- see find_self_contradictions
    """
    import pandas as pd

    if events.empty:
        return pd.DataFrame(columns=[
            "source", "session", "game", "player",
            "n_distinct_roles_claimed_self", "is_in_role_conflict",
            "n_conflicting_claimants", "conflicting_roles", "is_self_contradiction",
        ])

    self_roles = (
        events.groupby(["source", "session", "game", "speaker"])["role"]
        .apply(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={"speaker": "player", "role": "roles_claimed"})
    )
    self_roles["n_distinct_roles_claimed_self"] = self_roles["roles_claimed"].apply(len)

    # role -> {other claimants} lookup per game, from the conflicts table
    conflict_lookup = {}
    for row in conflicts.itertuples():
        key = (row.source, row.session, row.game, row.role)
        conflict_lookup[key] = set(row.players)

    def compute_conflict_fields(row):
        contested_roles = []
        other_claimants = set()
        for role in row["roles_claimed"]:
            key = (row["source"], row["session"], row["game"], role)
            claimants = conflict_lookup.get(key)
            if claimants:
                contested_roles.append(role)
                other_claimants |= (claimants - {row["player"]})
        return pd.Series({
            "is_in_role_conflict": len(contested_roles) > 0,
            "n_conflicting_claimants": len(other_claimants),
            "conflicting_roles": contested_roles,
        })

    conflict_fields = self_roles.apply(compute_conflict_fields, axis=1)
    result = pd.concat([self_roles, conflict_fields], axis=1)

    contradiction_lookup = set()
    if not contradictions.empty:
        for row in contradictions[contradictions["is_self_contradiction"]].itertuples():
            contradiction_lookup.add((row.source, row.session, row.game, row.player))
    result["is_self_contradiction"] = [
        (s, se, g, p) in contradiction_lookup
        for s, se, g, p in zip(result["source"], result["session"], result["game"], result["player"])
    ]

    return result[[
        "source", "session", "game", "player", "roles_claimed",
        "n_distinct_roles_claimed_self", "is_in_role_conflict",
        "n_conflicting_claimants", "conflicting_roles", "is_self_contradiction",
    ]]


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description="Detect same-role identity-claim conflicts.")
    parser.add_argument("--ic-targets-root", type=Path, default=None)
    parser.add_argument("--ready-root", type=Path, default=None)
    args = parser.parse_args()

    if args.ic_targets_root is None or args.ready_root is None:
        repo_root = find_repo_root()
        if args.ic_targets_root is None:
            args.ic_targets_root = (
                repo_root / "data" / "processed" / "lai2023" / "identity_claim_transcripts" / "ic_targets"
            )
        if args.ready_root is None:
            args.ready_root = (
                repo_root / "data" / "processed" / "lai2023" / "identity_claim_transcripts" / "ready_for_annotation"
            )

    print(f"IC targets root : {args.ic_targets_root}")
    print(f"Ready root      : {args.ready_root}")

    events = load_claim_events(args.ic_targets_root, args.ready_root)
    print(f"\n{len(events)} resolved claim events across all games.")

    conflicts = find_role_conflicts(events)
    print(f"{len(conflicts)} (game, role) conflicts found "
          f"across {conflicts[['source','session','game']].drop_duplicates().shape[0] if not conflicts.empty else 0} games.")

    claim_line_groups = load_claim_line_groups(args.ic_targets_root, args.ready_root)
    contradictions = find_self_contradictions(claim_line_groups)
    n_contradictions = int(contradictions["is_self_contradiction"].sum()) if not contradictions.empty else 0
    print(f"{n_contradictions} player(s) with self-contradictory claims "
          f"(out of {len(contradictions)} players who made >=2 claim-lines).")

    player_features = build_player_conflict_features(events, conflicts, contradictions)
    n_in_conflict = int(player_features["is_in_role_conflict"].sum()) if not player_features.empty else 0
    print(f"{len(player_features)} (game, player) rows with at least one claim; "
          f"{n_in_conflict} of them are part of a conflict.")

    conflicts_out = args.ic_targets_root / "role_conflicts.csv"
    features_out = args.ic_targets_root / "player_conflict_features.csv"
    contradictions_out = args.ic_targets_root / "self_contradictions.csv"

    conflicts_to_write = conflicts.copy()
    if not conflicts_to_write.empty:
        conflicts_to_write["players"] = conflicts_to_write["players"].apply(json.dumps)
        conflicts_to_write["line_numbers"] = conflicts_to_write["line_numbers"].apply(json.dumps)
    conflicts_to_write.to_csv(conflicts_out, index=False)

    contradictions_to_write = contradictions.copy()
    if not contradictions_to_write.empty:
        contradictions_to_write["claim_line_numbers"] = contradictions_to_write["claim_line_numbers"].apply(json.dumps)
        contradictions_to_write["claim_sets"] = contradictions_to_write["claim_sets"].apply(json.dumps)
    contradictions_to_write.to_csv(contradictions_out, index=False)

    features_to_write = player_features.copy()
    if not features_to_write.empty:
        features_to_write["roles_claimed"] = features_to_write["roles_claimed"].apply(json.dumps)
        features_to_write["conflicting_roles"] = features_to_write["conflicting_roles"].apply(json.dumps)
    features_to_write.to_csv(features_out, index=False)

    print(f"\nWrote: {conflicts_out}")
    print(f"Wrote: {contradictions_out}")
    print(f"Wrote: {features_out}")


if __name__ == "__main__":
    main()