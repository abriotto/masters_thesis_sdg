from __future__ import annotations

"""
Turn raw Jin et al. ONUW episodes into transcripts shaped like the lai2023 ones.

What is kept
------------
Every message with `visible_to == "all"`: the Moderator's public script (welcome,
night wake/close calls, phase and round boundaries) plus the 15 public discussion
turns. This mirrors the lai2023 transcripts, 87% of which contain the narrator
reading the night script aloud.

What is dropped
---------------
- Anything with restricted visibility (`visible_to` is a player id or a list):
  the Moderator's private results ("The two roles you checked are..."), the
  players' private night actions ("I want to switch my role with player2"), and
  the private vote declarations. An external observer never sees these.
- The final "Game over ..." line. It is marked `visible_to == "all"`, but it names
  the eliminated player and their role, i.e. it *is* the label. Keeping it would
  make the finetuning task trivial and the experiment meaningless. `--keep_outcome_line`
  exists only so this exclusion is explicit rather than silent.

Output
------
data/processed/jin2024_onuw/role_inference/
    transcripts/episode_XXX.txt   # same header + [n] Speaker: text layout as lai2023
    index.json                    # one record per episode, mirroring index_cleaned.json
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io_utils import find_repo_root, load_json


# Only used by --name_mode real, which is NOT the default.
#
# Remapping to first names would match the surface form of the lai2023 transcripts,
# where coreference has to be tracked by name rather than by an ordinal id. But with
# 120 episodes each name occupies only 7-20 slots, and sampling noise alone produced
# per-name werewolf rates from 0% to 58% against a 26.2% base rate. A LoRA adapter can
# absorb that, and any latent attribute of the name set (gender coding, origin) would
# carry over to the real names in the eval set. Upstream ids are free of this: each
# appears exactly 120 times, per-seat werewolf rates sit at 19-31%, and "player3" never
# occurs in lai2023, so nothing name-shaped can transfer. Speaking order is unchanged
# either way. Kept here as a documented alternative, not a recommendation.
PLAYER_NAME_POOL = [
    "Anika", "Bruno", "Clara", "Delia", "Dmitri", "Elena", "Emil", "Farah",
    "Felix", "Gabe", "Greta", "Hana", "Hugo", "Iris", "Ivan", "Jana",
    "Jonas", "Karl", "Kira", "Lena", "Liam", "Milo", "Mona", "Nadia",
    "Nils", "Nora", "Omar", "Otto", "Owen", "Petra", "Pia", "Priya",
    "Rafael", "Rosa", "Selma", "Theo", "Tobias", "Uma", "Vince", "Wendy",
    "Wren", "Yara", "Yusuf", "Zoe",
]

MODERATOR = "Moderator"

DAY_START_MARKER = "Night phase ends"
VOTE_MARKER = "Day phase ends"
OUTCOME_MARKER = "Game over"
ROUND_END_PATTERN = re.compile(r"^Discussion round (\d+) ends")

# Public Moderator lines belonging to the night phase, i.e. everything before the
# "Night phase ends" marker. Only used when --drop_night_lines is passed.
SMART_QUOTE_MAP = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "—": " - ",
    "–": "-",
}


def normalize_text(text: str) -> str:
    for source, target in SMART_QUOTE_MAP.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def is_public(message: dict[str, Any]) -> bool:
    return message.get("visible_to") == "all"


def assign_names(
    player_ids: list[str],
    episode_id: str,
    name_mode: str,
    seed: int,
) -> dict[str, str]:
    if name_mode == "player_ids":
        return {pid: pid for pid in player_ids}

    if len(player_ids) > len(PLAYER_NAME_POOL):
        raise ValueError(
            f"{episode_id}: {len(player_ids)} players but only "
            f"{len(PLAYER_NAME_POOL)} names in the pool."
        )

    # Seeded per episode so the mapping is reproducible and independent of ordering.
    rng = random.Random(f"{seed}:{episode_id}")
    chosen = rng.sample(PLAYER_NAME_POOL, len(player_ids))
    return dict(zip(player_ids, chosen))


def rename_in_text(text: str, name_map: dict[str, str]) -> str:
    """Replace player ids inside message content, longest id first."""
    if not name_map or all(k == v for k, v in name_map.items()):
        return text

    for pid in sorted(name_map, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(pid)}\b", name_map[pid], text, flags=re.IGNORECASE)
    return text


def select_messages(
    messages: list[dict[str, Any]],
    episode_id: str,
    keep_outcome_line: bool,
    drop_night_lines: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public = [m for m in messages if is_public(m)]
    if not public:
        raise ValueError(f"{episode_id}: no public messages found.")

    day_start = [
        i for i, m in enumerate(public) if m["content"].startswith(DAY_START_MARKER)
    ]
    outcome = [i for i, m in enumerate(public) if m["content"].startswith(OUTCOME_MARKER)]

    if len(day_start) != 1:
        raise ValueError(
            f"{episode_id}: expected exactly one {DAY_START_MARKER!r} marker, "
            f"found {len(day_start)}."
        )
    if len(outcome) != 1:
        raise ValueError(
            f"{episode_id}: expected exactly one {OUTCOME_MARKER!r} line, "
            f"found {len(outcome)}."
        )

    day_start_idx = day_start[0]
    outcome_idx = outcome[0]

    keep_indices = range(len(public))
    if not keep_outcome_line:
        keep_indices = [i for i in keep_indices if i != outcome_idx]
    if drop_night_lines:
        # Everything before the "Night phase ends" marker is night-phase narration.
        keep_indices = [i for i in keep_indices if i >= day_start_idx]

    kept = [public[i] for i in keep_indices]

    meta = {
        "num_public_messages": len(public),
        "num_kept_messages": len(kept),
        "outcome_line_dropped": not keep_outcome_line,
        "night_lines_dropped": drop_night_lines,
    }
    return kept, meta


def build_transcript(
    episode_id: str,
    game_key: str,
    kept: list[dict[str, Any]],
    name_map: dict[str, str],
    player_names: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    turn_records: list[dict[str, Any]] = []

    for line_no, message in enumerate(kept, start=1):
        agent = message["agent_name"]
        speaker = MODERATOR if agent == MODERATOR else name_map.get(agent, agent)
        content = normalize_text(rename_in_text(message["content"], name_map))

        lines.append(f"[{line_no}] {speaker}: {content}")

        round_match = ROUND_END_PATTERN.match(message["content"])
        turn_records.append(
            {
                "line_no": line_no,
                "speaker": speaker,
                "agent_id": agent,
                "is_moderator": agent == MODERATOR,
                "source_turn": message.get("turn"),
                "strategy": message.get("strategy") or None,
                "round_end": int(round_match.group(1)) if round_match else None,
                "is_vote_marker": message["content"].startswith(VOTE_MARKER),
            }
        )

    header = (
        f"Source: jin2024\n"
        f"Session: {episode_id}\n"
        f"Game: {game_key}\n"
        f"Players: {', '.join(player_names)}\n\n"
        f"Transcript:\n"
    )
    return header + "\n".join(lines) + "\n", turn_records


def build_round_cut_points(turn_records: list[dict[str, Any]]) -> dict[str, int]:
    """Line numbers at which each discussion round closes, for prefix augmentation."""
    cuts: dict[str, int] = {}
    for record in turn_records:
        if record["round_end"] is not None:
            cuts[f"round_{record['round_end']}"] = record["line_no"]
        if record["is_vote_marker"]:
            cuts["round_3"] = record["line_no"]
    return cuts


def process_episode(
    path: Path,
    name_mode: str,
    seed: int,
    keep_outcome_line: bool,
    drop_night_lines: bool,
) -> tuple[str, dict[str, Any]]:
    episode = load_json(path)
    episode_id = path.stem
    evaluation = episode["evaluation"]

    start_roles_by_id = evaluation["roles_assigned"]
    end_roles_by_id = evaluation["roles_ground_truth"]

    if set(start_roles_by_id) != set(end_roles_by_id):
        raise ValueError(f"{episode_id}: roles_assigned and roles_ground_truth disagree on players.")

    player_ids = sorted(end_roles_by_id, key=lambda pid: (len(pid), pid))
    name_map = assign_names(player_ids, episode_id, name_mode, seed)
    player_names = [name_map[pid] for pid in player_ids]

    kept, selection_meta = select_messages(
        messages=episode["messages"],
        episode_id=episode_id,
        keep_outcome_line=keep_outcome_line,
        drop_night_lines=drop_night_lines,
    )

    game_key = "Game1"
    transcript, turn_records = build_transcript(
        episode_id=episode_id,
        game_key=game_key,
        kept=kept,
        name_map=name_map,
        player_names=player_names,
    )

    voting_result = evaluation.get("voting_result") or {}
    dialogue_turns = [r for r in turn_records if not r["is_moderator"]]

    record: dict[str, Any] = {
        "source": "jin2024",
        "session_name": episode_id,
        "game_key": game_key,
        "source_json_path": None,  # filled in by main(), relative to repo root
        "processed_txt_path": None,
        "num_players": len(player_ids),
        "num_turns": len(turn_records),
        "num_dialogue_turns": len(dialogue_turns),
        "player_ids": player_ids,
        "player_names": player_names,
        "player_id_map": name_map,
        "start_roles": [start_roles_by_id[pid] for pid in player_ids],
        "end_roles": [end_roles_by_id[pid] for pid in player_ids],
        "role_pool": list(evaluation.get("role_pool", [])),
        "winner": evaluation.get("winner"),
        "voting_result": [voting_result.get(pid, 0) for pid in player_ids],
        "num_werewolves_end": sum(1 for pid in player_ids if end_roles_by_id[pid] == "Werewolf"),
        "strategies": [r["strategy"] for r in dialogue_turns],
        "round_cut_points": build_round_cut_points(turn_records),
        "selection": selection_meta,
    }
    return transcript, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw Jin et al. ONUW episodes into lai2023-style transcripts."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/jin2024_onuw/gpt4_dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/jin2024_onuw/role_inference",
    )
    parser.add_argument(
        "--name_mode",
        type=str,
        choices=["player_ids", "real"],
        default="player_ids",
        help=(
            "'player_ids' (default) keeps the upstream identifiers. 'real' remaps them "
            "onto first names disjoint from the lai2023 eval names - see the note on "
            "PLAYER_NAME_POOL for why this is NOT the default."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--drop_night_lines",
        action="store_true",
        help=(
            "Ablation: also drop the Moderator's public night script. Off by default, "
            "since 87%% of the lai2023 eval transcripts contain that narration."
        ),
    )
    parser.add_argument(
        "--keep_outcome_line",
        action="store_true",
        help=(
            "Keep the final 'Game over ...' line. This LEAKS THE LABEL and is for "
            "inspection only - never use it to build a training set."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    input_dir = repo_root / args.input_dir
    output_dir = repo_root / args.output_dir
    transcripts_dir = output_dir / "transcripts"

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Raw episodes not found at {input_dir}. Run src.finetuning.download_jin first."
        )

    episode_paths = sorted(input_dir.glob("episode_*.json"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.json files under {input_dir}.")

    if args.keep_outcome_line:
        print(
            "WARNING: --keep_outcome_line is set. The transcripts will contain the "
            "game outcome and role reveal. Do NOT train on this output.",
            flush=True,
        )

    transcripts_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for path in episode_paths:
        try:
            transcript, record = process_episode(
                path=path,
                name_mode=args.name_mode,
                seed=args.seed,
                keep_outcome_line=args.keep_outcome_line,
                drop_night_lines=args.drop_night_lines,
            )
        except Exception as exc:
            failures.append({"episode": path.stem, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED - {path.stem}: {exc}", flush=True)
            continue

        txt_path = transcripts_dir / f"{path.stem}.txt"
        if txt_path.exists() and not args.overwrite:
            print(f"SKIP - {txt_path.name} already exists (use --overwrite)", flush=True)
        else:
            txt_path.write_text(transcript, encoding="utf-8")

        record["source_json_path"] = str(path.relative_to(repo_root)).replace("\\", "/")
        record["processed_txt_path"] = str(txt_path.relative_to(repo_root)).replace("\\", "/")
        index.append(record)

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "num_episodes": len(index),
        "num_failures": len(failures),
        "failures": failures,
        "name_mode": args.name_mode,
        "seed": args.seed,
        "drop_night_lines": args.drop_night_lines,
        "keep_outcome_line": args.keep_outcome_line,
        "num_players_distribution": _counter([r["num_players"] for r in index]),
        "num_turns_distribution": _counter([r["num_turns"] for r in index]),
        "winner_distribution": _counter([r["winner"] for r in index]),
        "end_role_distribution": _counter([role for r in index for role in r["end_roles"]]),
        "werewolves_end_distribution": _counter([r["num_werewolves_end"] for r in index]),
    }
    (output_dir / "preprocess_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nProcessed {len(index)} episodes ({len(failures)} failed).")
    print(f"Transcripts: {transcripts_dir}")
    print(f"Index:       {index_path}")
    print(f"Turns per transcript: {stats['num_turns_distribution']}")
    print(f"Winner:               {stats['winner_distribution']}")
    print(f"Werewolves at end:    {stats['werewolves_end_distribution']}")


def _counter(values: list[Any]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


if __name__ == "__main__":
    main()
