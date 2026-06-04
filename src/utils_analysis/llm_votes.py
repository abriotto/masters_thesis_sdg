from pathlib import Path
import pandas as pd

from .paths import load_json


def prompt_dir_name(prompt_version):
    prompt_version = str(prompt_version)
    return prompt_version if prompt_version.startswith("prompt_") else f"prompt_{prompt_version}"


def source_to_dataset(source):
    if source in {"Youtube", "YouTube", "yt"}:
        return "yt"
    if source in {"Ego4D", "ego4d"}:
        return "ego4d"
    raise ValueError(f"Unknown source: {source}")


def resolve_llm_prompt_dir(results_dir, llm, prompt_version):
    """
    Resolves:
        results/voting/<llm-folder>/prompt_vX

    `llm` can be the exact folder name or a substring.
    """
    results_dir = Path(results_dir)
    pdir = prompt_dir_name(prompt_version)

    exact = results_dir / llm / pdir
    if exact.exists():
        return exact

    matches = [
        d / pdir
        for d in results_dir.iterdir()
        if d.is_dir() and llm in d.name and (d / pdir).exists()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            "Multiple matching LLM result folders found:\n"
            + "\n".join(str(m) for m in matches)
        )

    raise FileNotFoundError(
        f"Could not find result dir for llm={llm!r}, prompt={pdir!r} under {results_dir}"
    )


def canonicalize_vote_name(chosen_vote, players):
    """
    Robustly maps model output to one of the known player names.
    Returns None if no match is possible.
    """
    if chosen_vote is None:
        return None

    chosen_vote = str(chosen_vote).strip()

    if chosen_vote in players:
        return chosen_vote

    norm_map = {str(p).strip().lower(): p for p in players}
    return norm_map.get(chosen_vote.lower(), None)


def extract_chosen_vote(obj):
    """
    Supports the JSON structure produced by your voting runner.

    Preferred field:
        obj["validation"]["chosen_vote"]

    Fallback:
        obj["parsed_output"]["chosen_vote"]
    """
    validation = obj.get("validation", {}) or {}
    parsed = obj.get("parsed_output", {}) or {}

    return (
        validation.get("chosen_vote")
        or parsed.get("chosen_vote")
    )


def is_valid_vote_result(obj):
    """
    A result is considered valid when validation.is_valid is True.
    If the key is missing, returns False by default.
    """
    validation = obj.get("validation", {}) or {}
    return bool(validation.get("is_valid", False))


def load_llm_votes(results_dir, llm, prompt_version, keep_invalid=False, drop_duplicates=True):
    """
    Loads LLM voting results.

    Returns:
        votes_df, bad_df

    votes_df has one row per valid game-level LLM vote:
        dataset, source, session_key, game_id, chosen_vote, player_names, path

    bad_df stores skipped files and why they were skipped.
    """
    model_prompt_dir = resolve_llm_prompt_dir(results_dir, llm, prompt_version)

    rows = []
    bad_rows = []

    for source_folder in ("Youtube", "Ego4D"):
        folder = model_prompt_dir / source_folder
        if not folder.exists():
            continue

        for path in sorted(folder.glob("*.json")):
            obj = load_json(path)

            source = obj.get("source", source_folder)
            dataset = source_to_dataset(source)

            session_key = obj.get("session_name")
            game_id = obj.get("game_key")

            players = obj.get("player_names", [])
            players = [str(p).strip() for p in players]

            chosen_vote_raw = extract_chosen_vote(obj)
            chosen_vote = canonicalize_vote_name(chosen_vote_raw, players)

            valid = is_valid_vote_result(obj)

            reason_bad = None
            if not valid:
                reason_bad = "validation_not_valid"
            elif session_key is None or game_id is None:
                reason_bad = "missing_session_or_game_id"
            elif len(players) == 0:
                reason_bad = "missing_player_names"
            elif chosen_vote is None:
                reason_bad = f"chosen_vote_not_in_players: {chosen_vote_raw}"

            row = {
                "dataset": dataset,
                "source": source,
                "session_key": session_key,
                "game_id": game_id,
                "chosen_vote": chosen_vote,
                "chosen_vote_raw": chosen_vote_raw,
                "player_names": players,
                "path": str(path),
                "is_valid": valid,
                "reason_bad": reason_bad,
            }

            if reason_bad is None or keep_invalid:
                rows.append(row)

            if reason_bad is not None:
                bad_rows.append(row)

    votes_df = pd.DataFrame(rows)
    bad_df = pd.DataFrame(bad_rows)

    if drop_duplicates and len(votes_df) > 0:
        votes_df = votes_df.drop_duplicates(
            subset=["dataset", "session_key", "game_id"],
            keep="last",
        ).reset_index(drop=True)

    return votes_df, bad_df
