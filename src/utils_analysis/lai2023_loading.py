\
from pathlib import Path

from .paths import load_json


DATASETS = ("yt", "ego4d")
SPLITS = ("train", "val", "test")


def get_annotation_dirs(data_dir):
    data_dir = Path(data_dir)
    return {
        "yt": data_dir / "Youtube" / "split",
        "ego4d": data_dir / "Ego4D" / "split",
    }


def get_outcome_dirs(data_dir):
    data_dir = Path(data_dir)
    return {
        "yt": data_dir / "Youtube" / "vote_outcome_youtube_released",
        "ego4d": data_dir / "Ego4D" / "vote_outcome_ego4d",
    }


def load_annotation_splits(data_dir):
    """
    Returns:
        {
          "yt": {"train": [...], "val": [...], "test": [...]},
          "ego4d": {"train": [...], "val": [...], "test": [...]}
        }
    """
    annotation_dirs = get_annotation_dirs(data_dir)

    annot_splits = {}
    for dataset, folder in annotation_dirs.items():
        annot_splits[dataset] = {}
        for split in SPLITS:
            path = folder / f"{split}.json"
            annot_splits[dataset][split] = load_json(path)

    return annot_splits


def load_outcome_index(data_dir):
    """
    Loads human voting-outcome files.

    Returns:
        {
          "yt": {session_key: session_data_json},
          "ego4d": {session_key: session_data_json}
        }
    """
    outcome_dirs = get_outcome_dirs(data_dir)

    outcome_index = {}
    for dataset, folder in outcome_dirs.items():
        outcome_index[dataset] = {
            path.stem: load_json(path)
            for path in sorted(folder.glob("*.json"))
        }

    return outcome_index


def get_session_key(game, dataset):
    """
    Session identifier used to align split annotations with vote-outcome files.

    For YouTube:
        game["video_name"]

    For Ego4D:
        game["EG_ID"]
    """
    if dataset == "yt":
        return game["video_name"]
    if dataset == "ego4d":
        return game["EG_ID"]
    raise ValueError(f"Unknown dataset: {dataset}")


def get_game_id(game):
    return game["Game_ID"]


def get_outcome_record(game, dataset, outcome_index):
    """
    Returns the human vote-outcome record for a game.
    """
    session_key = get_session_key(game, dataset)
    game_id = get_game_id(game)
    return outcome_index[dataset][session_key][game_id]


def build_game_lookup(annot_splits):
    """
    Builds lookup tables for aligning external results, e.g. LLM votes, with
    Lai annotation split games.

    Returns:
        game_lookup[(dataset, session_key, game_id)] = game
        split_lookup[(dataset, session_key, game_id)] = split
    """
    game_lookup = {}
    split_lookup = {}

    for dataset, split_map in annot_splits.items():
        for split, games in split_map.items():
            for game in games:
                key = (dataset, get_session_key(game, dataset), get_game_id(game))
                game_lookup[key] = game
                split_lookup[key] = split

    return game_lookup, split_lookup


def count_games_by_split(annot_splits):
    return {
        dataset: {
            split: len(games)
            for split, games in split_map.items()
        }
        for dataset, split_map in annot_splits.items()
    }
