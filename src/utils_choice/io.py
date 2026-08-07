"""Loading vote targets, and writing result tables.

``save_tables`` owns its output directory: it writes the tables it is given and
deletes any other CSV there, so a directory can never mix results from the
current analysis with leftovers from a superseded one.
"""

from collections import Counter
import json

import pandas as pd

from .features import CIRCLE_OPTION, ckey

STOCHASTIC_RUNS = ("run_1", "run_2", "run_3")
GREEDY_RUN = "greedy_t0"


def load_vote_tables(analysis_root, tables_rel):
    """Per-file and per-game LLM vote tables, plus the rosters."""
    def short_label(name):
        import re
        m = re.search(r"(\d+B)", name)
        return m.group(1) if m else name

    file_frames, game_frames = [], []
    for model_dir in sorted(analysis_root.iterdir()):
        tables = model_dir / tables_rel
        if not (tables / "llm_vote_file_level.csv").exists():
            continue
        label = short_label(model_dir.name)
        f = pd.read_csv(tables / "llm_vote_file_level.csv"); f["model"] = label
        g = pd.read_csv(tables / "llm_vote_game_level.csv"); g["model"] = label
        file_frames.append(f); game_frames.append(g)
    votes = pd.concat(file_frames, ignore_index=True)
    games = pd.concat(game_frames, ignore_index=True)

    roster_by_key = {}
    for _, r in games.drop_duplicates(
            subset=["source", "session_name", "game_key"]).iterrows():
        players = r["player_names"]
        roster_by_key[ckey(r["source"], r["session_name"], r["game_key"])] = {
            "source": r["source"], "session": r["session_name"],
            "game": r["game_key"],
            "players": json.loads(players) if isinstance(players, str) else players}
    return votes, games, roster_by_key


def llm_vote_targets(votes, roster_by_key, run_labels):
    """{model: {game key: {alternative: count}}} plus a per-model summary.

    A circle vote is a choice the LLM made, so it becomes a count on the
    ``CIRCLE_OPTION`` alternative rather than a dropped instance. Runs whose
    named target cannot be matched to the roster are dropped from that game's
    denominator and counted.
    """
    targets, rows, unmatched = {}, [], Counter()
    sel = votes[votes["run_label"].isin(run_labels)
                & votes["status"].isin(["player_vote", "circle_vote"])]
    for (model, source, session, game), g in sel.groupby(
            ["model", "source", "session_name", "game_key"]):
        key = ckey(source, session, game)
        roster = roster_by_key.get(key, {}).get("players")
        if roster is None:
            continue
        nmap = {p.strip().lower(): p for p in roster}
        picks = []
        for _, r in g.iterrows():
            if bool(r["is_circle_vote"]):
                picks.append(CIRCLE_OPTION)
            else:
                name = r["chosen_player_name"]
                picks.append(nmap.get(str(name).strip().lower())
                             if isinstance(name, str) else None)
        good = [p for p in picks if p is not None]
        unmatched[model] += len(picks) - len(good)
        if not good:
            continue
        targets.setdefault(model, {})[key] = dict(Counter(good))
        rows.append({"model": model, "key": key, "n_runs": len(good),
                     "is_split": int(len(set(good)) > 1),
                     "circle_share": good.count(CIRCLE_OPTION) / len(good)})
    return targets, pd.DataFrame(rows), dict(unmatched)


def human_vote_targets(village_csv, roster_by_key, min_voters=2):
    """Village vote counts per game, restricted to village-aligned voters.

    Werewolves vote strategically, so their votes do not measure suspicion;
    games with fewer than ``min_voters`` village voters cannot express a
    distribution and are dropped.
    """
    village = pd.read_csv(village_csv)
    village["key"] = [ckey(*g.split(" / ")) for g in village["game_id"]]
    targets, rows, unmatched = {}, [], 0
    for _, r in village.iterrows():
        if r["n_village_aligned_votes"] < min_voters:
            continue
        key = r["key"]
        roster = roster_by_key.get(key, {}).get("players")
        if roster is None:
            continue
        nmap = {p.strip().lower(): p for p in roster}
        counts = {}
        for name, c in json.loads(r["village_vote_counts"]).items():
            p = nmap.get(str(name).strip().lower())
            if p is None:
                unmatched += c
            else:
                counts[p] = counts.get(p, 0) + c
        if not counts:
            continue
        targets[key] = counts
        rows.append({"key": key, "n_voters": sum(counts.values()),
                     "is_split": int(len(counts) > 1)})
    return targets, pd.DataFrame(rows), unmatched, village


def shares_from_targets(targets):
    """Vote distributions, for the ceiling and entropy calculations."""
    return [{k: v / sum(c.values()) for k, v in c.items()} for c in targets.values()]


def crowd_modal_map(village, min_voters=2):
    return {ckey(*r["game_id"].split(" / ")): set(json.loads(r["village_top_target_names"]))
            for _, r in village.iterrows()
            if r["n_village_aligned_votes"] >= min_voters}


def save_tables(out_dir, tables):
    """Write ``{name: DataFrame}`` as CSVs and remove every other CSV in the
    directory, so only the current analysis's outputs survive."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = set()
    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        df.to_csv(out_dir / f"{name}.csv", index=False)
        written.add(f"{name}.csv")
    stale = [p for p in out_dir.glob("*.csv") if p.name not in written]
    for p in stale:
        p.unlink()
    return sorted(written), sorted(p.name for p in stale)
