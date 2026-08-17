"""Feature construction for the vote choice models.

Two blocks describing the same persuasion acts at two resolutions:

* **A -- techniques**: the annotated act was used (Lai et al. annotations).
* **B -- resolved**: the same acts with their target and content -- who an
  accusation landed on, and what an identity claim said.

Consistency judgements (role conflicts, self-contradictions) are deliberately
excluded: they are verdicts on whether a claim held up, not persuasion acts.

Every block also has a temporal variant, counting the same features separately
in the first and second half of the game. The split cuts the annotated dialogue
into two equal halves: an utterance is ``early`` if its **ordinal position** in
the dialogue is ``<= n_utterances / 2``.

Position, not ``Rec_Id``, is what defines the cut. ``Rec_Id`` and the DeepSeek
``line_number`` both index the *full transcript*, which is about 5% longer than
the annotated dialogue because unannotated lines (announcer turns) are skipped.
Comparing either of them against ``n_utterances / 2`` mixes two scales and moves
the cut roughly 5% early, which is what this module used to do. Block B is
mapped onto the same scale by converting its ``line_number`` to a dialogue
ordinal, so both blocks still split at exactly the same point.
"""

from bisect import bisect_right
from collections import Counter
import json
import re

import numpy as np
import pandas as pd

try:                                    # notebooks put src/ on sys.path
    from utils.speakers import normalize_speaker
except ImportError:                     # imported as src.utils_choice.features
    from src.utils.speakers import normalize_speaker

PT_LABELS = ["Accusation", "Defense", "Interrogation", "Identity Declaration",
             "Evidence", "Call for Action"]
INFO_ROLES = {"Seer", "Robber", "Troublemaker", "Insomniac"}

CIRCLE_OPTION = "__NO_WEREWOLF__"
CIRCLE_FEATURE = "is_circle_option"

BLOCK_A = [f"pt_{l.lower().replace(' ', '_')}" for l in PT_LABELS] + ["n_utterances"]
ACC_FEATURES = ["werewolf_count", "deception_count"]
IC_FEATURES = ["claims_werewolf", "claims_info_role", "n_distinct_roles_claimed_self"]
BLOCK_B = ACC_FEATURES + IC_FEATURES


def temporal(names):
    """early/late variants of a list of feature names."""
    return [f"{n}_early" for n in names] + [f"{n}_late" for n in names]


BLOCK_A_T, BLOCK_B_T = temporal(BLOCK_A), temporal(BLOCK_B)
BLOCKS = {"A_techniques": BLOCK_A,
          "B_resolved": BLOCK_B,
          "C_both": BLOCK_A + BLOCK_B,
          "A_temporal": BLOCK_A_T,
          "B_temporal": BLOCK_B_T,
          "C_temporal": BLOCK_A_T + BLOCK_B_T}
PLAYER_FEATURES = BLOCK_A + BLOCK_B + BLOCK_A_T + BLOCK_B_T
ALL_FEATURES = PLAYER_FEATURES + [CIRCLE_FEATURE]


def cols_for(block, include_circle=True):
    """Feature columns for a block, with the alternative-specific constant for
    the 'No Werewolf' option unless the ballot has no such option."""
    return BLOCKS[block] + ([CIRCLE_FEATURE] if include_circle else [])


# --------------------------------------------------------------- game keys ---
def canonical_session(x):
    return re.sub(r"\s+", " ", str(x).strip().replace("#", " ")).lower()


def canonical_game(x):
    m = re.search(r"\d+", str(x))
    return f"game{int(m.group())}" if m else str(x).strip().lower()


def ckey(source, session, game):
    """Annotations, DeepSeek outputs and vote tables spell sessions
    differently; everything joins on this canonical triple."""
    return (str(source).strip(), canonical_session(session), canonical_game(game))


# ------------------------------------------------------------ temporal split --
def half_from_position(position, n_utt):
    """First or second half of the annotated dialogue, by ordinal position."""
    return "early" if n_utt and position <= n_utt / 2 else "late"


def half_from_line_number(line_number, recs, n_utt):
    """Same split for block B, whose positions are full-transcript line numbers.

    ``recs`` is the sorted list of ``Rec_Id`` for the game's annotated
    utterances. The number of them at or before ``line_number`` is that line's
    ordinal position in the dialogue, which puts block B on block A's scale.
    """
    return half_from_position(bisect_right(recs, line_number), n_utt)


# ------------------------------------------------------------------ block A --
# Lai et al. ship the Avalon games in split/avalon.json alongside the ONUW splits.
# Those are a different game and are excluded from this corpus, but a bare glob
# over split/*.json picks them up: 8 games and 2,841 annotated utterances. Match
# the three canonical split files by name so they are never loaded.
ONUW_SPLIT_FILES = ("train.json", "val.json", "test.json")


def iter_annotation_games(annot_root):
    for split_file in sorted(annot_root.rglob("split/*.json")):
        if split_file.name not in ONUW_SPLIT_FILES:
            continue
        source = split_file.parent.parent.name
        for game in json.loads(split_file.read_text(encoding="utf-8")):
            session = (game.get("video_name") or game.get("session")
                       or game.get("YT_ID") or game.get("EG_ID"))
            game_id = game.get("Game_ID") or game.get("game")
            if session is not None and game_id is not None:
                yield source, session, game_id, game.get("Dialogue", [])


def load_technique_features(annot_root):
    """Per-player technique counts, aggregate and early/late.

    Also returns the game timeline (length in utterances, and Rec_Id -> speaker)
    that the block-B loader needs so both blocks split at the same point.
    """
    rows, seen = [], set()
    game_length, rec_speaker = {}, {}
    for source, session, game_id, dialogue in iter_annotation_games(annot_root):
        key = ckey(source, session, game_id)
        if key in seen:                  # the same game can appear twice
            continue
        seen.add(key)
        n_utt = len(dialogue)
        game_length[key] = n_utt
        rec_speaker[key] = {}
        per_speaker = {}
        for i, utt in enumerate(dialogue, start=1):
            sp = normalize_speaker(utt.get("speaker"))
            if sp is None:
                continue
            rec = int(utt.get("Rec_Id", i))
            rec_speaker[key][rec] = sp
            half = half_from_position(i, n_utt)
            d = per_speaker.setdefault(sp, Counter())
            d["n_utterances"] += 1
            d[f"n_utterances_{half}"] += 1
            for ann in utt.get("annotation", []):
                if ann in PT_LABELS:
                    name = f"pt_{ann.lower().replace(' ', '_')}"
                    d[name] += 1
                    d[f"{name}_{half}"] += 1
        for sp, counts in per_speaker.items():
            rows.append({"key": key, "speaker": sp, **counts})

    pt_df = pd.DataFrame(rows)
    for c in BLOCK_A + BLOCK_A_T:
        if c not in pt_df:
            pt_df[c] = 0.0
    pt_df = pt_df.fillna(0)
    assert len(seen) > 0, f"no annotation games found under {annot_root}"
    assert np.allclose(pt_df[[f"{c}_early" for c in BLOCK_A]].values
                       + pt_df[[f"{c}_late" for c in BLOCK_A]].values,
                       pt_df[BLOCK_A].values), "early + late must equal the total"
    return pt_df, game_length, rec_speaker, seen


# ------------------------------------------------------------------ block B --
def load_accusation_features(acc_root, game_length, rec_speaker=None):
    """Werewolf- and deception-type accusations *received*, per player, split by
    half of the game via the accusing utterance's line number.

    ``rec_speaker`` supplies the game timeline from :func:`load_technique_features`
    and is required to place block B on block A's scale. It is optional only so
    that older call sites keep working; without it the split falls back to
    comparing a transcript line number against an utterance count, which cuts
    about 5% early.
    """
    rows = []
    for p in sorted(acc_root.rglob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = rec.get("metadata", {})
        if not all(meta.get(k) for k in ("source", "session", "game")):
            continue
        key = ckey(meta["source"], meta["session"], meta["game"])
        n_utt = game_length.get(key)
        recs = sorted(rec_speaker.get(key, {})) if rec_speaker is not None else None
        for item in rec.get("items", []):
            line_number = item.get("line_number", 0)
            if recs:
                half = half_from_line_number(line_number, recs, n_utt)
            else:
                half = "early" if n_utt and line_number <= n_utt / 2 else "late"
            for rel in item.get("relations", []):
                if rel.get("type") not in ("werewolf", "deception"):
                    continue
                for pl in rel.get("accused", []):
                    if pl != "UNKNOWN":
                        rows.append({"key": key, "player": pl,
                                     "relation_type": rel["type"], "half": half})
    acc_df = pd.DataFrame(rows)
    assert len(acc_df) > 0, f"no accusation records found under {acc_root}"

    counts = (acc_df.groupby(["key", "player", "relation_type"]).size()
              .unstack(fill_value=0).reset_index()
              .rename(columns={"werewolf": "werewolf_count",
                               "deception": "deception_count"}))
    per_half = (acc_df.groupby(["key", "player", "relation_type", "half"]).size()
                .unstack(["relation_type", "half"], fill_value=0))
    per_half.columns = [
        f"{'werewolf_count' if r == 'werewolf' else 'deception_count'}_{h}"
        for r, h in per_half.columns]
    counts = counts.merge(per_half.reset_index(), on=["key", "player"], how="left")
    for c in ACC_FEATURES + temporal(ACC_FEATURES):
        if c not in counts:
            counts[c] = 0.0
    return counts.fillna(0.0), len(acc_df)


def load_identity_claim_features(ic_csv, game_length, rec_speaker):
    """What each player claimed about their own role: aggregate values from the
    prepared table, early/late values from the per-game records (which carry
    line numbers the table does not)."""
    ic = pd.read_csv(ic_csv)
    ic["key"] = [ckey(s, se, g) for s, se, g in zip(ic["source"], ic["session"],
                                                    ic["game"])]

    def parse_roles(v):
        try:
            return set(json.loads(v)) if isinstance(v, str) else set()
        except json.JSONDecodeError:
            return set()

    roles = ic["roles_claimed"].apply(parse_roles)
    ic["claims_werewolf"] = roles.apply(lambda s: int("Werewolf" in s))
    ic["claims_info_role"] = roles.apply(lambda s: int(bool(s & INFO_ROLES)))
    ic["n_distinct_roles_claimed_self"] = pd.to_numeric(
        ic["n_distinct_roles_claimed_self"], errors="coerce").fillna(0)
    feats = ic[["key", "player"] + IC_FEATURES]

    rows = []
    for p in sorted(ic_csv.parent.rglob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        meta = rec.get("metadata", {})
        if not all(meta.get(k) for k in ("source", "session", "game")):
            continue
        key = ckey(meta["source"], meta["session"], meta["game"])
        n_utt = game_length.get(key)
        if not n_utt:
            continue
        per = {}
        recs = sorted(rec_speaker.get(key, {}))
        for item in rec.get("items", []):
            ln = item.get("line_number")
            sp = rec_speaker.get(key, {}).get(ln)
            if sp is None:
                continue
            half = half_from_line_number(ln, recs, n_utt)
            per.setdefault((sp, half), set()).update(item.get("claimed_roles") or [])
        for (sp, half), claimed in per.items():
            rows.append({"key": key, "player": sp,
                         f"claims_werewolf_{half}": int("Werewolf" in claimed),
                         f"claims_info_role_{half}": int(bool(claimed & INFO_ROLES)),
                         f"n_distinct_roles_claimed_self_{half}": float(len(claimed))})
    per_half = (pd.DataFrame(rows).groupby(["key", "player"]).max().reset_index()
                if rows else pd.DataFrame(columns=["key", "player"]))
    feats = feats.merge(per_half, on=["key", "player"], how="left")
    for c in temporal(IC_FEATURES):
        if c not in feats:
            feats[c] = 0.0
    assert len(feats) > 0, f"no identity-claim features found at {ic_csv}"
    return feats.fillna(0.0)


# ------------------------------------------------------------------ ballot ---
def build_ballot(roster_by_key, pt_df, acc_counts, ic_feats):
    """One row per alternative: every roster member, plus 'No Werewolf'.

    Missing means zero -- a player never annotated as doing or receiving
    anything genuinely has zero counts. The circle row carries only its
    alternative-specific constant.
    """
    rows = []
    for key, info in roster_by_key.items():
        nmap = {p.strip().lower(): p for p in info["players"]}
        feats = {p: dict.fromkeys(PLAYER_FEATURES, 0.0) for p in info["players"]}
        sources = [(pt_df[pt_df["key"] == key], BLOCK_A + BLOCK_A_T),
                   (acc_counts[acc_counts["key"] == key],
                    ACC_FEATURES + temporal(ACC_FEATURES)),
                   (ic_feats[ic_feats["key"] == key],
                    IC_FEATURES + temporal(IC_FEATURES))]
        for df, cols in sources:
            for _, r in df.iterrows():
                p = nmap.get(str(r.get("speaker", r.get("player"))).strip().lower())
                if p is None:
                    continue
                for c in cols:
                    if c in r and pd.notna(r[c]):
                        feats[p][c] += float(r[c])
        base = {"key": key, "source": info["source"], "game": info["game"]}
        for p, f in feats.items():
            rows.append({**base, "player": p, **f, CIRCLE_FEATURE: 0.0})
        rows.append({**base, "player": CIRCLE_OPTION,
                     **dict.fromkeys(PLAYER_FEATURES, 0.0), CIRCLE_FEATURE: 1.0})

    ballot = pd.DataFrame(rows)
    players_only = ballot[ballot[CIRCLE_FEATURE] == 0]
    dead = [c for c in BLOCK_A + BLOCK_B if (players_only[c] != 0).sum() == 0]
    assert not dead, f"features are all-zero -- a source failed to load: {dead}"
    return ballot
