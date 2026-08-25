"""Phase 1 feature construction for the RQ1 LLM surrogate (persuasion features only).

Rebuilds every feature directly from the current annotation sources -- the
enriched ``acc_targets`` and ``ic_targets`` JSONs, plus the original Lai
persuasion labels -- with ONE consistent speaker-normalisation pipeline
(:func:`utils.speakers.normalize_speaker`) applied everywhere a name is
matched to a roster. Nothing is silently dropped: every unmatched name is
recorded in the diagnostics returned alongside the feature table.

Two design points fixed relative to the Phase-0 audit of the old pipeline:

* **One source for identity claims.** Both the overall and the early/late
  identity-claim features come from the same per-item JSON records; there is
  no more aggregate-from-CSV / temporal-from-JSON split.
* **The dialogue midpoint uses every annotated utterance**, including the 38
  with an unresolvable speaker, because that is the population the thesis
  defines the midpoint over. The Rec_Id -> ordinal-position map used to place
  accusation/identity-claim line numbers is therefore built from *all*
  annotated rows, not just the ones with a resolved speaker (the Phase-0
  finding that the old ``rec_speaker`` dict silently excluded them, moving
  the block-B cut point).

The 13 candidate-specific persuasion features (raw, per player-game):

  Strategy (6):  accusation, defense, interrogation, evidence,
                 identity_declaration, call_for_action
  Enriched (7):  werewolf_accusations_made, deception_accusations_made,
                 werewolf_accusations_received, deception_accusations_received,
                 claims_werewolf, claims_tanner, claims_night_action_role

``claims_night_action_role`` = claims to Seer, Robber, Troublemaker,
Insomniac, Mason, Minion, Doppelgaenger or Drunk (own-card-changing or
information-revealing night actions; Werewolf and Tanner are excluded because
they have their own dedicated features).

Turn counts (``turns``) are tracked only as normalisation denominators, never
as a model feature.
"""

from collections import Counter, defaultdict
from bisect import bisect_right
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.speakers import normalize_speaker, is_non_player_speaker  # noqa: E402
from utils_choice.features import ckey, iter_annotation_games, PT_LABELS  # noqa: E402
from utils_choice.io import load_vote_tables                      # noqa: E402

UNKNOWN_TARGET = "UNKNOWN"
UNKNOWN_ROLE = "unknown"

STRATEGY_FEATURES = ["accusation", "defense", "interrogation", "evidence",
                     "identity_declaration", "call_for_action"]
ENRICHED_FEATURES = ["werewolf_accusations_made", "deception_accusations_made",
                     "werewolf_accusations_received", "deception_accusations_received",
                     "claims_werewolf", "claims_tanner", "claims_night_action_role"]
ALL_FEATURES_13 = STRATEGY_FEATURES + ENRICHED_FEATURES

_PT_TO_FEATURE = {
    "Accusation": "accusation", "Defense": "defense",
    "Interrogation": "interrogation", "Evidence": "evidence",
    "Identity Declaration": "identity_declaration",
    "Call for Action": "call_for_action",
}
assert set(_PT_TO_FEATURE) == set(PT_LABELS)

NIGHT_ACTION_ROLES = {"Seer", "Robber", "Troublemaker", "Insomniac", "Mason",
                      "Minion", "Doppelgaenger", "Drunk"}
ROLE_ALIASES = {"doppelganger": "Doppelgaenger", "doppelgaenger": "Doppelgaenger",
                "double werewolf": "Werewolf"}


def _fold(raw):
    d = unicodedata.normalize("NFKD", str(raw))
    return " ".join("".join(c for c in d if not unicodedata.combining(c)).split()).casefold()


def normalize_role(raw):
    """Display name for a claimed-role label, ASCII (no diacritics)."""
    key = _fold(raw)
    if not key:
        return None
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    return key.title()


def half_from_position(position, n_utt):
    return "early" if n_utt and position <= n_utt / 2 else "late"


def _dialogue_index(annot_root):
    """Per-game: dialogue length, and a Rec_Id -> ordinal-position map built
    from EVERY annotated row (speaker-resolved or not), plus the raw speaker
    text at each Rec_Id."""
    game_len, rec_to_pos, rec_to_rawspeaker, dialogues = {}, {}, {}, {}
    seen = set()
    for source, session, game_id, dialogue in iter_annotation_games(annot_root):
        k = ckey(source, session, game_id)
        if k in seen:
            continue
        seen.add(k)
        n = len(dialogue)
        game_len[k] = n
        dialogues[k] = dialogue
        pos_map, sp_map = {}, {}
        for i, u in enumerate(dialogue, start=1):
            rec = int(u.get("Rec_Id", i))
            pos_map[rec] = i
            sp_map[rec] = u.get("speaker")
        rec_to_pos[k] = pos_map
        rec_to_rawspeaker[k] = sp_map
    return game_len, rec_to_pos, rec_to_rawspeaker, dialogues


def _half_for_line(line_number, rec_to_pos, n_utt):
    """Ordinal position of a Rec_Id/line_number, then early/late by that
    position. Falls back to bisection against the sorted Rec_Id set if the
    exact key is absent (defensive; should not happen in this corpus)."""
    pos = rec_to_pos.get(line_number)
    if pos is None:
        keys = sorted(rec_to_pos)
        pos = bisect_right(keys, line_number)
    return half_from_position(pos, n_utt)


class _Accumulator:
    """One row of feature counts for one (game key, roster player)."""

    def __init__(self):
        self.d = defaultdict(float)

    def add(self, col, half, v=1.0):
        self.d[col] += v
        self.d[f"{col}_{half}"] += v


def build_feature_table(analysis_root, tables_rel, annot_root, acc_root, ic_root):
    """Returns (df, roster, nmaps, diagnostics).

    ``df`` has one row per (game key, roster player) with raw overall +
    early/late counts for the 13 persuasion features, plus ``turns``,
    ``turns_early``, ``turns_late`` (denominators). ``diagnostics`` is a dict
    of DataFrames recording every unmatched name, by source and reason.
    """
    votes, games, roster = load_vote_tables(analysis_root, tables_rel)
    nmaps = {k: {p.strip().lower(): p for p in v["players"]} for k, v in roster.items()}
    game_len, rec_to_pos, rec_to_rawspeaker, dialogues = _dialogue_index(annot_root)

    acc = defaultdict(_Accumulator)   # (key, player) -> Accumulator

    def resolve(k, raw_name):
        """Roster spelling via the ONE shared normalizer, or None."""
        if raw_name is None:
            return None
        players = list(roster.get(k, {}).get("players", []))
        norm = normalize_speaker(raw_name, player_names=players)
        if norm is None:
            return None
        return nmaps.get(k, {}).get(str(norm).strip().lower())

    diag_rows = []

    def log_unmatched(source, k, raw_name, reason):
        diag_rows.append({"source": source, "session": k[1], "game": k[2],
                          "raw_name": raw_name, "reason": reason})

    # ---------------------------------------------------------- block A ----
    for k, dialogue in dialogues.items():
        if k not in nmaps:
            continue
        n = game_len[k]
        for i, u in enumerate(dialogue, start=1):
            raw_sp = u.get("speaker")
            p = resolve(k, raw_sp)
            half = half_from_position(i, n)
            if p is None:
                if not is_non_player_speaker(raw_sp):
                    log_unmatched("original_speaker", k, raw_sp, "not_on_roster")
                continue
            row = acc[(k, p)]
            row.add("turns", half)
            for ann in (u.get("annotation") or []):
                if ann in _PT_TO_FEATURE:
                    row.add(_PT_TO_FEATURE[ann], half)

    # ---------------------------------------------------- block B: acc -----
    n_target_level = Counter()
    for pth in sorted(acc_root.rglob("*.json")):
        try:
            d = json.loads(pth.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        m = d.get("metadata", {})
        if not all(m.get(x) for x in ("source", "session", "game")):
            continue
        k = ckey(m["source"], m["session"], m["game"])
        if k not in nmaps:
            continue
        n = game_len.get(k)
        r2p = rec_to_pos.get(k, {})
        for item in d.get("items", []):
            ln = item.get("line_number", 0)
            half = _half_for_line(ln, r2p, n) if n else "late"
            accuser_raw = item.get("accuser")
            for rel in (item.get("relations") or []):
                t = rel.get("type")
                if t not in ("werewolf", "deception"):
                    continue
                for target_raw in (rel.get("accused") or []):
                    if target_raw == UNKNOWN_TARGET:
                        continue
                    n_target_level[t] += 1
                    tgt = resolve(k, target_raw)
                    if tgt is None:
                        log_unmatched("accusation_target", k, target_raw,
                                     "not_on_roster_or_ambiguous")
                    else:
                        acc[(k, tgt)].add(f"{t}_accusations_received", half)
                    acr = resolve(k, accuser_raw)
                    if acr is None:
                        log_unmatched("accusation_accuser", k, accuser_raw,
                                     "not_on_roster_or_ambiguous")
                    else:
                        acc[(k, acr)].add(f"{t}_accusations_made", half)

    # ------------------------------------------------- block B: identity ---
    ic_items = Counter()
    for pth in sorted(ic_root.rglob("*.json")):
        try:
            d = json.loads(pth.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        m = d.get("metadata", {})
        if not all(m.get(x) for x in ("source", "session", "game")):
            continue
        k = ckey(m["source"], m["session"], m["game"])
        if k not in nmaps:
            continue
        n = game_len.get(k)
        r2p = rec_to_pos.get(k, {})
        r2sp = rec_to_rawspeaker.get(k, {})
        for item in d.get("items", []):
            ic_items["items"] += 1
            claimed = item.get("claimed_roles") or []
            if not claimed:
                ic_items["empty"] += 1
                continue
            if any(_fold(r) == UNKNOWN_ROLE for r in claimed):
                ic_items["unknown"] += 1
                continue
            ic_items["resolved"] += 1
            if len(claimed) > 1:
                ic_items["multi"] += 1
            ln = item.get("line_number")
            raw_sp = r2sp.get(ln)
            p = resolve(k, raw_sp)
            half = _half_for_line(ln, r2p, n) if n else "late"
            if p is None:
                if raw_sp is None or not is_non_player_speaker(raw_sp):
                    log_unmatched("identity_claim_speaker", k, raw_sp,
                                 "not_on_roster_or_ambiguous")
                continue
            for raw_role in claimed:
                role = normalize_role(raw_role)
                if role is None:
                    continue
                if role == "Werewolf":
                    acc[(k, p)].add("claims_werewolf", half)
                elif role == "Tanner":
                    acc[(k, p)].add("claims_tanner", half)
                elif role in NIGHT_ACTION_ROLES:
                    acc[(k, p)].add("claims_night_action_role", half)
                # Villager / Hunter / Other: no content-specific feature
                # (they still counted toward pt_identity_declaration in block A).

    # --------------------------------------------------------- assemble ----
    rows = []
    for k, info in roster.items():
        for p in info["players"]:
            r = acc.get((k, p), _Accumulator()).d
            row = {"key": k, "source": k[0], "session": k[1], "game": k[2], "player": p}
            for feat in ALL_FEATURES_13 + ["turns"]:
                row[feat] = r.get(feat, 0.0)
                row[f"{feat}_early"] = r.get(f"{feat}_early", 0.0)
                row[f"{feat}_late"] = r.get(f"{feat}_late", 0.0)
            rows.append(row)
    df = pd.DataFrame(rows)

    for feat in ALL_FEATURES_13 + ["turns"]:
        assert np.allclose(df[f"{feat}_early"] + df[f"{feat}_late"], df[feat]), \
            f"early + late != overall for {feat}"
        assert (df[feat] >= 0).all(), f"negative value in {feat}"

    diagnostics = {"unresolved_names": pd.DataFrame(diag_rows),
                  "target_level_totals": n_target_level,
                  "ic_item_accounting": ic_items}
    return df, roster, nmaps, diagnostics
