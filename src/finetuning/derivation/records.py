"""Build a structured night-action record from one raw ONUW episode.

Spec step 2. Two rules drive the whole module:

- **Actors come from `roles_assigned`, never from who speaks.** A role whose card
  is in the centre has no actor and no wake-up. The players' own natural-language
  statements of intent are model-generated and inconsistently phrased, so they are
  never parsed.
- **Choices come from the Moderator's private confirmation**, keyed by the
  recipient in `visible_to`.

`visible_to` is polymorphic in this corpus: the string `"all"`, a bare player id
string, or a list of player ids. The private test is therefore `!= "all"`, and both
scalar and list forms are normalised to a list. The list form occurs only for the
Werewolf call.

Anything that does not match a known template, or an actor with no confirmation
where one is required, raises `ExtractionError`. There are deliberately no
fallbacks: a mismatch means the extraction is wrong, and that is what needs fixing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO_ROOT / "data" / "raw" / "jin2024_onuw" / "gpt4_dataset"

# Fixed by the rules of the game, not by the corpus.
CALL_ORDER = ["Werewolf", "Seer", "Robber", "Troublemaker", "Insomniac"]

# Roles that wake and therefore need a confirmation when dealt to a player.
# Villager never wakes.
NON_WAKING_ROLES = {"Villager"}

PLAYER_RE = re.compile(r"player\d+")

# The eight templates found by the step 1 inventory over all 120 games. Ordered
# most specific first; `_classify` takes the first match and raises if none do.
TEMPLATES = [
    ("werewolf_list", re.compile(
        r"^All werewolves in the game: (?P<players>player\d+(?:, player\d+)*)\.$")),
    ("seer_centre", re.compile(
        r"^The two roles you checked in role pool are: (?P<a>\w+), (?P<b>\w+)\.$")),
    ("seer_player", re.compile(
        r"^The role of (?P<target>player\d+) is (?P<role>\w+)\.$")),
    ("robber_switch", re.compile(
        r"^You switched your role with (?P<target>player\d+), "
        r"and your new role is (?P<role>\w+)\.$")),
    ("robber_decline", re.compile(
        r"^You did not switch with other player, so you remain your role\.$")),
    # Note: no trailing period on this one, in the corpus as well as here.
    ("troublemaker_swap", re.compile(
        r"^You successfully swapped roles between (?P<a>player\d+) "
        r"and (?P<b>player\d+)$")),
    ("insomniac_reveal", re.compile(
        r"^Your final role is (?P<role>\w+)\.$")),
]

# Which message kinds each waking role may produce.
KINDS_FOR_ROLE = {
    "Werewolf": {"werewolf_list"},
    "Seer": {"seer_centre", "seer_player"},
    "Robber": {"robber_switch", "robber_decline"},
    "Troublemaker": {"troublemaker_swap"},
    "Insomniac": {"insomniac_reveal"},
}

NIGHT_END_MARKER = "Night phase ends."
GAME_OVER_MARKER = "Game over."


class ExtractionError(Exception):
    """The episode does not match the structure the spec describes."""


@dataclass
class Action:
    """One entry in the night call order.

    `kind == "no_actor"` means the role's card is in the centre, so no player
    wakes. `observed_*` records what the Moderator told the actor; it is used as a
    consistency check against the simulation, never as an input to it.
    """

    role: str
    actor: Optional[str]
    kind: str
    targets: list = field(default_factory=list)
    observed_players: list = field(default_factory=list)
    observed_roles: list = field(default_factory=list)


@dataclass
class GameRecord:
    game_id: str
    players: list
    dealt: dict
    centre: list
    actions: list
    final: dict
    day_transcript: list


def load_game(game_id: str) -> dict:
    """Load one raw episode. utf-8 is explicit: several episodes contain
    characters the Windows default codepage cannot decode."""
    path = CORPUS_DIR / ("%s.json" % game_id)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def recipients(visible_to: Any) -> list:
    """Normalise the polymorphic `visible_to` field to a list of player ids."""
    if isinstance(visible_to, str):
        return [] if visible_to == "all" else [visible_to]
    if isinstance(visible_to, list):
        return list(visible_to)
    raise ExtractionError(
        "visible_to has unexpected type %s: %r" % (type(visible_to).__name__, visible_to))


def is_private(message: dict) -> bool:
    """The private test, per the corpus rather than per the spec's description:
    'not all', covering both the bare-string and list forms."""
    return message["visible_to"] != "all"


def _classify(content: str):
    for kind, pattern in TEMPLATES:
        match = pattern.match(content)
        if match:
            return kind, match
    raise ExtractionError("unrecognised private Moderator message: %r" % content)


def _parse_private_messages(game: dict, game_id: str) -> dict:
    """Return {kind: [(recipients, match), ...]} for private Moderator messages."""
    parsed: dict = {}
    for message in game["messages"]:
        if message["agent_name"] != "Moderator" or not is_private(message):
            continue
        kind, match = _classify(message["content"])
        parsed.setdefault(kind, []).append((recipients(message["visible_to"]), match))
    return parsed


def _one(parsed: dict, kinds: set, role: str, actor: str, game_id: str):
    """Exactly one confirmation of an allowed kind, addressed to the actor."""
    hits = [(kind, rec, match)
            for kind in kinds
            for rec, match in parsed.get(kind, [])]
    if len(hits) != 1:
        raise ExtractionError(
            "%s: %s dealt to %s but found %d confirmation(s) of kind %s; "
            "the corpus has no decline template for this role, so this is not "
            "something to branch on"
            % (game_id, role, actor, len(hits), sorted(kinds)))
    kind, rec, match = hits[0]
    if rec != [actor]:
        raise ExtractionError(
            "%s: %s confirmation addressed to %r but %s was dealt %s"
            % (game_id, role, rec, actor, role))
    return kind, match


def build_record(game: dict, game_id: str) -> GameRecord:
    evaluation = game["evaluation"]
    dealt = dict(evaluation["roles_assigned"])
    final = dict(evaluation["roles_ground_truth"])
    centre = list(evaluation["role_pool"])
    players = sorted(dealt, key=lambda p: int(p.replace("player", "")))

    if set(dealt) != set(final):
        raise ExtractionError(
            "%s: roles_assigned and roles_ground_truth cover different players" % game_id)

    by_role: dict = {}
    for player, role in dealt.items():
        by_role.setdefault(role, []).append(player)

    parsed = _parse_private_messages(game, game_id)
    actions = []

    for role in CALL_ORDER:
        actors = sorted(by_role.get(role, []),
                        key=lambda p: int(p.replace("player", "")))

        if not actors:
            # Card is in the centre; nobody wakes for this call.
            actions.append(Action(role=role, actor=None, kind="no_actor"))
            continue

        if role == "Werewolf":
            # One list message names every Werewolf, however many there are.
            hits = parsed.get("werewolf_list", [])
            if len(hits) != 1:
                raise ExtractionError(
                    "%s: %d Werewolf player(s) but %d werewolf_list message(s)"
                    % (game_id, len(actors), len(hits)))
            rec, match = hits[0]
            named = match.group("players").split(", ")
            if sorted(named) != sorted(actors):
                raise ExtractionError(
                    "%s: werewolf_list names %s but %s were dealt Werewolf"
                    % (game_id, named, actors))
            if sorted(rec) != sorted(actors):
                raise ExtractionError(
                    "%s: werewolf_list visible to %s but %s were dealt Werewolf"
                    % (game_id, rec, actors))
            actions.append(Action(role=role, actor=actors[0], kind="werewolf_list",
                                  observed_players=named))
            continue

        if len(actors) > 1:
            raise ExtractionError(
                "%s: %s dealt to more than one player: %s" % (game_id, role, actors))
        actor = actors[0]
        kind, match = _one(parsed, KINDS_FOR_ROLE[role], role, actor, game_id)

        if kind == "seer_centre":
            actions.append(Action(role=role, actor=actor, kind=kind,
                                  observed_roles=[match.group("a"), match.group("b")]))
        elif kind == "seer_player":
            actions.append(Action(role=role, actor=actor, kind=kind,
                                  targets=[match.group("target")],
                                  observed_roles=[match.group("role")]))
        elif kind == "robber_switch":
            actions.append(Action(role=role, actor=actor, kind=kind,
                                  targets=[match.group("target")],
                                  observed_roles=[match.group("role")]))
        elif kind == "robber_decline":
            actions.append(Action(role=role, actor=actor, kind=kind))
        elif kind == "troublemaker_swap":
            actions.append(Action(role=role, actor=actor, kind=kind,
                                  targets=[match.group("a"), match.group("b")]))
        elif kind == "insomniac_reveal":
            actions.append(Action(role=role, actor=actor, kind=kind,
                                  observed_roles=[match.group("role")]))
        else:  # pragma: no cover - KINDS_FOR_ROLE and TEMPLATES are in sync
            raise ExtractionError("%s: unhandled kind %r" % (game_id, kind))

    # Every private message must have been consumed by exactly one action.
    consumed = sum(1 for a in actions if a.kind != "no_actor")
    available = sum(len(v) for v in parsed.values())
    if consumed != available:
        raise ExtractionError(
            "%s: %d private message(s) present but %d consumed by actions"
            % (game_id, available, consumed))

    return GameRecord(
        game_id=game_id,
        players=players,
        dealt=dealt,
        centre=centre,
        actions=actions,
        final=final,
        day_transcript=extract_day_transcript(game),
    )


def extract_day_transcript(game: dict) -> list:
    """Public messages from the end of the night phase onward.

    Excludes the final 'Game over ...' line: it is marked `visible_to == "all"` but
    it names the eliminated player and their role, i.e. it is the label. This
    matches the exclusion preprocess_jin.py already makes.

    `thought`, `belief` and `strategy` are never read here, so they cannot leak.
    """
    out = []
    in_day = False
    for message in game["messages"]:
        if not in_day:
            if message["agent_name"] == "Moderator" and \
                    message["content"].startswith(NIGHT_END_MARKER):
                in_day = True
            continue
        if message["visible_to"] != "all":
            continue
        if message["content"].startswith(GAME_OVER_MARKER):
            continue
        out.append({"speaker": message["agent_name"],
                    "content": message["content"],
                    "turn": message["turn"]})
    return out


__all__ = [
    "CALL_ORDER",
    "CORPUS_DIR",
    "Action",
    "ExtractionError",
    "GameRecord",
    "build_record",
    "extract_day_transcript",
    "is_private",
    "load_game",
    "recipients",
]
