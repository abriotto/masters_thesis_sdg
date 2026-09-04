"""Render a simulated GameRecord as a deterministic derivation trace.

Spec step 5. Rigid template, no phrasing variation, no hedging, no imitation of
any model's native style. Every sentence below is a fixed format string; the only
things that vary are player ids and role names. Uniform output is deliberate: it
makes verbatim regurgitation detectable later.

    python -m src.finetuning.derivation.render            # five sample traces
    python -m src.finetuning.derivation.render episode_002

Sentence forms taken verbatim from the spec's worked example: the lone-Werewolf
call, the Seer centre-peek, the Robber switch, the Troublemaker swap, and the
card-in-the-centre form. The remaining forms are not in the spec and are written
to match it exactly in structure and cadence:

- Werewolf with two players (40 games), and with no player at all (3 games);
- Seer viewing a player rather than the centre (23 games);
- Robber declining to switch (1 game, episode_031);
- Insomniac viewing their own card (89 games).

Every state-changing step states the before and after card for each affected
position. View-only steps say so explicitly.

Line wrapping
-------------
Each numbered item is rendered as ONE unwrapped line. The spec's example shows
wrapped text with hanging indentation, but that is the spec document's own
formatting: a hard wrap would put the line breaks at content-dependent positions,
so games with longer role names would break in different places. That is exactly
the surface variation the rigid template exists to avoid. Wrapping is presentation
and can be reinstated in one place here if wanted.
"""
from __future__ import annotations

from src.finetuning.derivation.records import CALL_ORDER, GameRecord

# --------------------------------------------------------------------------
# Fixed sentence forms. Nothing outside this block produces prose.
# --------------------------------------------------------------------------
NO_ACTOR = ("The {role} card is in the centre, so no player wakes for this call. "
            "No cards move.")
NO_ACTOR_WEREWOLF = ("Both Werewolf cards are in the centre, so no player wakes for "
                     "this call. No cards move.")

WEREWOLF_LONE = ("{actor} was dealt Werewolf and sees no other Werewolf among the "
                 "players, so the remaining Werewolf card is in the centre. "
                 "Viewing only; no cards move.")
WEREWOLF_PAIR = ("{first} and {second} were dealt Werewolf and see each other as the "
                 "only Werewolves among the players. Viewing only; no cards move.")

SEER_CENTRE = ("{actor} was dealt Seer and views two centre cards, which are {first} "
               "and {second}. Viewing only; no cards move.")
SEER_PLAYER = ("{actor} was dealt Seer and views the card held by {target}, which is "
               "{role}. Viewing only; no cards move.")

ROBBER_SWITCH = ("{actor} was dealt Robber and takes the card held by {target}, which "
                 "is {taken}. {actor} now holds {taken}. {target} now holds {given}.")
ROBBER_DECLINE = ("{actor} was dealt Robber and does not switch with any player. "
                  "No cards move.")

TROUBLEMAKER_SWAP = ("{actor} was dealt Troublemaker and exchanges the cards of "
                     "{first} and {second} without viewing them. {first} held {role_a} "
                     "and now holds {role_b}. {second} held {role_b} and now holds "
                     "{role_a}.")
# Appended to the standard rendering when both cards are identical, so the swap
# changes nothing. The before/after lines are kept, not suppressed: they are true,
# and dropping them would make this step read differently from every other swap.
TROUBLEMAKER_NOOP = " Both cards are {role}, so the configuration is unchanged."

INSOMNIAC_REVEAL = ("{actor} was dealt Insomniac and views their own card, which is "
                    "{role}. Viewing only; no cards move.")


class RenderError(Exception):
    """The record does not support the rendering the spec requires."""


def _step_sentence(action, state: dict) -> str:
    """Return the sentence for one call, applying the action to `state`.

    `state` is mutated for the two card-moving calls, exactly as simulate.py does,
    so the before/after cards quoted here are the real ones.
    """
    role = action.role

    if action.kind == "no_actor":
        if role == "Werewolf":
            return NO_ACTOR_WEREWOLF
        return NO_ACTOR.format(role=role)

    if action.kind == "werewolf_list":
        wolves = sorted(action.observed_players,
                        key=lambda p: int(p.replace("player", "")))
        if len(wolves) == 1:
            return WEREWOLF_LONE.format(actor=wolves[0])
        if len(wolves) == 2:
            return WEREWOLF_PAIR.format(first=wolves[0], second=wolves[1])
        raise RenderError("unsupported Werewolf count: %s" % wolves)

    if action.kind == "seer_centre":
        first, second = action.observed_roles
        return SEER_CENTRE.format(actor=action.actor, first=first, second=second)

    if action.kind == "seer_player":
        return SEER_PLAYER.format(actor=action.actor, target=action.targets[0],
                                  role=action.observed_roles[0])

    if action.kind == "robber_switch":
        actor, target = action.actor, action.targets[0]
        taken, given = state[target], state[actor]
        state[actor], state[target] = taken, given
        return ROBBER_SWITCH.format(actor=actor, target=target,
                                    taken=taken, given=given)

    if action.kind == "robber_decline":
        return ROBBER_DECLINE.format(actor=action.actor)

    if action.kind == "troublemaker_swap":
        first, second = action.targets
        role_a, role_b = state[first], state[second]
        state[first], state[second] = role_b, role_a
        sentence = TROUBLEMAKER_SWAP.format(actor=action.actor, first=first,
                                            second=second, role_a=role_a,
                                            role_b=role_b)
        if role_a == role_b:
            sentence += TROUBLEMAKER_NOOP.format(role=role_a)
        return sentence

    if action.kind == "insomniac_reveal":
        return INSOMNIAC_REVEAL.format(actor=action.actor,
                                       role=action.observed_roles[0])

    raise RenderError("unhandled action kind %r" % action.kind)


def render_derivation(record: GameRecord) -> str:
    """Render the full derivation trace for one game."""
    if [a.role for a in record.actions] != CALL_ORDER:
        raise RenderError("%s: actions are not in call order" % record.game_id)

    state = dict(record.dealt)
    lines = ["Dealt cards:"]
    for player in record.players:
        lines.append("- %s: %s" % (player, record.dealt[player]))
    lines.append("- Centre: %s" % ", ".join(record.centre))
    lines.append("")
    lines.append("Night actions, in call order:")
    for index, action in enumerate(record.actions, start=1):
        lines.append("%d. %s. %s" % (index, action.role,
                                     _step_sentence(action, state)))
    lines.append("")
    lines.append("Final configuration:")
    for player in record.players:
        lines.append("- %s: %s" % (player, state[player]))

    if state != record.final:
        raise RenderError(
            "%s: rendered state %r disagrees with roles_ground_truth %r"
            % (record.game_id, state, record.final))
    return "\n".join(lines)


def render_answer(record: GameRecord) -> str:
    """The final configuration alone, as the spec's `answer` field."""
    return "\n".join("- %s: %s" % (p, record.final[p]) for p in record.players)


def main() -> int:
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.finetuning.derivation.records import build_record, load_game
    from src.finetuning.derivation.gate import game_ids

    requested = sys.argv[1:]
    if not requested:
        # Chosen for coverage of every rendering branch, not at random.
        requested = ["episode_002", "episode_031", "episode_070", "episode_109"]
        for gid in game_ids():
            record = build_record(load_game(gid), gid)
            kinds = {a.kind for a in record.actions}
            wolves = sum(1 for r in record.dealt.values() if r == "Werewolf")
            if "seer_player" in kinds and wolves == 2 and gid not in requested:
                requested.append(gid)
                break

    labels = {
        "episode_002": "golden case from the spec",
        "episode_031": "the corpus's only Robber decline",
        "episode_070": "no-op Troublemaker swap (both cards identical)",
        "episode_109": "no Werewolf player; both Werewolf cards in the centre",
    }

    for gid in requested:
        record = build_record(load_game(gid), gid)
        note = labels.get(gid, "two Werewolves and a Seer player-peek")
        print("=" * 74)
        print("%s  -  %s" % (gid, note))
        print("=" * 74)
        print(render_derivation(record))
        print()
    return 0


__all__ = ["RenderError", "render_answer", "render_derivation"]


if __name__ == "__main__":
    raise SystemExit(main())
