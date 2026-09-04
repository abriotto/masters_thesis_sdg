"""Simulate the ONUW night phase from a GameRecord.

Spec step 3. Call order is fixed: Werewolf, Seer, Robber, Troublemaker, Insomniac.

Two subtleties the spec singles out, both implemented here:

1. The actor is the player **dealt** that role, even if their card has since moved.
   A Troublemaker who was robbed still performs the Troublemaker action. This falls
   out of taking actors from `roles_assigned` in records.py.
2. Actions apply to whatever cards occupy the positions **at that moment**. Every
   read below is from `state`, never from `dealt`, with the single exception of the
   Werewolf call, which the rules define over the deal.

Observations are checks, not inputs
-----------------------------------
The Seer player-peek, the Robber's view of their new card and the Insomniac reveal
all state a card the Moderator showed the actor. The simulation never consumes
these to decide state; it computes state independently and then asserts agreement.
A disagreement is a hard failure - it means the extraction or the call order is
wrong - and raises SimulationMismatch rather than being reconciled.

Pass a list as `check_log` to record which observation checks actually executed.
A check that never runs is indistinguishable from a check that always passes, so
the gate reports these counts rather than trusting the absence of failures.
"""
from __future__ import annotations

from typing import Optional

from src.finetuning.derivation.records import CALL_ORDER, GameRecord

# The five observation checks, named so the gate can count them and so a
# SimulationMismatch can say which one tripped.
CHECK_WEREWOLF_LIST = "werewolf_list"
CHECK_CENTRE_PEEK = "seer_centre_multiset"
CHECK_SEER_PEEK = "seer_player_peek"
CHECK_ROBBER_DIRECTION = "robber_direction"
CHECK_INSOMNIAC_REVEAL = "insomniac_reveal"

OBSERVATION_CHECKS = [
    CHECK_INSOMNIAC_REVEAL,
    CHECK_SEER_PEEK,
    CHECK_ROBBER_DIRECTION,
    CHECK_WEREWOLF_LIST,
    CHECK_CENTRE_PEEK,
]


class SimulationMismatch(Exception):
    """A Moderator observation contradicts the simulated state.

    `check` names the invariant that tripped, so the gate can report it
    separately from a plain final-state disagreement.
    """

    def __init__(self, check: str, message: str):
        super().__init__(message)
        self.check = check


def simulate(record: GameRecord, check_log: Optional[list] = None) -> dict:
    """Return the final `player -> role` state after the night phase."""
    state = dict(record.dealt)

    def ran(name):
        if check_log is not None:
            check_log.append(name)

    seen = [action.role for action in record.actions]
    if seen != CALL_ORDER:
        raise SimulationMismatch(
            "call_order", "%s: actions are not in call order: %s" % (record.game_id, seen))

    for action in record.actions:
        if action.kind == "no_actor":
            continue

        elif action.kind == "werewolf_list":
            # View-only, and defined over the deal rather than current state.
            dealt_wolves = sorted(p for p, r in record.dealt.items() if r == "Werewolf")
            ran(CHECK_WEREWOLF_LIST)
            if sorted(action.observed_players) != dealt_wolves:
                raise SimulationMismatch(
                    CHECK_WEREWOLF_LIST,
                    "%s: Werewolf call showed %s but %s were dealt Werewolf"
                    % (record.game_id, action.observed_players, dealt_wolves))

        elif action.kind == "seer_centre":
            # View-only. The centre is untouched by every action in this corpus,
            # so the peeked pair must be a sub-multiset of it. Multiset and not
            # set: centres such as ('Villager', 'Villager', 'Werewolf') exist, so
            # a set test would wrongly accept a doubled peek.
            remaining = list(record.centre)
            ran(CHECK_CENTRE_PEEK)
            for role in action.observed_roles:
                if role not in remaining:
                    raise SimulationMismatch(
                        CHECK_CENTRE_PEEK,
                        "%s: Seer peeked %s in the centre but the centre is %s"
                        % (record.game_id, action.observed_roles, record.centre))
                remaining.remove(role)

        elif action.kind == "seer_player":
            # View-only. Read from state at this point, not from dealt.
            target = action.targets[0]
            observed = action.observed_roles[0]
            ran(CHECK_SEER_PEEK)
            if state[target] != observed:
                raise SimulationMismatch(
                    CHECK_SEER_PEEK,
                    "%s: Seer was shown %s holding %s but the simulation has %s"
                    % (record.game_id, target, observed, state[target]))

        elif action.kind == "robber_switch":
            actor, target = action.actor, action.targets[0]
            taken = state[target]
            state[target] = state[actor]
            state[actor] = taken
            observed = action.observed_roles[0]
            ran(CHECK_ROBBER_DIRECTION)
            if state[actor] != observed:
                raise SimulationMismatch(
                    CHECK_ROBBER_DIRECTION,
                    "%s: Robber %s was shown their new role as %s but the "
                    "simulation gives %s"
                    % (record.game_id, actor, observed, state[actor]))

        elif action.kind == "robber_decline":
            pass  # No cards move.

        elif action.kind == "troublemaker_swap":
            first, second = action.targets
            state[first], state[second] = state[second], state[first]

        elif action.kind == "insomniac_reveal":
            actor = action.actor
            observed = action.observed_roles[0]
            ran(CHECK_INSOMNIAC_REVEAL)
            if state[actor] != observed:
                raise SimulationMismatch(
                    CHECK_INSOMNIAC_REVEAL,
                    "%s: Insomniac %s was shown %s but the simulation gives %s"
                    % (record.game_id, actor, observed, state[actor]))

        else:
            raise SimulationMismatch(
                "unhandled_kind",
                "%s: unhandled action kind %r" % (record.game_id, action.kind))

    return state


def conservation_holds(record: GameRecord, state: dict) -> bool:
    """Spec step 4's second condition: cards are conserved."""
    before = sorted(list(record.dealt.values()) + list(record.centre))
    after = sorted(list(state.values()) + list(record.centre))
    return before == after


__all__ = [
    "OBSERVATION_CHECKS",
    "SimulationMismatch",
    "conservation_holds",
    "simulate",
]
