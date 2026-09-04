"""Unit tests for the ONUW night-action record builder and simulator.

    python -m src.finetuning.derivation.test_simulation

Covers the golden case from the spec (episode_002), the single Robber-decline
game in the corpus (episode_031), the three games with no Werewolf player, and
the two cases that must raise rather than be accommodated.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.derivation.records import (  # noqa: E402
    CORPUS_DIR,
    Action,
    ExtractionError,
    GameRecord,
    build_record,
    load_game,
)
from src.finetuning.derivation.simulate import (  # noqa: E402
    SimulationMismatch,
    simulate,
)


# --------------------------------------------------------------------------
# The golden case, stated literally from the spec rather than read from disk,
# so the simulator is tested independently of the extraction.
# --------------------------------------------------------------------------
GOLDEN_DEALT = {
    "player1": "Troublemaker",
    "player2": "Werewolf",
    "player3": "Seer",
    "player4": "Robber",
    "player5": "Villager",
}
GOLDEN_CENTRE = ["Insomniac", "Werewolf", "Villager"]
GOLDEN_FINAL = {
    "player1": "Troublemaker",
    "player2": "Seer",
    "player3": "Robber",
    "player4": "Werewolf",
    "player5": "Villager",
}


def check(label, got, want):
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + label)
    if not ok:
        print("          got : %r" % (got,))
        print("          want: %r" % (want,))
    return ok


def test_golden_simulator_only():
    """Spec golden case, hand-built actions, no corpus access."""
    actions = [
        Action(role="Werewolf", actor="player2", kind="werewolf_list",
               targets=[], observed_players=["player2"], observed_roles=[]),
        Action(role="Seer", actor="player3", kind="seer_centre",
               targets=[], observed_players=[], observed_roles=["Insomniac", "Villager"]),
        Action(role="Robber", actor="player4", kind="robber_switch",
               targets=["player2"], observed_players=[], observed_roles=["Werewolf"]),
        Action(role="Troublemaker", actor="player1", kind="troublemaker_swap",
               targets=["player2", "player3"], observed_players=[], observed_roles=[]),
        Action(role="Insomniac", actor=None, kind="no_actor",
               targets=[], observed_players=[], observed_roles=[]),
    ]
    record = GameRecord(
        game_id="golden",
        players=["player1", "player2", "player3", "player4", "player5"],
        dealt=dict(GOLDEN_DEALT),
        centre=list(GOLDEN_CENTRE),
        actions=actions,
        final=dict(GOLDEN_FINAL),
        day_transcript=[],
    )
    return check("golden case simulates to roles_ground_truth",
                 simulate(record), GOLDEN_FINAL)


def test_golden_from_corpus():
    """episode_002 is the corpus game the spec's golden case describes."""
    ok = True
    record = build_record(load_game("episode_002"), "episode_002")
    ok &= check("episode_002 dealt", record.dealt, GOLDEN_DEALT)
    ok &= check("episode_002 centre", record.centre, GOLDEN_CENTRE)
    ok &= check("episode_002 final (from roles_ground_truth)", record.final, GOLDEN_FINAL)
    ok &= check(
        "episode_002 action kinds in call order",
        [(a.role, a.kind) for a in record.actions],
        [("Werewolf", "werewolf_list"), ("Seer", "seer_centre"),
         ("Robber", "robber_switch"), ("Troublemaker", "troublemaker_swap"),
         ("Insomniac", "no_actor")],
    )
    seer = [a for a in record.actions if a.role == "Seer"][0]
    ok &= check("episode_002 Seer peeked Insomniac, Villager",
                seer.observed_roles, ["Insomniac", "Villager"])
    robber = [a for a in record.actions if a.role == "Robber"][0]
    ok &= check("episode_002 Robber target", robber.targets, ["player2"])
    tm = [a for a in record.actions if a.role == "Troublemaker"][0]
    ok &= check("episode_002 Troublemaker targets", tm.targets, ["player2", "player3"])
    ok &= check("episode_002 simulates to ground truth", simulate(record), GOLDEN_FINAL)
    return ok


def test_robber_decline_episode_031():
    """The only Robber decline in the corpus. One game cannot be left to the gate."""
    ok = True
    record = build_record(load_game("episode_031"), "episode_031")
    robber = [a for a in record.actions if a.role == "Robber"]
    ok &= check("episode_031 has exactly one Robber action", len(robber), 1)
    if not robber:
        return False
    ok &= check("episode_031 Robber kind is robber_decline",
                robber[0].kind, "robber_decline")
    ok &= check("episode_031 Robber has no target", robber[0].targets, [])
    ok &= check("episode_031 Robber keeps the Robber card",
                simulate(record)[robber[0].actor], "Robber")
    ok &= check("episode_031 simulates to ground truth", simulate(record), record.final)
    return ok


def test_no_werewolf_player_games():
    """episodes 109/110/111 have both Werewolf cards in the centre."""
    ok = True
    for gid in ("episode_109", "episode_110", "episode_111"):
        record = build_record(load_game(gid), gid)
        ww = [a for a in record.actions if a.role == "Werewolf"]
        ok &= check("%s Werewolf call has no actor" % gid,
                    [(a.kind, a.actor) for a in ww], [("no_actor", None)])
        ok &= check("%s both Werewolf cards in centre" % gid,
                    record.centre.count("Werewolf"), 2)
        ok &= check("%s simulates to ground truth" % gid, simulate(record), record.final)
    return ok


def test_troublemaker_decline_raises():
    """No decline template exists for the Troublemaker. Raise, do not branch."""
    game = load_game("episode_002")
    game["messages"] = [
        m for m in game["messages"]
        if not m["content"].startswith("You successfully swapped roles between")
    ]
    try:
        build_record(game, "episode_002_tm_removed")
    except ExtractionError as exc:
        print("  PASS  Troublemaker with no swap message raises ExtractionError")
        print("          %s" % exc)
        return True
    print("  FAIL  Troublemaker with no swap message did not raise")
    return False


def test_observation_disagreement_raises():
    """A Seer player-peek is a check on the simulation, never an input to it."""
    record = build_record(load_game("episode_004"), "episode_004")
    seer = [a for a in record.actions if a.role == "Seer"][0]
    if seer.kind != "seer_player":
        print("  FAIL  episode_004 Seer is not a player-peek; test needs a new fixture")
        return False
    seer.observed_roles = ["Tanner"]  # a role not dealt in this game at all
    try:
        simulate(record)
    except SimulationMismatch as exc:
        print("  PASS  Seer peek disagreeing with state raises SimulationMismatch")
        print("          %s" % exc)
        return True
    print("  FAIL  corrupted Seer peek did not raise")
    return False


def test_insomniac_disagreement_raises():
    """Same for the Insomniac reveal, on a game that has an Insomniac actor."""
    gid = None
    for candidate in ("episode_%03d" % i for i in range(1, 121)):
        try:
            rec = build_record(load_game(candidate), candidate)
        except (ExtractionError, FileNotFoundError):
            continue
        ins = [a for a in rec.actions if a.role == "Insomniac"]
        if ins and ins[0].actor is not None:
            gid, record, action = candidate, rec, ins[0]
            break
    if gid is None:
        print("  FAIL  no game with an Insomniac actor found")
        return False
    action.observed_roles = ["Tanner"]
    try:
        simulate(record)
    except SimulationMismatch as exc:
        print("  PASS  Insomniac reveal disagreeing with state raises "
              "SimulationMismatch (%s)" % gid)
        print("          %s" % exc)
        return True
    print("  FAIL  corrupted Insomniac reveal did not raise")
    return False


def main():
    print("corpus: %s" % CORPUS_DIR)
    tests = [
        ("golden case, simulator only", test_golden_simulator_only),
        ("golden case, from corpus (episode_002)", test_golden_from_corpus),
        ("Robber decline (episode_031)", test_robber_decline_episode_031),
        ("no Werewolf player (episodes 109-111)", test_no_werewolf_player_games),
        ("Troublemaker decline raises", test_troublemaker_decline_raises),
        ("Seer peek disagreement raises", test_observation_disagreement_raises),
        ("Insomniac reveal disagreement raises", test_insomniac_disagreement_raises),
    ]
    results = []
    for name, fn in tests:
        print("\n== %s" % name)
        results.append((name, bool(fn())))
    print("\n" + "=" * 70)
    failed = [n for n, ok in results if not ok]
    for name, ok in results:
        print("%-45s %s" % (name, "PASS" if ok else "FAIL"))
    print("=" * 70)
    print("%d/%d test groups passed" % (len(results) - len(failed), len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
