"""Correctness gate over the whole ONUW corpus.

Spec step 4. A game enters the training set only if both hold:

- the simulated final state equals `roles_ground_truth` for every player;
- the multiset of dealt cards plus centre equals final cards plus centre.

    python -m src.finetuning.derivation.gate

Failures are reported, never repaired. There is no fallback path and no way to
relax a check from the command line: a failure means the extraction is wrong, and
the fix belongs in records.py or simulate.py.

The observation checks are counted as well as failed on. A check that never runs
looks exactly like a check that always passes, so the run reports how many games
exercised each one.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.derivation.records import (  # noqa: E402
    CORPUS_DIR,
    ExtractionError,
    build_record,
    load_game,
)
from src.finetuning.derivation.simulate import (  # noqa: E402
    OBSERVATION_CHECKS,
    SimulationMismatch,
    conservation_holds,
    simulate,
)


def game_ids():
    return sorted(p.stem for p in CORPUS_DIR.glob("*.json"))


def run_gate(ids=None):
    ids = ids or game_ids()
    passed, failures = [], []
    check_counts = collections.Counter()
    check_games = collections.Counter()
    conservation_failures, final_state_failures, extraction_failures = [], [], []
    observation_failures = []

    for gid in ids:
        try:
            record = build_record(load_game(gid), gid)
        except ExtractionError as exc:
            failures.append(gid)
            extraction_failures.append((gid, str(exc)))
            continue

        log = []
        try:
            state = simulate(record, check_log=log)
        except SimulationMismatch as exc:
            failures.append(gid)
            observation_failures.append((gid, exc.check, str(exc)))
            for name in set(log):
                check_games[name] += 1
            check_counts.update(log)
            continue

        check_counts.update(log)
        for name in set(log):
            check_games[name] += 1

        diverged = [(p, state[p], record.final[p])
                    for p in record.players if state[p] != record.final[p]]
        conserved = conservation_holds(record, state)

        if diverged:
            final_state_failures.append((gid, diverged))
        if not conserved:
            before = sorted(list(record.dealt.values()) + list(record.centre))
            after = sorted(list(state.values()) + list(record.centre))
            conservation_failures.append((gid, before, after))

        if diverged or not conserved:
            failures.append(gid)
        else:
            passed.append(gid)

    return {
        "ids": ids,
        "passed": passed,
        "failed": sorted(set(failures)),
        "extraction_failures": extraction_failures,
        "observation_failures": observation_failures,
        "final_state_failures": final_state_failures,
        "conservation_failures": conservation_failures,
        "check_counts": check_counts,
        "check_games": check_games,
    }


def report(result) -> int:
    total = len(result["ids"])
    n_pass, n_fail = len(result["passed"]), len(result["failed"])

    print("=" * 74)
    print("CORRECTNESS GATE - %d games" % total)
    print("=" * 74)
    print("  PASS %d / %d" % (n_pass, total))
    print("  FAIL %d / %d" % (n_fail, total))
    print()

    print("-" * 74)
    print("BY FAILURE MODE (reported separately, a game may appear under two)")
    print("-" * 74)
    print("  extraction (record could not be built) : %d"
          % len(result["extraction_failures"]))
    print("  observation check tripped              : %d"
          % len(result["observation_failures"]))
    print("  final state != roles_ground_truth      : %d"
          % len(result["final_state_failures"]))
    print("  card conservation violated             : %d"
          % len(result["conservation_failures"]))
    print()

    if result["extraction_failures"]:
        print("-" * 74)
        print("EXTRACTION FAILURES")
        print("-" * 74)
        for gid, msg in result["extraction_failures"]:
            print("  %s" % gid)
            print("      %s" % msg)
        print()

    if result["observation_failures"]:
        print("-" * 74)
        print("OBSERVATION CHECK FAILURES")
        print("-" * 74)
        for gid, check, msg in result["observation_failures"]:
            print("  %s   invariant tripped: %s" % (gid, check))
            print("      %s" % msg)
        print()

    if result["final_state_failures"]:
        print("-" * 74)
        print("FINAL STATE DISAGREEMENTS")
        print("-" * 74)
        for gid, diverged in result["final_state_failures"]:
            print("  %s   %d player(s) diverged" % (gid, len(diverged)))
            for player, got, want in diverged:
                print("      %-9s simulation=%-13s roles_ground_truth=%s"
                      % (player, got, want))
        print()

    if result["conservation_failures"]:
        print("-" * 74)
        print("CONSERVATION FAILURES")
        print("-" * 74)
        for gid, before, after in result["conservation_failures"]:
            print("  %s" % gid)
            print("      dealt+centre: %s" % before)
            print("      final+centre: %s" % after)
        print()

    print("-" * 74)
    print("OBSERVATION CHECKS EXERCISED (proof they are not vacuous)")
    print("-" * 74)
    print("  %-24s %10s %10s" % ("check", "times run", "games"))
    for name in OBSERVATION_CHECKS:
        print("  %-24s %10d %10d"
              % (name, result["check_counts"][name], result["check_games"][name]))
    print("  %-24s %10d" % ("TOTAL", sum(result["check_counts"].values())))
    never = [n for n in OBSERVATION_CHECKS if result["check_counts"][n] == 0]
    if never:
        print()
        print("  WARNING: these checks never ran, so they prove nothing: %s" % never)
    print()

    print("=" * 74)
    if n_fail:
        print("GATE FAILED - %d game(s) must not enter the training set" % n_fail)
        print("Failing game ids: %s" % result["failed"])
    else:
        print("GATE PASSED - all %d games eligible for the training set" % total)
    print("=" * 74)
    return 1 if n_fail else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ONUW derivation correctness gate.")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write the passing game-id list as JSON.")
    args = parser.parse_args()

    result = run_gate()
    status = report(result)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"passed": result["passed"], "failed": result["failed"]},
                      handle, indent=2)
        print("wrote %s" % out)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
