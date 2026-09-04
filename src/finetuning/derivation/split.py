"""Game-level train/validation split, stratified by end-game Werewolf count.

Spec step 7.

    python -m src.finetuning.derivation.split --write

90 train / 30 validation. Wider than the previous 99/21 on purpose: 21 validation
games could not resolve anything.

The split is written to disk as explicit game-id lists (`split.json`), so a rerun
reproduces the partitions even if the shuffling code, the seed convention or the
corpus ordering ever changes. `train.jsonl` and `val.jsonl` are derived from that
list, never from a fresh shuffle.

Stratification key
------------------
Number of Werewolves in the FINAL configuration. Night actions only ever move
cards between players - the centre is never touched by any action in this corpus -
so the number of Werewolves among the players is invariant across the night, and
the end-game count equals the dealt count. The strata are therefore 0, 1 and 2
Werewolves with 3, 77 and 40 games.

The 3-game stratum is why this is stratified at all: an unstratified draw could
easily put all three or none of them in validation.

Validation sizes per stratum are allocated by largest remainder, so they sum to
exactly 30 rather than to whatever rounding produces.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.finetuning.derivation.build_dataset import DEFAULT_OUT, build_all  # noqa: E402

DEFAULT_SEED = 1234
DEFAULT_VAL_SIZE = 30


def allocate(strata: dict, val_size: int, total: int) -> dict:
    """Largest-remainder allocation of validation slots across strata."""
    exact = {key: len(games) * val_size / total for key, games in strata.items()}
    base = {key: int(value) for key, value in exact.items()}
    short = val_size - sum(base.values())
    order = sorted(exact, key=lambda k: (exact[k] - base[k], len(strata[k])), reverse=True)
    for key in order[:short]:
        base[key] += 1
    for key, games in strata.items():
        if base[key] > len(games):
            raise ValueError("stratum %s: %d validation slots but only %d games"
                             % (key, base[key], len(games)))
    return base


def make_split(examples, val_size=DEFAULT_VAL_SIZE, seed=DEFAULT_SEED):
    strata = collections.defaultdict(list)
    for example in examples:
        strata[example["num_werewolves_end"]].append(example["game_id"])
    for key in strata:
        strata[key].sort()

    quota = allocate(strata, val_size, len(examples))
    rng = random.Random(seed)

    train, val = [], []
    for key in sorted(strata):
        games = list(strata[key])
        rng.shuffle(games)
        val.extend(games[:quota[key]])
        train.extend(games[quota[key]:])

    train.sort()
    val.sort()

    if set(train) & set(val):
        raise ValueError("train and validation overlap: %s" % (set(train) & set(val)))
    if len(train) + len(val) != len(examples):
        raise ValueError("split does not cover every game")
    if len(val) != val_size:
        raise ValueError("validation has %d games, expected %d" % (len(val), val_size))

    return train, val, strata, quota


def report(examples, train, val, strata):
    by_id = {e["game_id"]: e for e in examples}
    counts = {
        "train": collections.Counter(by_id[g]["num_werewolves_end"] for g in train),
        "val": collections.Counter(by_id[g]["num_werewolves_end"] for g in val),
    }
    keys = sorted(strata)

    print("=" * 66)
    print("SPLIT - %d train / %d validation, stratified by end-game Werewolves"
          % (len(train), len(val)))
    print("=" * 66)
    print("%-14s %8s %8s %8s %10s" % ("werewolves", "corpus", "train", "val", "val share"))
    print("-" * 66)
    for key in keys:
        total = len(strata[key])
        tr, va = counts["train"][key], counts["val"][key]
        print("%-14s %8d %8d %8d %9.1f%%"
              % (key, total, tr, va, 100.0 * va / total))
    print("-" * 66)
    print("%-14s %8d %8d %8d %9.1f%%"
          % ("TOTAL", len(examples), len(train), len(val),
             100.0 * len(val) / len(examples)))
    print()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Stratified game-level split.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--val_size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    examples = build_all()
    train, val, strata, quota = make_split(examples, args.val_size, args.seed)
    counts = report(examples, train, val, strata)

    print("validation game ids (%d):" % len(val))
    for i in range(0, len(val), 6):
        print("   " + "  ".join(val[i:i + 6]))
    print()

    if not args.write:
        print("(dry run - pass --write to emit split.json, train.jsonl, val.jsonl)")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {e["game_id"]: e for e in examples}

    split_path = out_dir / "split.json"
    split_path.write_text(json.dumps({
        "seed": args.seed,
        "val_size": args.val_size,
        "stratify_by": "num_werewolves_end",
        "stratum_quota": {str(k): v for k, v in sorted(quota.items())},
        "train_werewolf_distribution": {str(k): counts["train"][k] for k in sorted(strata)},
        "val_werewolf_distribution": {str(k): counts["val"][k] for k in sorted(strata)},
        "train": train,
        "val": val,
    }, indent=2), encoding="utf-8")
    print("wrote %s" % split_path)

    for name, ids in (("train", train), ("val", val)):
        path = out_dir / ("%s.jsonl" % name)
        with open(path, "w", encoding="utf-8") as handle:
            for game_id in ids:
                handle.write(json.dumps(by_id[game_id], ensure_ascii=False) + "\n")
        print("wrote %s  (%d games)" % (path, len(ids)))
    return 0


__all__ = ["make_split"]


if __name__ == "__main__":
    raise SystemExit(main())
