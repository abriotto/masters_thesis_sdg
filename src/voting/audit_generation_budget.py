"""Audit every voting run for truncation, thought presence, and schema validity.

The generation cap was escalated during the base runs whenever a game truncated,
so the caps recorded across a run directory are not uniform. That is defensible
only if the FINAL corpus contains no truncated generation, and only if the
finetuned arm - which was given a higher cap because it thinks longer - is
likewise free of truncation. Neither claim is safe to assert without counting.

Reports per run directory:

  games        files found
  caps         the distinct max_new_tokens values recorded, with counts
  max/p95      output token counts, to show how much headroom the cap had
  trunc        generations reaching 95% of their OWN cap (the truncation proxy;
               a generation stopped by the cap sits at or just under it)
  no_thought   games with an empty internal_thoughts field
  invalid      games failing the voting schema validation

Run with no arguments to audit everything under results/voting.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Optional

TRUNCATION_RATIO = 0.95


def find_repo_root(start: Optional[Path] = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("repo root not found")


def audit_run(run_dir: Path) -> dict[str, Any]:
    caps: collections.Counter[Any] = collections.Counter()
    outputs: list[int] = []
    truncated: list[str] = []
    no_thought = invalid = 0

    for path in sorted(run_dir.rglob("*.json")):
        with path.open(encoding="utf-8") as fh:
            record = json.load(fh)

        cap = record.get("max_new_tokens")
        caps[cap] += 1

        debug = record.get("debug_info") or {}
        produced = debug.get("output_token_count", record.get("output_token_count"))
        if produced is not None:
            outputs.append(produced)
            if cap and produced >= TRUNCATION_RATIO * cap:
                truncated.append(f"{path.name} (cap {cap}, out {produced})")

        if not record.get("internal_thoughts"):
            no_thought += 1
        if not (record.get("validation") or {}).get("is_valid", True):
            invalid += 1

    outputs.sort()
    return {
        "games": sum(caps.values()),
        "caps": dict(sorted(caps.items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "max_out": max(outputs) if outputs else None,
        "p95_out": outputs[min(len(outputs) - 1, int(0.95 * len(outputs)))] if outputs else None,
        "median_out": int(statistics.median(outputs)) if outputs else None,
        "truncated": truncated,
        "no_thought": no_thought,
        "invalid": invalid,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results/voting")
    parser.add_argument("--pattern", default="prompt_*",
                        help="run-directory glob under each model directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_repo_root() / args.results_dir
    run_dirs = sorted(
        d for d in root.glob(f"*/{args.pattern}") if d.is_dir() and any(d.rglob("*.json"))
    )
    if not run_dirs:
        raise SystemExit(f"no run directories under {root}")

    total_truncated = 0
    header = f"{'run':<74}{'n':>5}{'maxOut':>8}{'p95':>7}{'trunc':>7}{'noThgt':>8}{'inval':>7}"
    print(header)
    print("-" * len(header))
    for run_dir in run_dirs:
        stats = audit_run(run_dir)
        total_truncated += len(stats["truncated"])
        label = str(run_dir.relative_to(root))
        if len(label) > 72:
            label = "..." + label[-69:]
        print(
            f"{label:<74}{stats['games']:>5}{stats['max_out'] or 0:>8}"
            f"{stats['p95_out'] or 0:>7}{len(stats['truncated']):>7}"
            f"{stats['no_thought']:>8}{stats['invalid']:>7}"
        )
        if len(stats["caps"]) > 1:
            print(f"    caps: {stats['caps']}")
        for entry in stats["truncated"][:5]:
            print(f"    TRUNCATED: {entry}")

    print()
    if total_truncated:
        print(f"*** {total_truncated} generation(s) within {TRUNCATION_RATIO:.0%} of cap - "
              "rerun those games at a higher --max_new_tokens ***")
    else:
        print(f"No generation reached {TRUNCATION_RATIO:.0%} of its cap in any run. "
              "The corpus is free of truncation and the caps are reportable as non-binding.")


if __name__ == "__main__":
    main()
