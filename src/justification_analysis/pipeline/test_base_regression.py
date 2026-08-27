"""Base regression: the refactored pipeline must reproduce the frozen results.

Compares programmatically against the canonical base tables that were on disk
before the cleanup, not visually and not against numbers retyped from a
report. Any difference is a bug in the refactor until proven otherwise - the
correct response is to fix the code, never to move the expectation.

    python -m src.justification_analysis.pipeline.test_base_regression
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.justification_analysis.comparison import discourse_final as fin  # noqa: E402
from src.justification_analysis.pipeline.config import AnalysisConfig  # noqa: E402

# Headline values from the frozen discourse handoff, independent of the CSVs
# on disk. If the CSVs were ever silently regenerated wrong, these still catch
# it.
FROZEN = {
    "candidates": 14209,
    "accepted": 5504,
    "nosense": 8705,
    "justifications": 2292,
    "games": 191,
    "words": 169748,
    "top_level": {"Comparison": 1608, "Contingency": 1534,
                  "Expansion": 1513, "Temporal": 849},
    "senses": {
        "Comparison.Contrast": 1576, "Expansion.Conjunction": 1473,
        "Contingency.Cause": 1103, "Temporal.Asynchronous": 666,
        "Contingency.Condition": 431, "Temporal.Synchrony": 183,
        "Expansion.Alternative": 39, "Comparison.Concession": 32,
        "Expansion.Restatement": 1,
    },
}

results = []


def check(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def compare_frames(name, fresh: pd.DataFrame, canonical_path: Path,
                   tolerance: float = 1e-9) -> None:
    """Cell-by-cell comparison against the canonical CSV on disk."""
    if not canonical_path.exists():
        check(f"{name}: canonical table present", False, str(canonical_path))
        return
    # The canonical CSVs were written with `to_csv(path)`, i.e. WITH the index,
    # and with a BOM. Flatten the fresh frame the same way before comparing so
    # the diff is about values, not about serialisation.
    disk = pd.read_csv(canonical_path, encoding="utf-8-sig")
    fresh = fresh.reset_index() if fresh.index.names != [None] else fresh
    if disk.shape != fresh.shape:
        check(f"{name}: shape", False, f"{fresh.shape} vs {disk.shape}")
        return
    mismatches = []
    for column in disk.columns:
        if column not in fresh.columns:
            mismatches.append(f"missing column {column}")
            continue
        a, b = disk[column], fresh[column]
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a):
            if not np.allclose(a.astype(float), b.astype(float),
                               rtol=0, atol=tolerance, equal_nan=True):
                worst = np.nanmax(np.abs(a.astype(float) - b.astype(float)))
                mismatches.append(f"{column} (max abs diff {worst:.3g})")
        elif not a.astype(str).equals(b.astype(str)):
            mismatches.append(f"{column} (text)")
    check(f"{name}: reproduces the canonical table", not mismatches,
          "; ".join(mismatches[:4]) if mismatches else f"{disk.shape}")


def main() -> int:
    config = AnalysisConfig(stage="base", repo_root=REPO_ROOT)
    print("Base regression against the frozen discourse results")
    print(f"stage={config.stage} prompt={config.prompt_version}\n")

    accepted, justifications = fin.load_production_data(config)
    candidates_total = len(accepted) + FROZEN["nosense"]

    print("1. corpus and artifact inventory")
    check("accepted relations", len(accepted) == FROZEN["accepted"],
          f"{len(accepted)}")
    check("justifications", len(justifications) == FROZEN["justifications"],
          f"{len(justifications)}")
    check("games", justifications["game_id"].nunique() == FROZEN["games"],
          f"{justifications['game_id'].nunique()}")
    check("word denominator",
          int(justifications["n_words"].sum()) == FROZEN["words"],
          f"{int(justifications['n_words'].sum()):,}")

    print("\n2. relation inventory")
    top_level = accepted["top_level"].value_counts().to_dict()
    check("four top-level totals", top_level == FROZEN["top_level"],
          str({k: top_level.get(k) for k in sorted(FROZEN['top_level'])}))
    senses = accepted["raw_sense"].value_counts().to_dict()
    check("nine level-2 sense totals", senses == FROZEN["senses"],
          f"{len(senses)} senses observed")
    check("top-level totals sum to accepted",
          sum(top_level.values()) == len(accepted))

    print("\n3. the final tables, recomputed and compared cell by cell")
    tables = fin.build_final_tables(accepted, justifications)
    canonical = config.final_discourse_tables
    for name, frame in tables.items():
        compare_frames(name, frame, canonical / f"{name}.csv")

    print("\n4. stochastic / greedy separation preserved")
    f1 = tables["F1_overall_density"]
    f1 = f1.reset_index() if f1.index.names != [None] else f1
    runs = f1.set_index(["model", "decoding_group"])["n_runs"].to_dict()
    check("stochastic averages 3 runs, greedy 1",
          all(n == (3 if str(d) == "Stochastic" else 1)
              for (m, d), n in runs.items()), str(runs))
    check("greedy SD undefined",
          f1.loc[f1["decoding_group"].astype(str).eq("Greedy"),
                 "relations_per_100_words_sd"].isna().all())

    print("\n5. the bootstrap is unchanged and deterministic")
    f5 = tables["F5_bootstrap_pairwise"]
    check("30 pairwise rows", len(f5) == 30, f"{len(f5)}")
    check("intervals excluding zero",
          int(f5["ci_excludes_zero"].sum()) == 25,
          f"{int(f5['ci_excludes_zero'].sum())} of 30 (frozen value: 25)")
    again = fin.paired_game_bootstrap(accepted, justifications)
    check("bootstrap reproduces identically on re-run",
          np.allclose(again["ci_low"], f5["ci_low"])
          and np.allclose(again["ci_high"], f5["ci_high"]))

    print()
    failed = [name for name, passed in results if not passed]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        print("\nDo NOT adjust an expectation to make this pass.")
        return 1
    print("The cleanup reproduces every frozen base result exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
