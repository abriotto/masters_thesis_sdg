"""Base regression for the semantic and justification-level joint analyses.

Run after converting those modules to the stage-aware config and the manifest
freshness gate. Nothing about their statistical definitions changed, so every
exported table must reproduce byte-for-byte against the canonical CSVs already
on disk.

    python -m src.justification_analysis.pipeline.test_downstream_regression
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.justification_analysis.joint import joint_final as jf  # noqa: E402
from src.justification_analysis.joint_justification import justification_joint as kj  # noqa: E402
from src.justification_analysis.pipeline.config import AnalysisConfig  # noqa: E402
from src.justification_analysis.semantic import semantic_final as sem  # noqa: E402

# Frozen headline values, independent of the CSVs on disk.
FROZEN = {
    "justifications": 2292,
    "sentences": 8044,
    "semantic_labels": 11526,
    "accepted_relations": 5504,
    "words": 169748,
    "games": 191,
}

# Specific joint findings the report is built on. Values are the stochastic
# means as reported; tolerance is generous because these are round-tripped
# through CSV, but a real change would move them far more than this.
JOINT_LIFT = {
    ("Gemma 4 2B", "Mechanical", "Contingency"): 1.42,
    ("Gemma 4 4B", "Mechanical", "Contingency"): 1.45,
    ("Gemma 4 31B", "Mechanical", "Contingency"): 1.25,
    ("Gemma 4 2B", "ClaimComparison", "Temporal"): 2.98,
    ("Gemma 4 4B", "ClaimComparison", "Temporal"): 2.24,
    ("Gemma 4 31B", "ClaimComparison", "Temporal"): 1.30,
    ("Gemma 4 2B", "Payoff", "Contingency"): 1.01,
    ("Gemma 4 4B", "Payoff", "Contingency"): 1.08,
    ("Gemma 4 31B", "Payoff", "Contingency"): 1.31,
}

results = []


def check(name, passed, detail=""):
    results.append((name, bool(passed)))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def compare(name, fresh: pd.DataFrame, path: Path, tol=1e-9):
    if not path.exists():
        check(f"{name}: canonical present", False, str(path))
        return
    disk = pd.read_csv(path)
    fresh_flat = pd.read_csv(pd.io.common.StringIO(fresh.to_csv(index=False)))
    if disk.shape != fresh_flat.shape:
        check(f"{name}", False, f"shape {fresh_flat.shape} vs {disk.shape}")
        return
    bad = []
    for column in disk.columns:
        if column not in fresh_flat.columns:
            bad.append(f"missing {column}")
            continue
        a, b = disk[column], fresh_flat[column]
        if pd.api.types.is_numeric_dtype(b) and pd.api.types.is_numeric_dtype(a):
            if not np.allclose(a.astype(float), b.astype(float),
                               rtol=0, atol=tol, equal_nan=True):
                bad.append(column)
        elif not a.astype(str).equals(b.astype(str)):
            bad.append(f"{column} (text)")
    check(f"{name}", not bad, "; ".join(bad[:3]) if bad else f"{disk.shape}")


def main() -> int:
    config = AnalysisConfig(stage="base", repo_root=REPO_ROOT)
    print("Downstream base regression (semantic + justification-level joint)\n")

    # -- semantic ----------------------------------------------------------
    print("1. semantic layer")
    data = sem.load_annotations(config=config)
    check("justifications", len(data["justifications"]) == FROZEN["justifications"],
          f"{len(data['justifications'])}")
    check("sentences", len(data["sentences"]) == FROZEN["sentences"],
          f"{len(data['sentences'])}")
    check("semantic assignments",
          len(data["labels"]) == FROZEN["semantic_labels"],
          f"{len(data['labels'])}")

    semantic_tables = sem.build_final_tables(data, REPO_ROOT)
    semantic_dir = sem.final_tables_dir(config)
    for name, frame in semantic_tables.items():
        compare(f"semantic {name}", frame, semantic_dir / f"{name}.csv")

    # -- joint -------------------------------------------------------------
    print("\n2. joint layer, loaded through the freshness gate")
    layers = jf.load_layers(config=config)
    check("candidate table came through the gate",
          "manifest" in layers and layers["manifest"]["corpus"]["fingerprint"],
          layers["manifest"]["corpus"]["fingerprint"][:16] + "...")
    alignment = jf.align_relations(layers)
    sentences = jf.build_joint_sentences(layers, alignment)
    metadata = kj.load_justification_metadata(config=config)
    justifications = kj.build_justification_frame(
        layers, alignment, sentences, metadata)

    check("accepted discourse relations",
          len(alignment["aligned"]) == FROZEN["accepted_relations"],
          f"{len(alignment['aligned'])}")
    check("nothing unaligned", len(alignment["unaligned"]) == 0)
    check("canonical sentences", len(sentences) == FROZEN["sentences"],
          f"{len(sentences)}")
    check("justifications", len(justifications) == FROZEN["justifications"],
          f"{len(justifications)}")
    check("WORD_PATTERN tokens",
          int(justifications["n_words"].sum()) == FROZEN["words"],
          f"{int(justifications['n_words'].sum()):,}")
    check("games", justifications["game_id"].nunique() == FROZEN["games"])

    joint_tables = kj.build_final_tables(layers, alignment, sentences,
                                         justifications)
    joint_dir = kj.final_tables_dir(config)
    for name, frame in joint_tables.items():
        compare(f"joint {name}", frame, joint_dir / f"{name}.csv")

    # -- the specific findings the report rests on -------------------------
    print("\n3. headline joint findings unchanged")
    lift = joint_tables["K4b_joint_prevalence_lift_summary"]
    lift = lift[lift["decoding_group"].astype(str).eq("Stochastic")]
    for (model, category, relation), expected in JOINT_LIFT.items():
        row = lift[lift["model"].astype(str).eq(model)
                   & lift["semantic_category"].astype(str).eq(category)
                   & lift["discourse_relation"].astype(str).eq(relation)]
        observed = float(row["mean_lift"].iloc[0]) if len(row) else float("nan")
        check(f"lift {category} x {relation} [{model.replace('Gemma 4 ', '')}]",
              abs(observed - expected) < 0.005,
              f"{observed:.2f} (frozen {expected:.2f})")

    print("\n4. stochastic / greedy separation")
    check("no joint table pools decodings",
          all(set(t["decoding_group"].astype(str)) <= {"Stochastic", "Greedy"}
              for t in joint_tables.values()
              if "decoding_group" in t.columns))
    summary = joint_tables["K2b_conditional_prevalence_summary"]
    check("stochastic averages 3 runs, greedy 1",
          set(summary.groupby("decoding_group", observed=True)["n_runs"]
              .unique().apply(tuple).to_dict().values()) == {(3,), (1,)})

    print()
    failed = [n for n, ok in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED:")
        for name in failed:
            print(f"  - {name}")
        print("\nDo NOT update a frozen result to make this pass.")
        return 1
    print("Semantic and joint base results reproduce exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
