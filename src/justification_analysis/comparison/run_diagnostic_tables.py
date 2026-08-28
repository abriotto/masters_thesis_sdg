"""Regenerate the DIAGNOSTIC thesis tables (10b-23) from the frozen inputs.

These tables are *not* production results. They belong to three closed strands:

  10b-14  DiMLex coverage and the `given` / `given that` sensitivity bounds
  15-18   the forced-span probe
  19-23   the rejected DiMLex-expanded hybrid

They exist so the thesis can state what was tried and why it was rejected. The
production discourse results are tables 01-05, 02b-02d, 05b-05e, 09 and 10, and
are produced by `run_final_discourse_results.py` and notebook 2 - this script
never touches them.

Why it exists: the tables above were originally written by throwaway session
code, so the files on disk had no generator in the repository. Everything here
is a thin driver over functions that were already committed
(`contingency_sensitivity`, `forced_span_summary`, `hybrid_experimental`); no
statistic is defined here for the first time, and no methodological decision is
reopened. Default behaviour is to recompute and compare against the files on
disk; nothing is overwritten without --regenerate.

    "C:/Users/annab/miniconda3/envs/sdglogs/python.exe" \
      src/justification_analysis/comparison/run_diagnostic_tables.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def find_repo_root(start: Path = None, repo_name: str = "masters_thesis_sdg") -> Path:
    current = (start or Path(__file__)).resolve()
    while current.name != repo_name:
        if current.parent == current:
            raise FileNotFoundError(f"repo root {repo_name!r} not found")
        current = current.parent
    return current


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.justification_analysis.comparison import contingency_sensitivity as cs  # noqa: E402
from src.justification_analysis.comparison import discourse_comparison as dc  # noqa: E402
from src.justification_analysis.comparison import discourse_statistics as ds  # noqa: E402
from src.justification_analysis.comparison import forced_span_summary as fs  # noqa: E402
from src.justification_analysis.comparison import hybrid_experimental as hx  # noqa: E402

ARTIFACTS = (
    REPO_ROOT / "analysis" / "cross_model" / "base" / "voting" / "prompt_v4"
    / "justification_analysis" / "discourse_parser"
)
TABLES = ARTIFACTS / "thesis_tables"
PROBE = ARTIFACTS / "forced_span_probe"
HYBRID = ARTIFACTS / "experimental_hybrid"

# Frozen input sizes, so a silently truncated file is caught before it becomes a
# quietly different table.
EXPECTED_ROWS = {
    "coverage_gap_triage.csv": 1661,
    "coverage_inspection_completed.csv": 30,
    "forced_span_predictions.csv": 531,
    "forced_span_predictions_all.csv": 1659,
    "hybrid_relations_experimental.csv": 5757,
}


def read_frozen(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    expected = EXPECTED_ROWS.get(path.name)
    if expected is not None and len(frame) != expected:
        raise RuntimeError(f"{path.name}: expected {expected} rows, got {len(frame)}")
    return frame


def compare_to_disk(frame: pd.DataFrame, path: Path, index_levels: int) -> str:
    """Column-wise comparison: numeric with a tolerance, everything else exact."""
    if not path.exists():
        return "MISSING"
    existing = pd.read_csv(
        path, encoding="utf-8-sig", index_col=list(range(index_levels))
    )
    fresh = frame.copy()
    if list(fresh.columns) != list(existing.columns):
        return f"DIFF columns +{sorted(set(fresh.columns) - set(existing.columns))} " \
               f"-{sorted(set(existing.columns) - set(fresh.columns))}"
    if len(fresh) != len(existing):
        return f"DIFF rows {len(fresh)} vs {len(existing)}"

    fresh_index = [str(i) for i in fresh.index]
    existing_index = [str(i) for i in existing.index]
    if fresh_index != existing_index:
        return "DIFF index order"

    for column in fresh.columns:
        left, right = fresh[column], existing[column]
        if pd.api.types.is_numeric_dtype(right) and pd.api.types.is_numeric_dtype(left):
            if not np.allclose(left.to_numpy(dtype=float),
                               right.to_numpy(dtype=float),
                               atol=1e-9, equal_nan=True):
                worst = np.nanmax(np.abs(left.to_numpy(dtype=float)
                                         - right.to_numpy(dtype=float)))
                return f"DIFF {column} (max abs {worst:.3g})"
        else:
            if not left.astype(str).eq(right.astype(str)).all():
                return f"DIFF {column} (text)"
    return "OK"


# ---------------------------------------------------------------------------
# 10b - manual coverage inspection, per form
# ---------------------------------------------------------------------------

def coverage_inspection_by_form(completed: pd.DataFrame) -> pd.DataFrame:
    """Counts of the 30 reviewed out-of-inventory cases, by lexical form.

    A case counts as valid only on an explicit `yes`; anything else that is not
    `no` is reported separately rather than folded into either side.
    """
    answer = completed["manual_valid_relation_missed_by_discopy"].astype(str).str.strip().str.lower()
    frame = pd.DataFrame({
        "form": completed["marker"],
        "valid": answer.eq("yes"),
        "not_valid": answer.eq("no"),
    })
    frame["blank"] = ~(frame["valid"] | frame["not_valid"])

    table = frame.groupby("form", as_index=False).agg(
        n_reviewed=("form", "size"),
        n_valid_explicit_connective=("valid", "sum"),
        n_not_valid=("not_valid", "sum"),
        n_blank_or_uncertain=("blank", "sum"),
    ).sort_values(["n_valid_explicit_connective", "n_reviewed"], ascending=False)

    total = {
        "form": "TOTAL",
        "n_reviewed": int(table["n_reviewed"].sum()),
        "n_valid_explicit_connective": int(table["n_valid_explicit_connective"].sum()),
        "n_not_valid": int(table["n_not_valid"].sum()),
        "n_blank_or_uncertain": int(table["n_blank_or_uncertain"].sum()),
    }
    table = pd.concat([table, pd.DataFrame([total])], ignore_index=True)
    for column in table.columns[1:]:
        table[column] = table[column].astype(int)
    return table.set_index("form")


# ---------------------------------------------------------------------------
# 18 - what the hybrid actually delivered against the sensitivity bounds
# ---------------------------------------------------------------------------

def hybrid_vs_sensitivity(
    sensitivity: pd.DataFrame,
    hybrid: pd.DataFrame,
    justifications: pd.DataFrame,
) -> pd.DataFrame:
    """The sensitivity bounds were an assumption; the probe measured the truth."""
    wide = sensitivity.pivot_table(
        index=["model", "decoding_group"], columns="variant",
        values="per_100_words", observed=False,
    ).rename(columns={"plausible": "plausible_sens",
                      "upper_bound": "upper_bound_sens"})

    contingency = hybrid.loc[hybrid["top_level"].eq("Contingency")]
    actual = hx.relation_rates(contingency, justifications, "hybrid").set_index(
        ["model", "decoding_group"]
    )["per_100_words"].rename("hybrid_actual")

    table = wide.join(actual)
    # The gain is computed from the unrounded rates and rounded once, at the
    # end. The original throwaway version rounded the rates first, which moved
    # 31B stochastic from 3.9 to 3.8 - immaterial to the rejected hybrid's
    # story, but there is no reason to keep the double rounding.
    table["hybrid_gain_pct"] = (
        100 * (table["hybrid_actual"] - table["original"]) / table["original"]
    ).round(1)
    for column in ("original", "plausible_sens", "upper_bound_sens", "hybrid_actual"):
        table[column] = table[column].round(3)
    return table[["original", "plausible_sens", "upper_bound_sens",
                  "hybrid_actual", "hybrid_gain_pct"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true",
                        help="overwrite the tables instead of only checking them")
    args = parser.parse_args()

    print("=" * 78)
    print("DIAGNOSTIC TABLES 10b-23 - closed strands, not production results")
    print("=" * 78)

    candidates = dc.load_discopy_candidates(ARTIFACTS / "discopy_explicit_candidates.csv")
    accepted = candidates.loc[candidates["is_connective"]].copy()
    justifications = ds.load_justification_frame(REPO_ROOT)

    gap = read_frozen(ARTIFACTS / "coverage_gap_triage.csv")
    inspection = read_frozen(ARTIFACTS / "coverage_inspection_completed.csv")
    forced_given = read_frozen(PROBE / "forced_span_predictions.csv")
    hybrid = read_frozen(HYBRID / "hybrid_relations_experimental.csv")

    sensitivity = cs.contingency_sensitivity(accepted, gap, justifications)

    tables = {
        # --- DiMLex coverage and sensitivity ------------------------------
        "10b_coverage_inspection_by_form": (
            coverage_inspection_by_form(inspection), 1),
        "11_given_forms_by_model": (
            cs.per_form_rates(gap, justifications).set_index(
                ["model", "decoding_group", "form"]), 3),
        "12_contingency_sensitivity": (
            sensitivity.set_index(["model", "decoding_group", "variant"]), 3),
        "13_contingency_ordering_check": (
            cs.ordering_check(sensitivity).set_index(
                ["decoding_group", "variant"]), 2),
        "14_four_class_profiles_sensitivity": (
            cs.four_class_profiles(accepted, gap, justifications).set_index(
                ["model", "decoding_group", "variant"]), 3),
        # --- forced-span probe --------------------------------------------
        "15_forced_span_by_form": (
            fs.acceptance_by(forced_given, ["form"]).set_index("form"), 1),
        "16_forced_span_by_model": (
            fs.acceptance_by(forced_given, ["model", "decoding_group"]).set_index(
                ["model", "decoding_group"]), 2),
        "17_forced_span_accepted_top_level": (
            fs.accepted_top_level_distribution(forced_given).set_index("form"), 1),
        "18_hybrid_actual_vs_sensitivity": (
            hybrid_vs_sensitivity(sensitivity, hybrid, justifications), 2),
        # --- rejected hybrid ----------------------------------------------
        "19_hybrid_gains_by_form": (hx.gains_by_form(hybrid), 1),
        "20_hybrid_gains_by_model": (hx.gains_by_model(hybrid), 2),
        "21_standard_vs_hybrid_rates": (
            hx.compare_standard_hybrid(
                hybrid.loc[hybrid["provenance"].eq(hx.STANDARD)], hybrid,
                justifications), 1),
        "22_hybrid_class_profiles": (hx.class_profiles(hybrid), 1),
        "23_hybrid_sense_shift": (hx.sense_shift(hybrid), 1),
    }

    print("\nrecomputed vs on disk")
    statuses = {}
    for name, (frame, levels) in tables.items():
        status = compare_to_disk(frame, TABLES / f"{name}.csv", levels)
        statuses[name] = status
        print(f"   [{status.split()[0]:4}] {name}")
        if status.startswith("DIFF"):
            print(f"          {status}")

    if args.regenerate:
        print("\n--regenerate: rewriting")
        for name, (frame, _) in tables.items():
            frame.to_csv(TABLES / f"{name}.csv", encoding="utf-8-sig")
            try:
                ds.to_latex(frame, TABLES / f"{name}.tex",
                            caption=name.replace("_", " "))
            except Exception as error:  # pragma: no cover - formatting only
                print(f"   (latex skipped for {name}: {error})")
            print(f"   wrote {name}.csv")

    n_ok = sum(1 for s in statuses.values() if s == "OK")
    print(f"\n{n_ok}/{len(statuses)} diagnostic tables reproduce from committed code.")
    return 0 if n_ok == len(statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
