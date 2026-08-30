"""RQ3: what changed in stated justification structure from BASE to fine-tuned.

Runs the frozen RQ2 discourse pipeline unchanged over the fine-tuned corpus and
reports within-model, game-matched BASE-FT differences. Stochastic runs only -
the fine-tuned models were never run greedily.

    "C:/Users/annab/miniconda3/envs/sdglogs/python.exe" \
      src/justification_analysis/comparison/run_ft_discourse_contrast.py

Both conditions are loaded through the manifest verifier, so a parser artifact
that does not belong to the corpus being analysed stops the run.

Outputs go to the FT stage namespace and cannot touch the frozen base tables.
"""
from __future__ import annotations

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

from src.justification_analysis.comparison import discourse_ft_contrast as fc  # noqa: E402

# Derived, not imposed: asserted so a silent change in what is usable is caught.
EXPECTED_MATCHED = {
    "Gemma 4 2B": (190, 570),
    "Gemma 4 4B": (191, 573),
    "Gemma 4 31B": (188, 564),
}

# A sense outside FOCUS_SENSES is reported only if it moves by at least this
# much, in relations per 100 words. Prevents a catalogue of tiny changes.
SENSE_REPORT_THRESHOLD = 0.05


def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print("=" * 78)
    print("RQ3 DISCOURSE CONTRAST - BASE vs fine-tuned, matched games")
    print("=" * 78)

    base = fc.load_condition(fc.BASE_STAGE)
    ft = fc.load_condition(fc.FT_STAGE)
    print(f"BASE corpus : {len(base.corpus):,} stochastic justifications, "
          f"{len(base.accepted):,} accepted relations")
    print(f"FT corpus   : {len(ft.corpus):,} stochastic justifications, "
          f"{len(ft.accepted):,} accepted relations")
    print(f"BASE artifact fingerprint : "
          f"{base.manifest['corpus']['fingerprint'][:16]}...")
    print(f"FT artifact fingerprint   : "
          f"{ft.manifest['corpus']['fingerprint'][:16]}...")

    models = base.config.model_order

    # --- A. input audit ---------------------------------------------------
    print("\n" + "-" * 78)
    print("A. INPUT AUDIT")
    print("-" * 78)
    matched = {}
    audit_rows, exclusion_frames = [], []
    for model in models:
        games = fc.matched_games(base, ft, model)
        matched[model] = games
        expected_games, expected_just = EXPECTED_MATCHED[model]
        n_just = 3 * len(games)
        ok = (len(games) == expected_games) and (n_just == expected_just)
        audit_rows.append({
            "model": model, "matched_games": len(games),
            "expected_games": expected_games,
            "justifications_per_condition": n_just,
            "expected_justifications": expected_just,
            "matches_expected": ok,
        })
        exclusions = fc.excluded_games(base, ft, model)
        if not exclusions.empty:
            exclusion_frames.append(exclusions)
    audit = pd.DataFrame(audit_rows)
    print(audit.to_string(index=False))
    if exclusion_frames:
        print("\nExcluded games (dropped from BOTH conditions):")
        print(pd.concat(exclusion_frames, ignore_index=True).to_string(index=False))
    assert audit["matches_expected"].all(), "matched sets differ from expectation"

    # --- B. descriptive profile ------------------------------------------
    print("\n" + "-" * 78)
    print("B. BASIC DISCOURSE PROFILE (matched games)")
    print("-" * 78)
    profile_rows = []
    for model in models:
        for condition in (base, ft):
            values = fc.descriptive_profile(condition, model, matched[model])
            values.update(model=model,
                          condition="BASE" if condition is base else "FT")
            profile_rows.append(values)
    profile = pd.DataFrame(profile_rows)[
        ["model", "condition", "n_justifications", "n_runs",
         "pct_justifications_with_relation", "mean_words_per_justification",
         "relations_per_100_words"]
    ]
    print(profile.round(3).to_string(index=False))

    # --- C/D. contrasts ---------------------------------------------------
    senses = sorted(set(fc._sense_metrics(base.accepted))
                    | set(fc._sense_metrics(ft.accepted)))
    print(f"\nlevel-2 senses present in either condition: {len(senses)}")

    contrasts = pd.concat(
        [fc.contrast_model(base, ft, model, matched[model], senses)
         for model in models],
        ignore_index=True,
    )

    print("\n" + "-" * 78)
    print("C. OVERALL AND TOP-LEVEL DENSITY (FT - BASE, per 100 words)")
    print("-" * 78)
    top = contrasts.loc[contrasts["level"].isin(["overall", "top_level"])].copy()
    top["metric"] = pd.Categorical(
        top["metric"], ["All relations", *fc.CATEGORY_ORDER], ordered=True)
    top = top.sort_values(["model", "metric"])
    print(top[["model", "metric", "n_games", "base", "ft", "delta",
               "ci_low", "ci_high", "ci_excludes_zero"]]
          .round(3).to_string(index=False))

    print("\n" + "-" * 78)
    print("D. LEVEL-2 SENSES")
    print("-" * 78)
    sense_rows = contrasts.loc[contrasts["level"].eq("sense")].copy()
    focus = sense_rows["metric"].isin(fc.FOCUS_SENSES)
    substantial = sense_rows["delta"].abs().ge(SENSE_REPORT_THRESHOLD)
    reported = sense_rows.loc[focus | substantial].copy()
    reported["reason"] = np.where(reported["metric"].isin(fc.FOCUS_SENSES),
                                  "focus", "substantial change")
    print(reported[["model", "metric", "base", "ft", "delta",
                    "ci_low", "ci_high", "ci_excludes_zero", "reason"]]
          .round(3).to_string(index=False))
    omitted = sense_rows.loc[~(focus | substantial), "metric"].unique()
    print(f"\nomitted as negligible (|delta| < {SENSE_REPORT_THRESHOLD} and not "
          f"a focus sense): {sorted(omitted)}")

    # --- composition ------------------------------------------------------
    print("\n" + "-" * 78)
    print("E. COMPOSITION WITHIN THE MAJOR CLASSES (share of the class, %)")
    print("-" * 78)
    composition_specs = [
        ("Contingency", ["Contingency.Cause", "Contingency.Condition"]),
        ("Temporal", ["Temporal.Asynchronous", "Temporal.Synchrony"]),
        ("Comparison", ["Comparison.Contrast"]),
        ("Expansion", ["Expansion.Conjunction"]),
    ]
    composition = pd.concat(
        [fc.composition_contrast(base, ft, model, matched[model], parts, whole)
         for model in models for whole, parts in composition_specs],
        ignore_index=True,
    )
    print(composition[["model", "whole", "part", "base_pct", "ft_pct",
                       "delta_pct", "ci_low", "ci_high", "ci_excludes_zero"]]
          .round(2).to_string(index=False))

    # --- outputs ----------------------------------------------------------
    out_dir = ft.config.discourse_dir / "thesis_tables" / "rq3_base_ft_contrast"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "R0_input_audit": audit,
        "R1_descriptive_profile": profile,
        "R2_density_contrast": top,
        "R3_sense_contrast": sense_rows,
        "R4_composition_contrast": composition,
    }
    if exclusion_frames:
        written["R0b_excluded_games"] = pd.concat(exclusion_frames,
                                                  ignore_index=True)
    for name, frame in written.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\nwrote {len(written)} tables to "
          f"{out_dir.relative_to(REPO_ROOT)}")

    # --- integrity --------------------------------------------------------
    print("\n" + "-" * 78)
    print("F. INTEGRITY CHECKS")
    print("-" * 78)
    checks = []
    for model in models:
        rows = top.loc[top["model"].eq(model)]
        for column in ("base", "ft"):
            classes = rows.loc[rows["level"].eq("top_level"), column].sum()
            overall = float(rows.loc[rows["metric"].eq("All relations"), column].iloc[0])
            checks.append((f"{model}: {column} class densities sum to overall",
                           np.isclose(classes, overall)))
        sense_sum = contrasts.loc[
            contrasts["model"].eq(model) & contrasts["level"].eq("sense"), "ft"].sum()
        overall_ft = float(rows.loc[rows["metric"].eq("All relations"), "ft"].iloc[0])
        checks.append((f"{model}: FT sense densities sum to overall",
                       np.isclose(sense_sum, overall_ft)))
        checks.append((f"{model}: point estimate inside its own CI",
                       bool(((rows["delta"] >= rows["ci_low"])
                             & (rows["delta"] <= rows["ci_high"])).all())))
    checks.append(("no greedy run entered the contrast",
                   set(base.corpus["decoding_group"]) == {"Stochastic"}
                   and set(ft.corpus["decoding_group"]) == {"Stochastic"}))
    for label, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        return 1
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
