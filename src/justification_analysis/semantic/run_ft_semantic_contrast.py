"""RQ3: what changed in the SEMANTIC CONTENT of stated justifications, BASE -> FT.

Reuses the frozen RQ2 semantic definitions over the fine-tuned annotation run
and reports within-model, game-matched BASE-FT differences. Stochastic runs
only - the fine-tuned models were never run greedily.

    "C:/Users/annab/miniconda3/envs/sdglogs/python.exe" \
      src/justification_analysis/semantic/run_ft_semantic_contrast.py

Deliberately NOT rerun here: co-occurrence, lift, semantic breadth,
category-correctness association, mixed-game correctness. RQ3 asks whether the
stated content changed, and those are RQ2 characterisations of a corpus rather
than contrasts between two.

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

from src.justification_analysis.semantic import semantic_ft_contrast as fc  # noqa: E402

# Derived, not imposed: asserted so a silent change in what is usable is caught.
# Same sets as the RQ3 discourse contrast, by construction - the exclusion rule
# is imported from it rather than restated.
EXPECTED_MATCHED = {
    "Gemma 4 2B": (190, 570),
    "Gemma 4 4B": (191, 573),
    "Gemma 4 31B": (188, 564),
}

# Categories read in support of the primary outcome. Everything else is
# reported only if it moves substantially, so the table is not a catalogue.
SUPPORTING_CATEGORIES = (
    "Testimony", "ClaimComparison", "Payoff", "Uncertainty", "Behavioral",
)

# A non-focus category is called out only if it moves by at least this much.
REPORT_THRESHOLD_PP = 5.0


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print("=" * 78)
    print("RQ3 SEMANTIC CONTRAST - BASE vs fine-tuned, matched games")
    print("=" * 78)

    base = fc.load_condition(fc.BASE_STAGE)
    ft = fc.load_condition(fc.FT_STAGE)
    print(f"BASE annotations : {len(base.justifications):,} stochastic "
          f"justifications, {len(base.labels):,} category assignments")
    print(f"FT annotations   : {len(ft.justifications):,} stochastic "
          f"justifications, {len(ft.labels):,} category assignments")
    print(f"BASE annotation run : {base.config.semantic_run}")
    print(f"FT annotation run   : {ft.config.semantic_run}")

    models = base.config.model_order

    # --- A. input audit ---------------------------------------------------
    print("\n" + "-" * 78)
    print("A. INPUT AUDIT")
    print("-" * 78)
    matched = {}
    audit_rows, exclusion_frames, coverage_frames = [], [], []
    for model in models:
        games = fc.matched_games(base, ft, model)
        matched[model] = games
        expected_games, expected_just = EXPECTED_MATCHED[model]
        n_just = 3 * len(games)
        audit_rows.append({
            "model": model, "matched_games": len(games),
            "expected_games": expected_games,
            "justifications_per_condition": n_just,
            "expected_justifications": expected_just,
            "matches_expected": (len(games) == expected_games
                                 and n_just == expected_just),
        })
        exclusions = fc.excluded_games(base, ft, model)
        if not exclusions.empty:
            exclusion_frames.append(exclusions)
        for condition in (base, ft):
            gaps = fc.annotation_coverage(condition, model, games)
            if not gaps.empty:
                coverage_frames.append(gaps)
    audit = pd.DataFrame(audit_rows)
    print(audit.to_string(index=False))

    if exclusion_frames:
        print("\nExcluded games (dropped from BOTH conditions):")
        print(pd.concat(exclusion_frames, ignore_index=True).to_string(index=False))
    if coverage_frames:
        print("\nMatched-game rows WITHOUT an annotation "
              "(these would break the contrast):")
        print(pd.concat(coverage_frames, ignore_index=True).to_string(index=False))
    else:
        print("\nEvery justification in every matched game is annotated in both "
              "conditions.")

    assert audit["matches_expected"].all(), "matched sets differ from expectation"

    # --- B. descriptive profile ------------------------------------------
    print("\n" + "-" * 78)
    print("B. JUSTIFICATION PROFILE ON THE MATCHED GAMES")
    print("-" * 78)
    profile = pd.DataFrame([
        fc.descriptive_profile(condition, model, matched[model])
        for model in models for condition in (base, ft)
    ])
    profile["stage"] = profile["stage"].str.upper()
    print(profile[["model", "stage", "n_justifications",
                   "mean_sentences_per_justification",
                   "mean_labels_per_justification",
                   "mean_distinct_categories",
                   "labels_per_100_sentences"]].round(3).to_string(index=False))

    # --- C. PRIMARY: justification-level prevalence -----------------------
    print("\n" + "-" * 78)
    print("C. JUSTIFICATION-LEVEL PREVALENCE (FT - BASE, percentage points)")
    print("   PRIMARY OUTCOME: Mechanical")
    print("-" * 78)
    prevalence = pd.concat(
        [fc.prevalence_contrast(base, ft, model, matched[model])
         for model in models],
        ignore_index=True,
    )
    prevalence["category"] = pd.Categorical(
        prevalence["category"], fc.CATEGORIES, ordered=True)
    prevalence = prevalence.sort_values(["model", "category"])
    print(prevalence[["model", "category", "n_games", "base_pct", "ft_pct",
                      "delta_pp", "ci_low", "ci_high", "ci_excludes_zero"]]
          .round(2).to_string(index=False))

    print("\n>>> PRIMARY: " + fc.PRIMARY_CATEGORY)
    primary = prevalence.loc[prevalence["category"].astype(str)
                             .eq(fc.PRIMARY_CATEGORY)]
    print(primary[["model", "n_games", "base_pct", "ft_pct", "delta_pp",
                   "ci_low", "ci_high", "ci_excludes_zero"]]
          .round(2).to_string(index=False))

    # --- D. per-run spread behind the primary outcome ---------------------
    print("\n" + "-" * 78)
    print("D. PER-RUN PREVALENCE BEHIND THE MEANS "
          "(primary + supporting categories)")
    print("-" * 78)
    run_level = pd.concat(
        [fc.run_level_prevalence(condition, model, matched[model])
         for model in models for condition in (base, ft)],
        ignore_index=True,
    )
    focus = [fc.PRIMARY_CATEGORY, *SUPPORTING_CATEGORIES]
    shown = run_level.loc[run_level["category"].isin(focus)].copy()
    wide = shown.pivot_table(
        index=["model", "category"], columns=["stage", "run_label"],
        values="prevalence_pct",
    ).round(1)
    print(wide.to_string())

    # --- E. sentence-normalised sensitivity -------------------------------
    print("\n" + "-" * 78)
    print("E. SENTENCE-NORMALISED SENSITIVITY "
          "(assignments per 100 sentences, FT - BASE)")
    print("   SENSITIVITY ONLY - never a primary result")
    print("-" * 78)
    density = pd.concat(
        [fc.density_contrast(base, ft, model, matched[model])
         for model in models],
        ignore_index=True,
    )
    density["category"] = pd.Categorical(
        density["category"], fc.CATEGORIES, ordered=True)
    density = density.sort_values(["model", "category"])
    print(density[["model", "category", "base_per_100_sentences",
                   "ft_per_100_sentences", "delta", "ci_low", "ci_high",
                   "ci_excludes_zero"]].round(2).to_string(index=False))

    # --- F. agreement between the two metrics -----------------------------
    print("\n" + "-" * 78)
    print("F. DOES NORMALISING FOR LENGTH CHANGE THE CONCLUSION?")
    print("-" * 78)

    def direction(row_delta, excludes_zero):
        if not excludes_zero:
            return "no detectable change"
        return "increase" if row_delta > 0 else "decrease"

    # Renamed rather than merged with a suffix: only `ci_excludes_zero`
    # collides, so a suffix would rename one column and leave the other bare.
    joined = prevalence.merge(
        density[["model", "category", "delta", "ci_excludes_zero"]].rename(
            columns={"delta": "delta_density",
                     "ci_excludes_zero": "ci_excludes_zero_density"}),
        on=["model", "category"],
    )
    joined["prevalence_direction"] = [
        direction(d, z) for d, z in
        zip(joined["delta_pp"], joined["ci_excludes_zero"])
    ]
    joined["density_direction"] = [
        direction(d, z) for d, z in
        zip(joined["delta_density"], joined["ci_excludes_zero_density"])
    ]
    joined["agree"] = joined["prevalence_direction"].eq(joined["density_direction"])
    print(joined[["model", "category", "delta_pp", "prevalence_direction",
                  "delta_density", "density_direction", "agree"]]
          .round(2).to_string(index=False))

    disagreements = joined.loc[~joined["agree"]]
    if disagreements.empty:
        print("\nThe two metrics agree on direction for every model x category.")
    else:
        print(f"\n{len(disagreements)} model x category cell(s) where the "
              f"sentence-normalised metric does NOT agree with prevalence:")
        print(disagreements[["model", "category", "prevalence_direction",
                             "density_direction"]].to_string(index=False))

    # --- G. what is worth reporting ---------------------------------------
    print("\n" + "-" * 78)
    print(f"G. CHANGES WORTH READING (primary, supporting, or |delta| >= "
          f"{REPORT_THRESHOLD_PP} pp)")
    print("-" * 78)
    is_focus = prevalence["category"].astype(str).isin(
        [fc.PRIMARY_CATEGORY, *SUPPORTING_CATEGORIES])
    is_large = prevalence["delta_pp"].abs().ge(REPORT_THRESHOLD_PP)
    reported = prevalence.loc[is_focus | is_large].copy()
    reported["reason"] = np.where(
        reported["category"].astype(str).eq(fc.PRIMARY_CATEGORY), "PRIMARY",
        np.where(is_focus.loc[reported.index], "supporting", "large change"))
    print(reported[["model", "category", "base_pct", "ft_pct", "delta_pp",
                    "ci_low", "ci_high", "ci_excludes_zero", "reason"]]
          .round(2).to_string(index=False))
    omitted = sorted(set(prevalence["category"].astype(str))
                     - set(reported["category"].astype(str)))
    print(f"\nomitted as neither focus nor large: {omitted}")

    # --- outputs ----------------------------------------------------------
    out_dir = ft.config.semantic_dir / "thesis_tables" / "rq3_base_ft_contrast"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "S0_input_audit": audit,
        "S1_descriptive_profile": profile,
        "S2_prevalence_contrast": prevalence,
        "S3_run_level_prevalence": run_level,
        "S4_density_sensitivity_contrast": density,
        "S5_metric_agreement": joined,
    }
    if exclusion_frames:
        written["S0b_excluded_games"] = pd.concat(exclusion_frames,
                                                  ignore_index=True)
    if coverage_frames:
        written["S0c_unannotated_rows"] = pd.concat(coverage_frames,
                                                    ignore_index=True)
    for name, frame in written.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\nwrote {len(written)} tables to {out_dir.relative_to(REPO_ROOT)}")

    # --- integrity --------------------------------------------------------
    print("\n" + "-" * 78)
    print("H. INTEGRITY CHECKS")
    print("-" * 78)
    checks = [
        ("no greedy run entered the contrast",
         set(base.justifications["decoding_group"].astype(str)) == {"Stochastic"}
         and set(ft.justifications["decoding_group"].astype(str)) == {"Stochastic"}),
        ("point estimate lies inside its own prevalence CI",
         bool(((prevalence["delta_pp"] >= prevalence["ci_low"] - 1e-9)
               & (prevalence["delta_pp"] <= prevalence["ci_high"] + 1e-9)).all())),
        ("point estimate lies inside its own density CI",
         bool(((density["delta"] >= density["ci_low"] - 1e-9)
               & (density["delta"] <= density["ci_high"] + 1e-9)).all())),
        ("delta equals ft - base in every prevalence row",
         bool(np.allclose(prevalence["delta_pp"],
                          prevalence["ft_pct"] - prevalence["base_pct"]))),
        ("every prevalence lies in [0, 100]",
         bool(prevalence[["base_pct", "ft_pct"]].ge(0).all().all()
              and prevalence[["base_pct", "ft_pct"]].le(100).all().all())),
        ("all eight categories are present for every model",
         bool((prevalence.groupby("model", observed=True)["category"]
               .nunique() == len(fc.CATEGORIES)).all())),
        ("both conditions contribute the same number of justifications",
         bool((profile.groupby("model")["n_justifications"]
               .nunique() == 1).all())),
    ]
    for label, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        return 1
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
