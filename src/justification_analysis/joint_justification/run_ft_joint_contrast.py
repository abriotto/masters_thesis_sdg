"""RQ3: did familiarisation change how semantic content is organised through
explicit discourse relations?

Runs the frozen RQ2 joint definitions over the fine-tuned corpus and reports
within-model, game-matched BASE-FT differences with paired game-level bootstrap
CIs. Stochastic runs only - the fine-tuned models were never run greedily.

    "C:/Users/annab/miniconda3/envs/sdglogs/python.exe" \
      src/justification_analysis/joint_justification/run_ft_joint_contrast.py

The contrast is ANCHORED to the associations RQ2 established. It does not
sweep the 7 x 5 grid of semantic x discourse pairs per model and report
whichever moved; that would be a multiple-comparisons exercise dressed up as a
finding. The anchored set is `joint_ft_contrast.ANCHORED_PAIRS`.

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

from src.justification_analysis.joint_justification import joint_ft_contrast as jc  # noqa: E402

EXPECTED_MATCHED = {
    "Gemma 4 2B": (190, 570),
    "Gemma 4 4B": (191, 573),
    "Gemma 4 31B": (188, 564),
}

METRIC_LABELS = {
    "category_prevalence": "category prevalence (%)",
    "conditional_prevalence": "P(relation | category) (%)",
    "conditional_density": "density within category (/100 words)",
    "baseline_density": "model-wide baseline density (/100 words)",
    "localization": "localization (% sharing a sentence)",
    "lift": "lift",
}
METRIC_ORDER = list(METRIC_LABELS)


def show(frame: pd.DataFrame, columns, decimals=3) -> None:
    print(frame[columns].round(decimals).to_string(index=False))


def contrast_table(base, ft, matched, pairs, kind="top_level") -> pd.DataFrame:
    rows = []
    for model in EXPECTED_MATCHED:
        for category, relation in pairs:
            rows.append(jc.condition_contrast(
                base, ft, model, matched[model], category, relation, kind))
    table = pd.concat(rows, ignore_index=True)
    table["metric"] = pd.Categorical(table["metric"], METRIC_ORDER, ordered=True)
    return table.sort_values(["semantic_category", "discourse_relation",
                              "model", "metric"])


def main() -> int:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print("=" * 78)
    print("RQ3 JOINT DISCOURSE x SEMANTIC CONTRAST - BASE vs FT, matched games")
    print("=" * 78)

    base = jc.load_condition(jc.BASE_STAGE)
    ft = jc.load_condition(jc.FT_STAGE)
    models = list(EXPECTED_MATCHED)

    # Relations are counted on the STOCHASTIC justifications only. The raw
    # `aligned` table spans every decoding, so quoting it would compare a BASE
    # total that includes greedy against an FT total that has no greedy.
    def stochastic_relations(condition) -> int:
        ids = set(condition.justifications["justification_id"])
        return int(condition.alignment["aligned"]
                   ["justification_id_canonical"].isin(ids).sum())

    print(f"BASE : {len(base.justifications):,} stochastic justifications, "
          f"{stochastic_relations(base):,} aligned relations")
    print(f"FT   : {len(ft.justifications):,} stochastic justifications, "
          f"{stochastic_relations(ft):,} aligned relations")

    # --- A1. layer alignment on justification ids -------------------------
    print("\n" + "-" * 78)
    print("A1. DISCOURSE / SEMANTIC LAYER ALIGNMENT ON JUSTIFICATION IDS")
    print("-" * 78)
    alignment_audit = pd.concat(
        [jc.id_alignment(condition) for condition in (base, ft)],
        ignore_index=True)
    print(alignment_audit.to_string(index=False))

    misaligned = alignment_audit.loc[
        alignment_audit["check"].isin([
            "ids in semantic but not in the joint frame",
            "ids in the joint frame but not in semantic",
            "relations that failed byte-exact alignment"]), "observed"]
    assert (misaligned == 0).all(), \
        "the discourse and semantic layers do not align exactly on ids"
    print("\nBoth layers cover exactly the same justification ids in both "
          "conditions, and every relation aligned byte-exactly.")

    # --- A2. matched sets equal those of the other two RQ3 strands --------
    print("\n" + "-" * 78)
    print("A2. MATCHED GAME SETS vs THE SEPARATE RQ3 DISCOURSE AND SEMANTIC RUNS")
    print("-" * 78)
    from src.justification_analysis.comparison import discourse_ft_contrast as dfc
    from src.justification_analysis.semantic import semantic_ft_contrast as sfc

    discourse_base = dfc.load_condition(dfc.BASE_STAGE)
    discourse_ft = dfc.load_condition(dfc.FT_STAGE)
    semantic_base = sfc.load_condition(sfc.BASE_STAGE)
    semantic_ft = sfc.load_condition(sfc.FT_STAGE)

    matched, audit_rows = {}, []
    for model in models:
        joint_games = jc.matched_games(base, ft, model)
        discourse_games = dfc.matched_games(discourse_base, discourse_ft, model)
        semantic_games = sfc.matched_games(semantic_base, semantic_ft, model)
        matched[model] = joint_games
        expected_games, expected_just = EXPECTED_MATCHED[model]
        audit_rows.append({
            "model": model,
            "joint_games": len(joint_games),
            "expected_games": expected_games,
            "justifications_per_condition": 3 * len(joint_games),
            "expected_justifications": expected_just,
            "same_set_as_RQ3_discourse": joint_games == discourse_games,
            "same_set_as_RQ3_semantic": joint_games == semantic_games,
            "matches_expected": (len(joint_games) == expected_games
                                 and 3 * len(joint_games) == expected_just),
        })
    audit = pd.DataFrame(audit_rows)
    print(audit.to_string(index=False))
    assert audit["matches_expected"].all(), "matched sets differ from expectation"
    assert audit["same_set_as_RQ3_discourse"].all(), \
        "matched games differ from the RQ3 discourse analysis"
    assert audit["same_set_as_RQ3_semantic"].all(), \
        "matched games differ from the RQ3 semantic analysis"

    exclusions = pd.concat(
        [jc.excluded_games(base, ft, model) for model in models],
        ignore_index=True)
    if not exclusions.empty:
        print("\nExcluded games (dropped from BOTH conditions):")
        print(exclusions.to_string(index=False))

    # --- B. PRIMARY: Mechanical x Contingency ------------------------------
    print("\n" + "=" * 78)
    print("B. PRIMARY JOINT OUTCOME - Mechanical x Contingency")
    print("=" * 78)
    primary = contrast_table(base, ft, matched, [jc.PRIMARY_PAIR])
    for model in models:
        rows = primary.loc[primary["model"].eq(model)].copy()
        rows["metric"] = rows["metric"].map(METRIC_LABELS)
        support = rows.iloc[0]
        print(f"\n{model}  ({support['n_games']} games; Mechanical "
              f"justifications per run: BASE {support['base_support_per_run']:.1f}, "
              f"FT {support['ft_support_per_run']:.1f}"
              + ("  <-- LOW SUPPORT" if support["low_support_diagnostic"] else "")
              + ")")
        show(rows, ["metric", "base", "ft", "delta", "ci_low", "ci_high",
                    "ci_excludes_zero"])

    print("\n" + "-" * 78)
    print("B2. FINE-GRAINED Contingency COMPONENTS WITHIN Mechanical")
    print("-" * 78)
    primary_sense = contrast_table(
        base, ft, matched,
        [("Mechanical", "Contingency.Cause"),
         ("Mechanical", "Contingency.Condition")],
        kind="sense")
    fine = primary_sense.loc[primary_sense["metric"].isin(
        ["conditional_density", "lift", "localization"])].copy()
    fine["metric"] = fine["metric"].map(METRIC_LABELS)
    show(fine, ["model", "discourse_relation", "metric", "base", "ft", "delta",
                "ci_low", "ci_high", "ci_excludes_zero",
                "low_support_diagnostic"])

    # --- C. E2B Contingency composition ------------------------------------
    print("\n" + "=" * 78)
    print("C. WHERE Contingency SURFACES - share of Contingency SENTENCES that "
          "also carry each category")
    print("=" * 78)
    composition = pd.concat(
        [jc.relation_composition(base, ft, model, matched[model], "Contingency")
         for model in models],
        ignore_index=True)
    for model in models:
        rows = composition.loc[composition["model"].eq(model)]
        head = rows.iloc[0]
        print(f"\n{model}  (Contingency sentences per run: BASE "
              f"{head['base_relation_sentences_per_run']:.1f}, FT "
              f"{head['ft_relation_sentences_per_run']:.1f})")
        show(rows.sort_values("base_pct_of_relation_sentences", ascending=False),
             ["semantic_category", "base_pct_of_relation_sentences",
              "ft_pct_of_relation_sentences", "delta_pp", "ci_low", "ci_high",
              "ci_excludes_zero"], decimals=1)

    # --- D. ClaimComparison x Temporal -------------------------------------
    print("\n" + "=" * 78)
    print("D. ClaimComparison x Temporal")
    print("=" * 78)
    temporal = contrast_table(base, ft, matched, [("ClaimComparison", "Temporal")])
    rows = temporal.loc[temporal["metric"].isin(
        ["conditional_density", "baseline_density", "lift", "localization",
         "conditional_prevalence"])].copy()
    rows["metric"] = rows["metric"].map(METRIC_LABELS)
    show(rows, ["model", "metric", "base", "ft", "delta", "ci_low", "ci_high",
                "ci_excludes_zero", "low_support_diagnostic"])

    print("\nTemporal.Asynchronous component:")
    temporal_sense = contrast_table(
        base, ft, matched, [("ClaimComparison", "Temporal.Asynchronous")],
        kind="sense")
    rows = temporal_sense.loc[temporal_sense["metric"].isin(
        ["conditional_density", "lift"])].copy()
    rows["metric"] = rows["metric"].map(METRIC_LABELS)
    show(rows, ["model", "metric", "base", "ft", "delta", "ci_low", "ci_high",
                "ci_excludes_zero", "low_support_diagnostic"])

    # --- E. other anchored BASE associations -------------------------------
    print("\n" + "=" * 78)
    print("E. OTHER ANCHORED BASE ASSOCIATIONS - SocialJudgment x Contingency")
    print("=" * 78)
    other = contrast_table(base, ft, matched,
                           [("SocialJudgment", "Contingency")])
    rows = other.loc[other["metric"].isin(
        ["conditional_density", "lift", "localization"])].copy()
    rows["metric"] = rows["metric"].map(METRIC_LABELS)
    show(rows, ["model", "metric", "base", "ft", "delta", "ci_low", "ci_high",
                "ci_excludes_zero"])

    material = rows.loc[rows["ci_excludes_zero"]]
    if material.empty:
        print("\nNothing here changes materially: every interval spans zero.")
    else:
        print(f"\n{len(material)} cell(s) change materially and are reported "
              f"above.")

    # --- outputs ----------------------------------------------------------
    all_contrasts = pd.concat([primary, primary_sense, temporal,
                               temporal_sense, other], ignore_index=True)
    out_dir = (ft.config.joint_dir / "thesis_tables" / "rq3_base_ft_contrast")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "J0_layer_alignment_audit": alignment_audit,
        "J0b_matched_game_audit": audit,
        "J1_anchored_contrasts": all_contrasts,
        "J2_contingency_sentence_composition": composition,
    }
    if not exclusions.empty:
        written["J0c_excluded_games"] = exclusions
    for name, frame in written.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\nwrote {len(written)} tables to {out_dir.relative_to(REPO_ROOT)}")

    # --- integrity --------------------------------------------------------
    print("\n" + "-" * 78)
    print("F. INTEGRITY CHECKS")
    print("-" * 78)
    finite = all_contrasts.dropna(subset=["ci_low", "ci_high"])

    # Not a pass/fail check. A replicate is invalid when the resample happens to
    # contain no justification carrying the category, which is a real property
    # of a cell with single-digit support (E2B/E4B Mechanical), not a defect.
    # It is listed so an interval computed on fewer replicates is read as such.
    degenerate = all_contrasts.loc[
        all_contrasts["n_valid_replicates"] < all_contrasts["n_replicates"]]
    if degenerate.empty:
        print("  [note] every bootstrap replicate was valid in every cell")
    else:
        print(f"  [note] {len(degenerate)} cell(s) had replicates in which the "
              f"category was absent from the resample:")
        print(degenerate[["model", "semantic_category", "discourse_relation",
                          "metric", "n_valid_replicates", "n_replicates",
                          "base_support_per_run"]].to_string(index=False))

    checks = [
        ("no greedy run entered the contrast",
         set(base.justifications["decoding_group"].astype(str)) == {"Stochastic"}
         and set(ft.justifications["decoding_group"].astype(str)) == {"Stochastic"}),
        ("point estimate lies inside its own CI",
         bool(((finite["delta"] >= finite["ci_low"] - 1e-9)
               & (finite["delta"] <= finite["ci_high"] + 1e-9)).all())),
        ("delta equals ft - base everywhere",
         bool(np.allclose(all_contrasts["delta"].fillna(0),
                          (all_contrasts["ft"] - all_contrasts["base"]).fillna(0)))),
        ("baseline density is condition-specific, not shared",
         bool(all_contrasts.loc[all_contrasts["metric"].astype(str)
                                .eq("baseline_density"), "base"].notna().all())),
        ("sentence composition shares lie in [0, 100]",
         bool(composition[["base_pct_of_relation_sentences",
                           "ft_pct_of_relation_sentences"]]
              .ge(0).all().all()
              and composition[["base_pct_of_relation_sentences",
                               "ft_pct_of_relation_sentences"]]
              .le(100).all().all())),
        ("every model contributes its expected justification count",
         bool(all(3 * len(matched[m]) == EXPECTED_MATCHED[m][1]
                  for m in models))),
    ]
    for label, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        return 1
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
