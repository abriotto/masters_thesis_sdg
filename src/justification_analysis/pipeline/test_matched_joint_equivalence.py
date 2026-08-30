"""The matched RQ3 joint formulas must equal the frozen RQ2 joint ones.

`joint_ft_contrast` recomputes conditional prevalence, conditional density,
lift and localization from `count_tensors` output rather than calling the
frozen table functions, because those aggregate all three models jointly and
have no way to express "the same 188 games on both sides". A second
implementation of a frozen definition is exactly the kind of thing that
silently drifts.

This test removes the doubt by running both on the ONE input where both are
valid: the full 191-game BASE stochastic set, all three models, every
(category, relation) cell - not only the anchored ones.

Six things are compared:

  1. conditional prevalence   - against `conditional_prevalence`;
  2. conditional density      - against `conditional_density`
                                (`relations_per_100_words`, the canonical
                                ratio of sums);
  3. justification-level lift - against `joint_prevalence_and_lift`;
  4. localization             - against `localization_rate`;
  5. baseline density         - against the frozen discourse path
                                (`discourse_final.top_level_density`), which
                                is where the model-wide denominator is defined;
  6. sense level              - 2 and 3 again with kind="sense", so the
                                Cause/Condition components are covered too.

Plus a self-contrast: BASE against BASE must give a difference of exactly zero
with a degenerate interval on every metric, which checks the paired bootstrap
wiring rather than the arithmetic.

    python -m src.justification_analysis.pipeline.test_matched_joint_equivalence
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.justification_analysis.comparison import discourse_final as dfin  # noqa: E402
from src.justification_analysis.joint_justification import justification_joint as jj  # noqa: E402
from src.justification_analysis.joint_justification import joint_ft_contrast as jc  # noqa: E402

TOLERANCE = 1e-9

results = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, bool(passed), detail))


def close(left, right) -> bool:
    return bool(np.allclose(np.asarray(left, dtype=float),
                            np.asarray(right, dtype=float),
                            rtol=0, atol=TOLERANCE, equal_nan=True))


def worst(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size == 0:
        return float("nan")
    difference = np.abs(left - right)
    return float(np.nanmax(difference)) if np.isfinite(difference).any() else 0.0


def main() -> int:
    print("=" * 78)
    print("EQUIVALENCE: matched RQ3 joint formulas vs frozen RQ2 joint formulas")
    print("full 191-game BASE stochastic set - the input where both are valid")
    print("=" * 78)

    base = jc.load_condition(jc.BASE_STAGE)
    models = list(jj.MODEL_ORDER)

    # The full BASE set, matched against itself: the subsetting must be inert.
    games_by_model = {}
    for model in models:
        games = jc.matched_games(base, base, model)
        games_by_model[model] = games
        check(f"{model}: BASE matched against itself retains 191 games",
              len(games) == 191, f"{len(games)} games")

    # --- the frozen path --------------------------------------------------
    stochastic = base.justifications
    frozen_prevalence = jj.conditional_prevalence(stochastic, "top_level")
    frozen_density = jj.conditional_density(stochastic, "top_level")
    frozen_lift = jj.joint_prevalence_and_lift(stochastic, "top_level")
    frozen_local = jj.localization_rate(base.sentences_joint, stochastic,
                                        "top_level")
    frozen_sense_density = jj.conditional_density(stochastic, "sense")
    frozen_sense_lift = jj.joint_prevalence_and_lift(stochastic, "sense")

    def frozen_mean(frame: pd.DataFrame, value: str) -> pd.Series:
        subset = frame.loc[frame["decoding_group"].astype(str).eq("Stochastic")]
        return (subset.assign(
            model=subset["model"].astype(str),
            semantic_category=subset["semantic_category"].astype(str),
            discourse_relation=subset["discourse_relation"].astype(str))
            .groupby(["model", "semantic_category", "discourse_relation"],
                     observed=True)[value].mean().sort_index())

    # --- the matched path -------------------------------------------------
    def matched_frame(kind: str, relations) -> pd.DataFrame:
        rows = []
        for model in models:
            for category in jc.CATEGORY_ORDER:
                for relation in relations:
                    rows.append(jc.point_estimates(
                        base, model, games_by_model[model], category,
                        relation, kind))
        return pd.DataFrame(rows).rename(columns={"model": "model"}).set_index(
            ["model", "semantic_category", "discourse_relation"]).sort_index()

    top_relations = [*jj.TOP_LEVEL_ORDER, jj.ANY_RELATION]
    matched = matched_frame("top_level", top_relations)

    # 1. conditional prevalence
    left = 100 * frozen_mean(frozen_prevalence, "conditional_prevalence")
    right = matched["conditional_prevalence"].reindex(left.index)
    check("conditional prevalence P(r|c) is identical",
          close(left, right), f"max |diff| = {worst(left, right):.3g} pp "
                              f"over {len(left)} cells")

    # 2. conditional density
    left = frozen_mean(frozen_density, "relations_per_100_words")
    right = matched["conditional_density"].reindex(left.index)
    check("conditional density (relations per 100 words within c) is identical",
          close(left, right), f"max |diff| = {worst(left, right):.3g} "
                              f"over {len(left)} cells")

    # 3. lift
    left = frozen_mean(frozen_lift, "lift")
    right = matched["lift"].reindex(left.index)
    check("justification-level lift is identical",
          close(left, right), f"max |diff| = {worst(left, right):.3g} "
                              f"over {len(left)} cells")

    # 4. localization
    left = 100 * frozen_mean(frozen_local, "localization_rate")
    right = matched["localization"].reindex(left.index)
    check("localization rate is identical",
          close(left, right), f"max |diff| = {worst(left, right):.3g} pp "
                              f"over {len(left)} cells")

    # 5. baseline density, against the frozen DISCOURSE path
    accepted, discourse_justifications = dfin.load_production_data()
    frozen_top = dfin.top_level_density(
        dfin.run_level_table(accepted, discourse_justifications))
    frozen_top = frozen_top.loc[
        frozen_top["decoding_group"].astype(str).eq("Stochastic")]
    baseline_rows = []
    for model in models:
        row = frozen_top.loc[frozen_top["model"].astype(str).eq(model)]
        for relation in jj.TOP_LEVEL_ORDER:
            column = f"{relation}_per_100_words_mean"
            baseline_rows.append({
                "model": model, "discourse_relation": relation,
                "frozen": float(row[column].iloc[0]),
                "matched": float(matched.loc[
                    (model, "Mechanical", relation), "baseline_density"]),
            })
    baseline = pd.DataFrame(baseline_rows)
    check("model-wide baseline density matches the frozen discourse path",
          close(baseline["frozen"], baseline["matched"]),
          f"max |diff| = {worst(baseline['frozen'], baseline['matched']):.3g} "
          f"over {len(baseline)} cells")

    # The baseline must not depend on which category row it was read from.
    spread = []
    for model in models:
        for relation in jj.TOP_LEVEL_ORDER:
            values = [matched.loc[(model, c, relation), "baseline_density"]
                      for c in jc.CATEGORY_ORDER]
            spread.append(max(values) - min(values))
    check("baseline density is invariant to the category it is read from",
          close(spread, np.zeros(len(spread))), f"max spread = {max(spread):.3g}")

    # 6. sense level
    matched_sense = matched_frame("sense", jj.SENSE_ORDER)
    left = frozen_mean(frozen_sense_density, "relations_per_100_words")
    right = matched_sense["conditional_density"].reindex(left.index)
    check("sense-level conditional density is identical",
          close(left, right), f"max |diff| = {worst(left, right):.3g} "
                              f"over {len(left)} cells")
    left = frozen_mean(frozen_sense_lift, "lift")
    right = matched_sense["lift"].reindex(left.index)
    check("sense-level lift is identical",
          close(left, right), f"max |diff| = {worst(left, right):.3g} "
                              f"over {len(left)} cells")

    # --- self-contrast: the paired bootstrap wiring ------------------------
    self_contrast = pd.concat(
        [jc.condition_contrast(base, base, model, games_by_model[model],
                               category, relation, n_replicates=200)
         for model in models
         for category, relation in jc.ANCHORED_PAIRS],
        ignore_index=True,
    )
    check("BASE vs BASE delta is exactly zero on every joint metric",
          bool((self_contrast["delta"].fillna(0) == 0).all()),
          f"max |delta| = {self_contrast['delta'].abs().max():.3g}")
    check("BASE vs BASE interval is degenerate on every joint metric",
          bool((self_contrast["ci_low"].fillna(0) == 0).all()
               and (self_contrast["ci_high"].fillna(0) == 0).all()
               and not self_contrast["ci_excludes_zero"].any()),
          f"widest interval = "
          f"{(self_contrast['ci_high'] - self_contrast['ci_low']).abs().max():.3g}")

    composition = pd.concat(
        [jc.relation_composition(base, base, model, games_by_model[model],
                                 "Contingency", n_replicates=200)
         for model in models],
        ignore_index=True,
    )
    check("BASE vs BASE sentence composition delta is exactly zero",
          bool((composition["delta_pp"].fillna(0) == 0).all()),
          f"max |delta| = {composition['delta_pp'].abs().max():.3g} pp")

    # --- report -----------------------------------------------------------
    print("\n" + "-" * 78)
    width = max(len(name) for name, _, _ in results)
    for name, passed, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
    print("-" * 78)

    failures = [name for name, passed, _ in results if not passed]
    print(f"{len(results) - len(failures)}/{len(results)} checks passed")
    if failures:
        print("\nFAILED:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("\nThe matched RQ3 joint path reproduces the frozen RQ2 joint results "
          "on the full BASE corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
