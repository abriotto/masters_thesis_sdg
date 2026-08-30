"""The matched RQ3 run-level formulas must equal the frozen RQ2 ones.

`discourse_final.run_level_table` asserts 191 justifications per run, so it
cannot be called on a matched subset of 188 or 190 games. `discourse_ft_contrast`
therefore recomputes the run level itself. That is a second implementation of a
frozen definition, which is exactly the kind of thing that silently drifts.

This test removes the doubt by running both on the ONE input where both are
valid: the full 191-game BASE stochastic set. On that input the two must agree
to numerical precision on every discourse metric, or the RQ3 contrast is not
measuring what RQ2 measured.

Four levels are compared:

  1. per run          - words, relation counts, densities, class counts,
                        class densities, class shares;
  2. per model        - the averaged densities, against the frozen F1/F2 path
                        (`overall_density`, `top_level_density`);
  3. level-2 senses   - against the frozen `fine_grained_senses`;
  4. descriptives     - coverage and mean length.

Plus a self-contrast: BASE against BASE must give a difference of exactly zero
with a degenerate interval, which checks the paired bootstrap wiring rather
than the arithmetic.

    python -m src.justification_analysis.pipeline.test_matched_runlevel_equivalence
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
from src.justification_analysis.comparison import discourse_ft_contrast as fc  # noqa: E402

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
        return 0.0
    return float(np.nanmax(np.abs(left - right)))


def main() -> int:
    print("=" * 78)
    print("EQUIVALENCE: matched RQ3 run-level vs frozen RQ2 run-level")
    print("full 191-game BASE stochastic set - the input where both are valid")
    print("=" * 78)

    # --- the frozen path --------------------------------------------------
    accepted, justifications = fin.load_production_data()
    frozen_run = fin.run_level_table(accepted, justifications)
    frozen_run = frozen_run.loc[
        frozen_run["decoding_group"].astype(str).eq("Stochastic")
    ].copy()
    frozen_overall = fin.overall_density(fin.run_level_table(accepted, justifications))
    frozen_top = fin.top_level_density(fin.run_level_table(accepted, justifications))
    frozen_senses = fin.fine_grained_senses(accepted, justifications)

    # --- the RQ3 path -----------------------------------------------------
    base = fc.load_condition(fc.BASE_STAGE)
    senses = sorted(base.accepted["raw_sense"].dropna().unique())
    models = base.config.model_order

    check("both paths see the same accepted relations (stochastic)",
          len(base.accepted) == int(
              accepted["justification_id"].isin(base.corpus["justification_id"]).sum()),
          f"{len(base.accepted)} vs "
          f"{int(accepted['justification_id'].isin(base.corpus['justification_id']).sum())}")

    max_deviation = 0.0

    for model in models:
        games = sorted(base.corpus.loc[base.corpus["model"].eq(model), "game_id"].unique())
        check(f"{model}: full game set is 191", len(games) == 191, str(len(games)))

        runs, words, metrics = fc.build_matrices(base, model, games, senses)
        frozen_model = frozen_run.loc[frozen_run["model"].astype(str).eq(model)] \
            .set_index("run_label").loc[runs]

        # --- 1. per run ---------------------------------------------------
        pairs = [
            ("total_words", words.sum(axis=1), frozen_model["total_words"]),
            ("total_relations", metrics["overall"].sum(axis=1),
             frozen_model["total_relations"]),
            ("relations_per_100_words",
             100 * metrics["overall"].sum(axis=1) / words.sum(axis=1),
             frozen_model["relations_per_100_words"]),
        ]
        for category in fc.PDTB_TOP_LEVEL:
            pairs.append((f"n_{category}", metrics[category].sum(axis=1),
                          frozen_model[f"n_{category}"]))
            pairs.append((f"{category}_per_100_words",
                          100 * metrics[category].sum(axis=1) / words.sum(axis=1),
                          frozen_model[f"{category}_per_100_words"]))
            pairs.append((f"{category}_pct_of_relations",
                          100 * metrics[category].sum(axis=1)
                          / metrics["overall"].sum(axis=1),
                          frozen_model[f"{category}_pct_of_relations"]))

        for label, mine, theirs in pairs:
            deviation = worst(mine, theirs)
            max_deviation = max(max_deviation, deviation)
            check(f"{model}: per-run {label}", close(mine, theirs),
                  f"max |diff| = {deviation:.3g}")

        # --- 2. per model, averaged --------------------------------------
        mine_overall = fc._density(metrics["overall"], words)
        theirs_overall = float(
            frozen_overall.loc[
                frozen_overall["model"].astype(str).eq(model)
                & frozen_overall["decoding_group"].astype(str).eq("Stochastic"),
                "relations_per_100_words_mean"].iloc[0])
        max_deviation = max(max_deviation, abs(mine_overall - theirs_overall))
        check(f"{model}: averaged overall density",
              close(mine_overall, theirs_overall),
              f"{mine_overall:.12f} vs {theirs_overall:.12f}")

        for category in fc.PDTB_TOP_LEVEL:
            mine_value = fc._density(metrics[category], words)
            theirs_value = float(
                frozen_top.loc[
                    frozen_top["model"].astype(str).eq(model)
                    & frozen_top["decoding_group"].astype(str).eq("Stochastic"),
                    f"{category}_per_100_words_mean"].iloc[0])
            max_deviation = max(max_deviation, abs(mine_value - theirs_value))
            check(f"{model}: averaged {category} density",
                  close(mine_value, theirs_value),
                  f"{mine_value:.12f} vs {theirs_value:.12f}")

        # --- 3. level-2 senses -------------------------------------------
        for sense in senses:
            mine_value = fc._density(metrics[sense], words)
            match = frozen_senses.loc[
                frozen_senses["model"].astype(str).eq(model)
                & frozen_senses["decoding_group"].astype(str).eq("Stochastic")
                & frozen_senses["raw_sense"].eq(sense),
                "mean_per_100_words"]
            theirs_value = float(match.iloc[0]) if len(match) else 0.0
            max_deviation = max(max_deviation, abs(mine_value - theirs_value))
            check(f"{model}: averaged {sense} density",
                  close(mine_value, theirs_value),
                  f"{mine_value:.12f} vs {theirs_value:.12f}")

        # --- 4. descriptives ---------------------------------------------
        profile = fc.descriptive_profile(base, model, games)
        theirs_pct = float(frozen_model["pct_justifications_with_relation"].mean())
        theirs_words = float(
            (frozen_model["total_words"] / frozen_model["n_justifications"]).mean())
        max_deviation = max(max_deviation,
                            abs(profile["pct_justifications_with_relation"] - theirs_pct),
                            abs(profile["mean_words_per_justification"] - theirs_words))
        check(f"{model}: pct justifications with a relation",
              close(profile["pct_justifications_with_relation"], theirs_pct),
              f"{profile['pct_justifications_with_relation']:.12f} vs {theirs_pct:.12f}")
        check(f"{model}: mean words per justification",
              close(profile["mean_words_per_justification"], theirs_words),
              f"{profile['mean_words_per_justification']:.12f} vs {theirs_words:.12f}")
        check(f"{model}: profile density equals the frozen one",
              close(profile["relations_per_100_words"], theirs_overall))

    # --- 5. self-contrast: BASE vs BASE must be exactly zero --------------
    model = models[0]
    games = sorted(base.corpus.loc[base.corpus["model"].eq(model), "game_id"].unique())
    self_contrast = fc.contrast_model(base, base, model, games, senses,
                                      n_replicates=200)
    check("self-contrast: every delta is exactly zero",
          bool((self_contrast["delta"].abs() < TOLERANCE).all()),
          f"max |delta| = {self_contrast['delta'].abs().max():.3g}")
    check("self-contrast: every interval is degenerate at zero",
          bool((self_contrast["ci_low"].abs() < TOLERANCE).all()
               and (self_contrast["ci_high"].abs() < TOLERANCE).all()))
    check("self-contrast: base and ft columns identical",
          close(self_contrast["base"], self_contrast["ft"]))
    check("self-contrast: point estimates match the frozen overall density",
          close(float(self_contrast.loc[
              self_contrast["metric"].eq("All relations"), "base"].iloc[0]),
              float(frozen_overall.loc[
                  frozen_overall["model"].astype(str).eq(model)
                  & frozen_overall["decoding_group"].astype(str).eq("Stochastic"),
                  "relations_per_100_words_mean"].iloc[0])))

    # --- report -----------------------------------------------------------
    failed = [row for row in results if not row[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    print(f"largest absolute deviation anywhere: {max_deviation:.3g} "
          f"(tolerance {TOLERANCE:g})")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"  [FAIL] {name}  -- {detail}")
        return 1
    print("\nThe matched RQ3 formulas reproduce the frozen RQ2 run level exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
