"""The matched RQ3 semantic formulas must equal the frozen RQ2 ones.

`semantic_ft_contrast` recomputes prevalence and sentence-normalised density
itself rather than calling `semantic_final`, because the frozen functions have
no way to express "the same 188 games on both sides" and `presence_tensor`
asserts a hole-free grid across all three models at once. A second
implementation of a frozen definition is exactly the kind of thing that
silently drifts.

This test removes the doubt by running both on the ONE input where both are
valid: the full 191-game BASE stochastic set. On that input the two must agree
to numerical precision, or the RQ3 semantic contrast is not measuring what RQ2
measured.

Four things are compared:

  1. per-run prevalence   - against `run_level_prevalence`, all eight
                            categories, every model and run;
  2. per-model prevalence - the three-run mean the contrast reports as
                            `base_pct`, against `model_prevalence`;
  3. per-model density    - `base_per_100_sentences`, against
                            `density_sensitivity` (seven substantive
                            categories; the frozen function drops `Other`);
  4. matched-game counts  - the full BASE set matched against itself must
                            retain all 191 games, i.e. the subsetting is
                            inert when there is nothing to exclude.

Plus a self-contrast: BASE against BASE must give a difference of exactly zero
with a degenerate interval, which checks the paired bootstrap wiring rather
than the arithmetic.

    python -m src.justification_analysis.pipeline.test_matched_semantic_equivalence
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.justification_analysis.semantic import semantic_final as sf  # noqa: E402
from src.justification_analysis.semantic import semantic_ft_contrast as fc  # noqa: E402

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
    return float(np.nanmax(np.abs(left - right)))


def main() -> int:
    print("=" * 78)
    print("EQUIVALENCE: matched RQ3 semantic formulas vs frozen RQ2 formulas")
    print("full 191-game BASE stochastic set - the input where both are valid")
    print("=" * 78)

    # --- the frozen path --------------------------------------------------
    frozen = sf.load_annotations()
    justifications = frozen["justifications"]
    stochastic = justifications.loc[
        justifications["decoding_group"].astype(str).eq("Stochastic")]

    frozen_run = sf.run_level_prevalence(stochastic)
    frozen_model = sf.model_prevalence(frozen_run)
    frozen_density = sf.density_sensitivity(stochastic, frozen["labels"])

    # --- the matched path -------------------------------------------------
    base = fc.load_condition(fc.BASE_STAGE)
    models = list(sf.MODEL_ORDER)

    print(f"\nfrozen stochastic justifications : {len(stochastic):,}")
    print(f"matched-path stochastic rows      : {len(base.justifications):,}")
    check("the two paths load the same stochastic rows",
          len(stochastic) == len(base.justifications),
          f"{len(stochastic)} vs {len(base.justifications)}")

    # 4. the subsetting is inert when nothing needs excluding.
    games_by_model = {}
    for model in models:
        games = fc.matched_games(base, base, model)
        games_by_model[model] = games
        check(f"{model}: BASE matched against itself retains 191 games",
              len(games) == 191, f"{len(games)} games")

    # 1. per-run prevalence.
    matched_run = pd.concat(
        [fc.run_level_prevalence(base, model, games_by_model[model])
         for model in models],
        ignore_index=True,
    )
    left = frozen_run.assign(
        model=frozen_run["model"].astype(str),
        category=frozen_run["category"].astype(str),
    ).set_index(["model", "run_label", "category"]).sort_index()
    right = matched_run.set_index(["model", "run_label", "category"]).sort_index()
    check("per-run tables cover the same cells",
          list(left.index) == list(right.index),
          f"{len(left)} vs {len(right)} rows")
    check("per-run n_present is identical",
          close(left["n_present"], right["n_present"]),
          f"max |diff| = {worst(left['n_present'], right['n_present']):.3g}")
    check("per-run prevalence is identical",
          close(100 * left["prevalence"], right["prevalence_pct"]),
          f"max |diff| = "
          f"{worst(100 * left['prevalence'], right['prevalence_pct']):.3g} pp")

    # 2/3. per-model prevalence and density, as the contrast reports them.
    #      BASE against BASE, so `base_pct` is the quantity under test and the
    #      delta doubles as the self-contrast check below.
    prevalence = pd.concat(
        [fc.prevalence_contrast(base, base, model, games_by_model[model],
                                n_replicates=200)
         for model in models],
        ignore_index=True,
    )
    density = pd.concat(
        [fc.density_contrast(base, base, model, games_by_model[model],
                             n_replicates=200)
         for model in models],
        ignore_index=True,
    )

    frozen_model_stoch = frozen_model.loc[
        frozen_model["decoding_group"].astype(str).eq("Stochastic")].assign(
        model=lambda f: f["model"].astype(str),
        category=lambda f: f["category"].astype(str),
    ).set_index(["model", "category"]).sort_index()
    contrast_prevalence = prevalence.set_index(["model", "category"]).sort_index()
    check("per-model prevalence cells match",
          list(frozen_model_stoch.index) == list(contrast_prevalence.index),
          f"{len(frozen_model_stoch)} vs {len(contrast_prevalence)} rows")
    check("per-model prevalence (three-run mean) is identical",
          close(100 * frozen_model_stoch["prevalence_mean"],
                contrast_prevalence["base_pct"]),
          f"max |diff| = "
          f"{worst(100 * frozen_model_stoch['prevalence_mean'], contrast_prevalence['base_pct']):.3g} pp")

    frozen_density_stoch = frozen_density.loc[
        frozen_density["decoding_group"].astype(str).eq("Stochastic")].assign(
        model=lambda f: f["model"].astype(str),
        category=lambda f: f["category"].astype(str),
    ).set_index(["model", "category"]).sort_index()
    # The frozen density function drops `Other`; compare on its own cells.
    contrast_density = density.set_index(["model", "category"]).sort_index()
    contrast_density = contrast_density.loc[frozen_density_stoch.index]
    check("per-model density cells match",
          len(frozen_density_stoch) == len(contrast_density),
          f"{len(frozen_density_stoch)} vs {len(contrast_density)} rows")
    check("per-model density (assignments per 100 sentences) is identical",
          close(frozen_density_stoch["density_mean"],
                contrast_density["base_per_100_sentences"]),
          f"max |diff| = "
          f"{worst(frozen_density_stoch['density_mean'], contrast_density['base_per_100_sentences']):.3g}")

    # --- self-contrast: the paired bootstrap wiring ------------------------
    check("BASE vs BASE prevalence delta is exactly zero",
          bool((prevalence["delta_pp"] == 0).all()),
          f"max |delta| = {prevalence['delta_pp'].abs().max():.3g} pp")
    check("BASE vs BASE prevalence interval is degenerate",
          bool((prevalence["ci_low"] == 0).all()
               and (prevalence["ci_high"] == 0).all()
               and not prevalence["ci_excludes_zero"].any()),
          f"widest interval = "
          f"{(prevalence['ci_high'] - prevalence['ci_low']).abs().max():.3g}")
    check("BASE vs BASE density delta is exactly zero",
          bool((density["delta"] == 0).all()),
          f"max |delta| = {density['delta'].abs().max():.3g}")
    check("BASE vs BASE density interval is degenerate",
          bool((density["ci_low"] == 0).all()
               and (density["ci_high"] == 0).all()
               and not density["ci_excludes_zero"].any()),
          f"widest interval = "
          f"{(density['ci_high'] - density['ci_low']).abs().max():.3g}")

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
    print("\nThe matched RQ3 semantic path reproduces the frozen RQ2 results "
          "on the full BASE corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
