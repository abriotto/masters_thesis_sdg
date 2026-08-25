"""Phase 1 orchestration: the fidelity grid, LOFO, L1 selection frequency,
coefficient bootstrap, and the group-composition clustered bootstrap used for
every uncertainty interval in the RQ1 surrogate analysis.

Kept separate from :mod:`rq1_model` (single fit/predict primitives) so the
notebook can stay a short sequence of calls into this module.
"""

import numpy as np
import pandas as pd

from utils_choice.rq1_model import (ALL_FEATURES_13, BLOCKS, REPRESENTATIONS,
                                    feature_columns, run_nested_cv, fit_predict_penalized,
                                    per_game_metrics, per_game_avg_over_repeats,
                                    select_penalty_full_data, fit_final, make_design,
                                    L2_PENALTY_GRID, L1_PENALTY_GRID, ASC_COL)
from utils_choice.rq1_cv import outer_splits

N_BOOT_FIDELITY = 2000
N_BOOT_COEF = 1000


# ------------------------------------------------------------- null model ---
def run_null(frame, fold_df):
    per_game, fold_meta = run_nested_cv(frame, fold_df, cols=[], kind="l2")
    return per_game, fold_meta


# ------------------------------------------------------------ fidelity grid-
def run_fidelity_grid(frames_by_model, fold_df, blocks=("strategy", "enriched", "combined"),
                      representations=REPRESENTATIONS):
    """frames_by_model: {model_label: long_frame}. Returns
    (per_game_long, fold_level_long) with one row group per
    (model, representation, block, ...)."""
    pg_rows, fold_rows = [], []
    for model, frame in frames_by_model.items():
        for repr_ in representations:
            for block in blocks:
                cols = feature_columns(repr_, block)
                pg, fm = run_nested_cv(frame, fold_df, cols, kind="l2")
                pg["model"] = model; pg["representation"] = repr_; pg["block"] = block
                fm["model"] = model; fm["representation"] = repr_; fm["block"] = block
                pg_rows.append(pg); fold_rows.append(fm)
    return pd.concat(pg_rows, ignore_index=True), pd.concat(fold_rows, ignore_index=True)


# ------------------------------------------------------- group bootstrap ----
def _games_by_composition(per_game_df):
    out = {}
    for c, g in per_game_df.groupby("composition_id"):
        out[c] = g["key"].tolist()
    return out


def bootstrap_group_mean(per_game_df, metric, n_boot=N_BOOT_FIDELITY, seed=42):
    """Percentile bootstrap CI for the mean of ``metric`` over games,
    resampling group compositions (with replacement, whole composition's
    games move together)."""
    games_by_comp = _games_by_composition(per_game_df)
    comps = np.array(list(games_by_comp))
    val_by_key = dict(zip(per_game_df["key"], per_game_df[metric]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(comps, size=len(comps), replace=True)
        vals = [val_by_key[k] for c in sampled for k in games_by_comp[c]]
        boot[b] = np.mean(vals)
    point = float(per_game_df[metric].mean())
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def bootstrap_paired_diff(per_game_a, per_game_b, metric="loss", n_boot=N_BOOT_FIDELITY,
                          seed=42, label_a="a", label_b="b"):
    """95% percentile bootstrap CI for mean(b) - mean(a) over games, resampled
    by group composition (a composition's games always move together)."""
    merged = per_game_a[["key", "composition_id", metric]].merge(
        per_game_b[["key", metric]], on="key", suffixes=("_a", "_b"))
    diff_by_key = dict(zip(merged["key"], merged[f"{metric}_b"] - merged[f"{metric}_a"]))
    games_by_comp = _games_by_composition(merged)
    comps = np.array(list(games_by_comp))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(comps, size=len(comps), replace=True)
        vals = [diff_by_key[k] for c in sampled for k in games_by_comp[c]]
        boot[b] = np.mean(vals)
    mean_diff = float(np.mean(list(diff_by_key.values())))
    return {"comparison": f"{label_b} - {label_a}", "metric": metric,
           "mean_diff": mean_diff, "ci_lo": float(np.percentile(boot, 2.5)),
           "ci_hi": float(np.percentile(boot, 97.5)),
           "excludes_zero": bool(np.percentile(boot, 2.5) > 0 or np.percentile(boot, 97.5) < 0)}


# ------------------------------------------------------------------ LOFO ----
#: column name(s) that one of the 13 features contributes, per representation.
#: A temporal representation splits each feature into two columns, so dropping
#: the feature must drop BOTH -- an earlier version only ever removed a single
#: exact-name match, which silently removed *nothing* under the temporal
#: representations and would have reported a LOFO importance of ~0 for every
#: feature instead of failing.
_DROP_PATTERNS = {
    "count_overall": ("{f}",),
    "rate_overall": ("{f}_rate",),
    "count_temporal": ("{f}_early", "{f}_late"),
    "rate_temporal": ("{f}_early_rate", "{f}_late_rate"),
}


def _cols_without(representation, block, drop_feature):
    cols = feature_columns(representation, block)
    patterns = _DROP_PATTERNS[representation]
    targets = {p.format(f=drop_feature) for p in patterns}
    kept = [c for c in cols if c not in targets]
    n_dropped = len(cols) - len(kept)
    assert n_dropped == len(patterns), (
        f"dropping {drop_feature!r} from {representation}/{block} removed "
        f"{n_dropped} column(s), expected {len(patterns)} ({sorted(targets)})")
    return kept


def run_lofo(frame, fold_df, representation, block="combined",
            features=ALL_FEATURES_13):
    """Held-out log-loss delta (without-feature minus full) for every
    feature, using the SAME outer folds as the main grid."""
    full_cols = feature_columns(representation, block)
    pg_full, _ = run_nested_cv(frame, fold_df, full_cols, kind="l2")
    pg_full_avg = per_game_avg_over_repeats(pg_full)

    rows = []
    for feat in features:
        cols_wo = _cols_without(representation, block, feat)
        pg_wo, _ = run_nested_cv(frame, fold_df, cols_wo, kind="l2")
        pg_wo_avg = per_game_avg_over_repeats(pg_wo)
        diff = bootstrap_paired_diff(pg_full_avg, pg_wo_avg, metric="loss",
                                     label_a="full", label_b=f"without_{feat}")
        rows.append({"feature": feat, "representation": representation, "block": block,
                    "lofo_importance": diff["mean_diff"], "ci_lo": diff["ci_lo"],
                    "ci_hi": diff["ci_hi"], "excludes_zero": diff["excludes_zero"]})
    return pd.DataFrame(rows), pg_full_avg


# --------------------------------------------------------------- L1 freq ----
def run_lasso_selection(frame, fold_df, representation, block="combined",
                        features=ALL_FEATURES_13):
    cols = feature_columns(representation, block)
    pg, fold_meta, coef_rows = run_nested_cv(frame, fold_df, cols, kind="l1", collect_coef=True)
    n_models = fold_meta.shape[0]
    rows = []
    for feat, col in zip(features, cols):
        c = coef_rows[coef_rows["feature"] == col]
        nz = c[c["coef"] != 0]
        pos = nz[nz["coef"] > 0]; neg = nz[nz["coef"] < 0]
        rows.append({"feature": feat, "representation": representation, "block": block,
                    "n_outer_models": n_models,
                    "selection_freq": len(nz) / n_models,
                    "positive_selection_freq": len(pos) / n_models,
                    "negative_selection_freq": len(neg) / n_models,
                    "median_nonzero_coef": float(nz["coef"].median()) if len(nz) else np.nan,
                    "median_selected_lambda": float(fold_meta["selected_penalty"].median())})
    return pd.DataFrame(rows), coef_rows


# ---------------------------------------------------------- coefficients ---
def combined_coefficients(frame, representation, block="combined", features=ALL_FEATURES_13,
                          penalty_grid=None, seed=999):
    cols = feature_columns(representation, block)
    penalty = select_penalty_full_data(frame, cols, kind="l2", penalty_grid=penalty_grid, seed=seed)
    m = fit_final(frame, cols, "l2", penalty)
    betas = m.coef_[:len(cols)]
    tab = pd.DataFrame({"feature": features, "column": cols, "beta": betas,
                        "odds_ratio": np.exp(betas)})
    tab["selected_penalty"] = penalty
    return tab, penalty


def bootstrap_coefficients(frame, representation, block, penalty, features=ALL_FEATURES_13,
                           n_boot=N_BOOT_COEF, seed=7):
    cols = feature_columns(representation, block)
    games_by_comp = {}
    for c, g in frame.drop_duplicates("key")[["key", "composition_id"]].groupby("composition_id"):
        games_by_comp[c] = g["key"].tolist()
    comps = np.array(list(games_by_comp))
    by_key = {k: g for k, g in frame.groupby("key")}

    rng = np.random.default_rng(seed)
    boot_betas, failures = [], []
    for b in range(n_boot):
        sampled = rng.choice(comps, size=len(comps), replace=True)
        parts = []
        dup_counter = {}
        for c in sampled:
            for k in games_by_comp[c]:
                dup_counter[k] = dup_counter.get(k, 0) + 1
                gdf = by_key[k].copy()
                # keys are tuples; force a homogeneous string type so a
                # duplicated draw's suffixed key can coexist with the
                # original in the same "key" column (np.unique needs one
                # sortable dtype, not a tuple/str mix).
                gdf["key"] = (str(k) if dup_counter[k] == 1
                             else f"{k}__dup{dup_counter[k]}")
                parts.append(gdf)
        boot_frame = pd.concat(parts, ignore_index=True)
        try:
            X, unpen = make_design(boot_frame, boot_frame, cols)
            from utils_choice.rq1_model import _fit
            m = _fit(X, boot_frame["count"].values, boot_frame["key"].values,
                     penalty, unpen, "l2")
            if not m.converged_:
                failures.append({"boot_iter": b, "reason": "did_not_converge"})
                continue
            boot_betas.append(m.coef_[:len(cols)])
        except Exception as exc:                                    # noqa: BLE001
            failures.append({"boot_iter": b, "reason": str(exc)})

    boot_betas = np.array(boot_betas)
    rows = []
    for i, (feat, col) in enumerate(zip(features, cols)):
        b = boot_betas[:, i]
        rows.append({"feature": feat, "column": col,
                    "beta_ci_lo": float(np.percentile(b, 2.5)),
                    "beta_ci_hi": float(np.percentile(b, 97.5)),
                    "odds_ratio_ci_lo": float(np.exp(np.percentile(b, 2.5))),
                    "odds_ratio_ci_hi": float(np.exp(np.percentile(b, 97.5))),
                    "n_boot_used": len(boot_betas), "n_failures": len(failures)})
    return pd.DataFrame(rows), pd.DataFrame(failures)
