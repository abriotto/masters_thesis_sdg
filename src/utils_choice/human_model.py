"""Thesis human-vote protocol: standardisation, grouped nested CV, bootstrap.

One estimator only -- an L2-penalised binary logistic regression with
``class_weight="balanced"``. Only the regularisation strength is tuned, on a
broad logarithmic grid, by ROC-AUC in a composition-grouped inner CV. No class
weights are searched, no decision threshold is tuned, no tree models, no
interactions.

Two design points that make the human numbers comparable with the LLM ones:

* **The scaler is fit on unique player-game vectors**, never on the duplicated
  pair rows. A game with six players contributes 30 ordered pairs and a game
  with four contributes 12, so fitting on pair rows would let roster size
  reweight the standardisation. Fitting on player-games also means the same
  feature is scaled identically on the voter and the candidate side, which is
  what makes the two blocks of coefficients directly comparable.
* **The fold structure is the LLM analysis's fold structure**, taken verbatim
  from :func:`utils_choice.rq1_cv.build_outer_folds`, so human and LLM models
  are evaluated on the same held-out games.

Repeated CV is aggregated *before* scoring: each ordered pair gets one
held-out probability per repeat, those five are averaged, and AUC/F1 are
computed once from the pair-level averages. The five repeats are five looks at
the same 2,882 pairs, not 14,410 observations.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils_choice.rq1_cv import N_INNER_FOLDS, inner_fold_map, outer_splits

#: Broad logarithmic grid. sklearn's ``C`` is *inverse* regularisation
#: strength, so this spans 1e-4 (very heavy penalty) to 1e4 (effectively
#: unpenalised). Boundary-selection frequency is reported, not ignored.
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

MAX_ITER = 5000
N_BOOT = 1000
FULL_DATA_CV_SEED = 999
N_FULL_DATA_FOLDS = 5


def _new_model(C):
    return LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                              solver="lbfgs", max_iter=MAX_ITER)


def fit_logreg(X, y, C):
    """Fit and assert convergence; returns the fitted estimator."""
    model = _new_model(C).fit(X, y)
    assert np.all(model.n_iter_ < MAX_ITER), \
        f"logistic regression hit max_iter={MAX_ITER} at C={C}"
    return model


def make_pair_design(player_mat, voter_ix, cand_ix, fit_rows):
    """Standardise player vectors on ``fit_rows`` only, then form pair rows.

    ``fit_rows`` is an array of row indices into ``player_mat`` -- the unique
    player-games of the training partition. Returns ``(X, scaler)`` where
    ``X`` is ``[scaled voter block | scaled candidate block]``.
    """
    scaler = StandardScaler().fit(player_mat[np.unique(fit_rows)])
    scaled = scaler.transform(player_mat)
    X = np.hstack([scaled[voter_ix], scaled[cand_ix]])
    assert np.isfinite(X).all(), "non-finite value in the pair design matrix"
    return X, scaler


def _fit_rows_for(voter_ix, cand_ix, mask):
    """Unique player-game rows referenced by the pair rows selected by mask."""
    return np.unique(np.concatenate([voter_ix[mask], cand_ix[mask]]))


def select_C_inner(player_mat, voter_ix, cand_ix, y, keys, comp_of_key,
                   train_mask, seed, c_grid=C_GRID):
    """Pick ``C`` by mean inner-CV ROC-AUC, grouped by player composition.

    The inner split is generated from the outer-training games only, so no
    outer-test pair can ever reach it, and the scaler is refit inside every
    inner fold on that fold's own training player-games.
    """
    train_keys = set(pd.unique(keys[train_mask]))
    inner_map = inner_fold_map(train_keys, comp_of_key, n_folds=N_INNER_FOLDS,
                               seed=seed)
    fold_of = np.array([inner_map.get(k, -1) for k in keys])

    scores = []
    for C in c_grid:
        aucs = []
        for f in range(N_INNER_FOLDS):
            itr = train_mask & (fold_of != f)
            ite = train_mask & (fold_of == f)
            if not itr.any() or not ite.any():
                continue
            if len(np.unique(y[itr])) < 2 or len(np.unique(y[ite])) < 2:
                continue
            X, _ = make_pair_design(player_mat, voter_ix, cand_ix,
                                    _fit_rows_for(voter_ix, cand_ix, itr))
            model = fit_logreg(X[itr], y[itr], C)
            aucs.append(roc_auc_score(y[ite], model.predict_proba(X[ite])[:, 1]))
        scores.append(float(np.mean(aucs)) if aucs else -np.inf)
    best = int(np.argmax(scores))
    return c_grid[best], scores[best]


def run_nested_cv(pairs, player_mat, voter_ix, cand_ix, fold_df, c_grid=C_GRID):
    """Repeated grouped nested CV. Returns ``(oof, fold_meta)``.

    ``oof`` has one row per (repeat, pair) with the held-out probability;
    ``fold_meta`` records the selected ``C`` and fold sizes.
    """
    y = pairs["label"].to_numpy()
    keys = pairs["key"].to_numpy(dtype=object)
    oof_rows, fold_meta = [], []

    for rep in sorted(fold_df["repeat"].unique()):
        test_keys_by_fold, comp_of_key = outer_splits(fold_df, rep)
        covered = np.zeros(len(pairs), dtype=bool)
        for f, test_keys in test_keys_by_fold.items():
            test_mask = np.array([k in test_keys for k in keys])
            train_mask = ~test_mask
            if not test_mask.any() or not train_mask.any():
                continue
            covered |= test_mask
            seed = rep * 100 + f
            C, inner_auc = select_C_inner(player_mat, voter_ix, cand_ix, y, keys,
                                          comp_of_key, train_mask, seed, c_grid)
            X, _ = make_pair_design(player_mat, voter_ix, cand_ix,
                                    _fit_rows_for(voter_ix, cand_ix, train_mask))
            model = fit_logreg(X[train_mask], y[train_mask], C)
            prob = model.predict_proba(X[test_mask])[:, 1]
            assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(), \
                "held-out probability outside [0, 1]"
            sub = pairs.loc[test_mask, ["key", "voter", "candidate", "label"]].copy()
            sub["prob"] = prob
            sub["repeat"] = rep
            sub["fold"] = f
            sub["composition_id"] = [comp_of_key[k] for k in sub["key"]]
            oof_rows.append(sub)
            fold_meta.append({"repeat": rep, "fold": f, "selected_C": C,
                              "inner_mean_auc": inner_auc,
                              "n_train_pairs": int(train_mask.sum()),
                              "n_test_pairs": int(test_mask.sum()),
                              "n_test_games": int(pd.unique(keys[test_mask]).size)})
        assert covered.all(), f"repeat {rep}: some pairs were never held out"

    oof = pd.concat(oof_rows, ignore_index=True)
    n_rep = fold_df["repeat"].nunique()
    assert len(oof) == len(pairs) * n_rep, \
        "each pair must be held out exactly once per repeat"
    return oof, pd.DataFrame(fold_meta)


def average_oof(oof):
    """One averaged held-out probability per (game, voter, candidate)."""
    avg = (oof.groupby(["key", "voter", "candidate"], as_index=False)
           .agg(label=("label", "first"), prob=("prob", "mean"),
                composition_id=("composition_id", "first"),
                n_repeats=("prob", "size")))
    assert avg["n_repeats"].nunique() == 1, "unequal repeat coverage across pairs"
    assert avg["label"].isin((0, 1)).all()
    return avg


def score(avg, threshold=0.5):
    """Held-out ROC-AUC (primary), F1 at 0.5 (secondary), prevalence."""
    y, p = avg["label"].to_numpy(), avg["prob"].to_numpy()
    return {"auc": float(roc_auc_score(y, p)),
            "f1": float(f1_score(y, (p >= threshold).astype(int), zero_division=0)),
            "prevalence": float(y.mean()), "n_pairs": int(len(y))}


# ------------------------------------------------------- clustered bootstrap -
def _games_by_composition(avg):
    return {c: g["key"].unique().tolist()
            for c, g in avg.groupby("composition_id")}


def _resample_indices(avg, n_boot, seed):
    """``n_boot`` draws of whole player compositions, as row-index arrays.

    The same draws are reused across feature sets, which is what makes the
    paired differences paired.
    """
    games_by_comp = _games_by_composition(avg)
    comps = np.array(sorted(games_by_comp))
    rows_by_game = {k: g.index.to_numpy() for k, g in avg.groupby("key")}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(comps, size=len(comps), replace=True)
        draws.append(np.concatenate([rows_by_game[k] for c in sampled
                                     for k in games_by_comp[c]]))
    return draws


def bootstrap_metrics(avg, n_boot=N_BOOT, seed=42, threshold=0.5):
    """Percentile intervals for AUC and F1, clustered on player composition."""
    avg = avg.reset_index(drop=True)
    draws = _resample_indices(avg, n_boot, seed)
    y, p = avg["label"].to_numpy(), avg["prob"].to_numpy()
    boot, failures = {"auc": [], "f1": []}, 0
    for idx in draws:
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            failures += 1
            continue
        boot["auc"].append(roc_auc_score(yy, pp))
        boot["f1"].append(f1_score(yy, (pp >= threshold).astype(int), zero_division=0))
    point = score(avg, threshold)
    out = {"n_pairs": point["n_pairs"], "prevalence": point["prevalence"],
           "n_boot": n_boot, "n_boot_failed": failures}
    for m in ("auc", "f1"):
        vals = np.asarray(boot[m])
        out[m] = point[m]
        out[f"{m}_ci_lo"] = float(np.percentile(vals, 2.5))
        out[f"{m}_ci_hi"] = float(np.percentile(vals, 97.5))
    return out


def bootstrap_paired_difference(avg_a, avg_b, label_a, label_b, n_boot=N_BOOT,
                                seed=42, threshold=0.5):
    """``b - a`` in AUC and F1, both models scored on the same resamples."""
    keys = ["key", "voter", "candidate"]
    merged = (avg_a[keys + ["label", "prob", "composition_id"]]
              .merge(avg_b[keys + ["prob"]], on=keys, suffixes=("_a", "_b"))
              .reset_index(drop=True))
    assert len(merged) == len(avg_a) == len(avg_b), "pair sets differ between models"
    draws = _resample_indices(merged, n_boot, seed)
    y = merged["label"].to_numpy()
    pa, pb = merged["prob_a"].to_numpy(), merged["prob_b"].to_numpy()

    rows = []
    for metric in ("auc", "f1"):
        def value(yy, pp):
            if metric == "auc":
                return roc_auc_score(yy, pp)
            return f1_score(yy, (pp >= threshold).astype(int), zero_division=0)

        diffs, failures = [], 0
        for idx in draws:
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                failures += 1
                continue
            diffs.append(value(yy, pb[idx]) - value(yy, pa[idx]))
        diffs = np.asarray(diffs)
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        rows.append({"comparison": f"{label_b} - {label_a}", "metric": metric,
                     "value_a": value(y, pa), "value_b": value(y, pb),
                     "mean_diff": value(y, pb) - value(y, pa),
                     "ci_lo": lo, "ci_hi": hi,
                     "interval_excludes_zero": bool(lo > 0 or hi < 0),
                     "n_boot": n_boot, "n_boot_failed": failures})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- coefficients
def select_C_full_data(player_mat, voter_ix, cand_ix, y, keys, comp_of_key,
                       c_grid=C_GRID, n_folds=N_FULL_DATA_FOLDS,
                       seed=FULL_DATA_CV_SEED):
    """Grouped K-fold CV choice of ``C`` on the complete dataset."""
    fold_map = inner_fold_map(set(pd.unique(keys)), comp_of_key,
                              n_folds=n_folds, seed=seed)
    fold_of = np.array([fold_map[k] for k in keys])
    scores = []
    for C in c_grid:
        aucs = []
        for f in range(n_folds):
            tr, te = fold_of != f, fold_of == f
            if not tr.any() or not te.any():
                continue
            X, _ = make_pair_design(player_mat, voter_ix, cand_ix,
                                    _fit_rows_for(voter_ix, cand_ix, tr))
            model = fit_logreg(X[tr], y[tr], C)
            aucs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
        scores.append(float(np.mean(aucs)) if aucs else -np.inf)
    best = int(np.argmax(scores))
    return c_grid[best], scores[best]


def fit_final_model(player_mat, voter_ix, cand_ix, y, C):
    """Standardise on the referenced player-games, then fit on every pair row."""
    fit_rows = _fit_rows_for(voter_ix, cand_ix, np.ones(len(voter_ix), dtype=bool))
    X, scaler = make_pair_design(player_mat, voter_ix, cand_ix, fit_rows)
    return fit_logreg(X, y, C), X, scaler


def coefficient_table(model, cols, side_names=(("voter", "Voter"),
                                               ("candidate", "Candidate"))):
    """One row per predictor: side, feature column, beta, odds ratio."""
    betas = model.coef_[0]
    assert len(betas) == 2 * len(cols)
    rows = []
    for block, (side_key, side_label) in enumerate(side_names):
        for i, col in enumerate(cols):
            b = float(betas[block * len(cols) + i])
            rows.append({"side": side_label, "side_key": side_key, "column": col,
                         "beta": b, "odds_ratio": float(np.exp(b))})
    return pd.DataFrame(rows)


def bootstrap_coefficients(pairs, player_mat, voter_ix, cand_ix, cols, C,
                           n_boot=N_BOOT, seed=7):
    """Refit scaler and model inside each composition resample, ``C`` fixed.

    Whole player compositions are resampled with replacement. A game drawn
    twice contributes two independent copies of its player-game rows, so the
    scaler sees the resampled distribution rather than the original one; the
    pair rows are remapped onto those copies by offset arithmetic rather than
    by rebuilding DataFrames, which keeps 1,000 replicates to a few seconds.
    """
    assert pairs.index.equals(pd.RangeIndex(len(pairs))), \
        "pairs must carry a clean RangeIndex aligned with voter_ix / cand_ix"
    y = pairs["label"].to_numpy()

    # per game: its player-matrix rows, and its pair rows expressed as
    # offsets into that game's own block of player rows. ``key`` is a tuple,
    # so games are grouped with pandas rather than by comparing against an
    # object array (which numpy would try to broadcast element-wise).
    game_players, game_pairs = {}, {}
    for key, g in pairs.groupby("key", sort=False):
        mask = np.zeros(len(pairs), dtype=bool)
        mask[g.index.to_numpy()] = True
        rows = np.unique(np.concatenate([voter_ix[mask], cand_ix[mask]]))
        local = {r: i for i, r in enumerate(rows)}
        game_players[key] = rows
        game_pairs[key] = (np.array([local[r] for r in voter_ix[mask]]),
                           np.array([local[r] for r in cand_ix[mask]]),
                           y[mask])

    games_by_comp = {}
    for key, comp in zip(pairs["key"], pairs["composition_id"]):
        games_by_comp.setdefault(comp, set()).add(key)
    games_by_comp = {c: sorted(v) for c, v in games_by_comp.items()}
    comps = np.array(sorted(games_by_comp))

    rng = np.random.default_rng(seed)
    betas, failures = [], []
    for b in range(n_boot):
        sampled = rng.choice(comps, size=len(comps), replace=True)
        blocks, v_parts, c_parts, y_parts, offset = [], [], [], [], 0
        for comp in sampled:
            for key in games_by_comp[comp]:
                rows = game_players[key]
                lv, lc, ly = game_pairs[key]
                blocks.append(player_mat[rows])
                v_parts.append(lv + offset)
                c_parts.append(lc + offset)
                y_parts.append(ly)
                offset += len(rows)
        boot_mat = np.vstack(blocks)
        bv, bc = np.concatenate(v_parts), np.concatenate(c_parts)
        by = np.concatenate(y_parts)
        try:
            X, _ = make_pair_design(boot_mat, bv, bc, np.arange(len(boot_mat)))
            betas.append(fit_logreg(X, by, C).coef_[0])
        except Exception as exc:                                   # noqa: BLE001
            failures.append({"boot_iter": b, "reason": str(exc)})
    betas = np.asarray(betas)
    assert len(betas) == 0 or betas.shape[1] == 2 * len(cols)
    return betas, pd.DataFrame(failures)


def coefficient_bootstrap_table(betas, cols, failures):
    """Percentile intervals aligned with :func:`coefficient_table`'s rows."""
    rows = []
    for block, side_label in enumerate(("Voter", "Candidate")):
        for i, col in enumerate(cols):
            b = betas[:, block * len(cols) + i]
            rows.append({"side": side_label, "column": col,
                         "beta_ci_lo": float(np.percentile(b, 2.5)),
                         "beta_ci_hi": float(np.percentile(b, 97.5)),
                         "odds_ratio_ci_lo": float(np.exp(np.percentile(b, 2.5))),
                         "odds_ratio_ci_hi": float(np.exp(np.percentile(b, 97.5))),
                         "n_boot_used": int(len(betas)),
                         "n_boot_failed": int(len(failures))})
    return pd.DataFrame(rows)


def penalty_boundary_report(fold_meta, c_grid=C_GRID):
    """How often the inner CV chose an endpoint of the grid."""
    sel = fold_meta["selected_C"]
    lo, hi = min(c_grid), max(c_grid)
    return {"n_fits": int(len(sel)),
            "at_lower_bound": int((sel == lo).sum()),
            "at_upper_bound": int((sel == hi).sum()),
            "pct_at_boundary": round(100 * float(((sel == lo) | (sel == hi)).mean()), 2),
            "median_selected_C": float(sel.median()),
            "distribution": sel.value_counts().sort_index().to_dict()}
