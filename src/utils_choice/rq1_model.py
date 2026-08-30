"""Phase 1 modelling: representations, standardisation, ridge/L1 conditional
logit, metrics, and the grouped nested-CV driver for the RQ1 surrogate.

Reuses the validated :class:`utils_choice.model.ConditionalLogit` unchanged.
Everything else here is new: the four feature representations, the
train-only standardisation rule that keeps the ``No Werewolf`` alternative's
feature vector at exactly zero, the null (ASC-only) baseline, the two
fidelity metrics (held-out log loss and stochastic top-choice agreement with
exact-tie averaging), and the nested-CV loop that ties them together.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils_choice.model import ConditionalLogit
from utils_choice.io import llm_vote_targets, STOCHASTIC_RUNS
from utils_choice.rq1_features import STRATEGY_FEATURES, ENRICHED_FEATURES, ALL_FEATURES_13
from utils_choice.rq1_cv import outer_splits, inner_fold_map, N_INNER_FOLDS

CIRCLE_OPTION = "__NO_WEREWOLF__"
ASC_COL = "is_no_werewolf"
#: Ridge (L2) penalty search grid. Extended upward from the original
#: (0.01 ... 100) after an audit found the inner CV selecting the old maximum
#: in 619 of 900 outer fits -- up to 96-100% for the temporal specifications --
#: which meant the optimum was not demonstrably inside the searched range.
L2_PENALTY_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

#: Lasso (L1) penalty grid, deliberately left at the original range: the L1
#: analysis is a descriptive feature-selection sensitivity check and was not
#: boundary-saturated, so its published selection frequencies stay comparable.
L1_PENALTY_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

#: Backwards-compatible alias (the original single shared grid).
PENALTY_GRID = L1_PENALTY_GRID


def penalty_grid_for(kind, penalty_grid=None):
    """Default penalty grid for a penalty type, unless one is given.

    The two grids differ, so the choice must key off ``kind`` rather than a
    single shared constant -- otherwise extending the ridge range would
    silently also change the lasso results.
    """
    if penalty_grid is not None:
        return penalty_grid
    if kind == "l2":
        return L2_PENALTY_GRID
    if kind == "l1":
        return L1_PENALTY_GRID
    raise ValueError(f"unknown penalty kind {kind!r}")

BLOCKS = {"strategy": STRATEGY_FEATURES, "enriched": ENRICHED_FEATURES,
          "combined": ALL_FEATURES_13}

PRODUCED_FEATURES = STRATEGY_FEATURES + ["werewolf_accusations_made",
                                          "deception_accusations_made",
                                          "claims_werewolf", "claims_tanner",
                                          "claims_night_action_role"]
RECEIVED_FEATURES = ["werewolf_accusations_received", "deception_accusations_received"]
assert set(PRODUCED_FEATURES) | set(RECEIVED_FEATURES) == set(ALL_FEATURES_13)


# ---------------------------------------------------------------- rates -----
def add_rate_columns(df):
    """Adds *_rate, *_early_rate, *_late_rate for every one of the 13
    features, plus other_turns / other_turns_early / other_turns_late.

    Returns (df, zero_denominator_report). Every produced-feature rate uses
    the candidate's own turns as denominator; every received-feature rate
    uses the OTHER rostered players' turns. 0/0 is defined as rate 0, and
    every such case is asserted to have a zero numerator (never silently
    substituted) and recorded in the report.
    """
    df = df.copy()
    for h in ("", "_early", "_late"):
        df[f"game_turns{h}"] = df.groupby("key")[f"turns{h}"].transform("sum")
        df[f"other_turns{h}"] = df[f"game_turns{h}"] - df[f"turns{h}"]
        assert (df[f"other_turns{h}"] > 0).all(), \
            f"other_turns{h} is zero for some player-game -- denominator failure"

    zero_rows = []

    def rate(numer_col, denom_col, out_col, feat, half):
        num = df[numer_col].values.astype(float)
        den = df[denom_col].values.astype(float)
        zero = den == 0
        if zero.any():
            bad = df.loc[zero & (num != 0)]
            assert bad.empty, (f"{numer_col} is nonzero where {denom_col}==0 "
                               f"(rows: {bad[['key','player']].to_dict('records')})")
            for _, r in df.loc[zero].iterrows():
                zero_rows.append({"feature": feat, "half": half or "overall",
                                  "key": r["key"], "player": r["player"],
                                  "numerator_col": numer_col, "denominator_col": denom_col})
        out = np.zeros(len(df))
        out[~zero] = 100.0 * num[~zero] / den[~zero]
        df[out_col] = out

    for feat in PRODUCED_FEATURES:
        rate(feat, "turns", f"{feat}_rate", feat, "")
        rate(f"{feat}_early", "turns_early", f"{feat}_early_rate", feat, "early")
        rate(f"{feat}_late", "turns_late", f"{feat}_late_rate", feat, "late")
    for feat in RECEIVED_FEATURES:
        rate(feat, "other_turns", f"{feat}_rate", feat, "")
        rate(f"{feat}_early", "other_turns_early", f"{feat}_early_rate", feat, "early")
        rate(f"{feat}_late", "other_turns_late", f"{feat}_late_rate", feat, "late")

    return df, pd.DataFrame(zero_rows)


# ---------------------------------------------------------- representations -
def feature_columns(representation, block):
    """Column names for one (representation, block) cell of the grid."""
    feats = BLOCKS[block]
    if representation == "count_overall":
        return list(feats)
    if representation == "rate_overall":
        return [f"{f}_rate" for f in feats]
    if representation == "count_temporal":
        return [f"{f}_early" for f in feats] + [f"{f}_late" for f in feats]
    if representation == "rate_temporal":
        return [f"{f}_early_rate" for f in feats] + [f"{f}_late_rate" for f in feats]
    raise ValueError(representation)


REPRESENTATIONS = ("count_overall", "rate_overall", "count_temporal", "rate_temporal")


# --------------------------------------------------------------- ballot -----
def build_model_frame(feat_df, roster, comp_of_key, votes, model_label,
                      run_labels=STOCHASTIC_RUNS):
    """Long frame: one row per (game key, alternative), for one LLM.

    ``feat_df`` already carries every raw + rate + early/late column (from
    :func:`add_rate_columns`). Player rows keep those values; the circle
    (``No Werewolf``) row is exactly zero on every persuasion column and
    carries ``is_no_werewolf = 1``. ``count`` is the pooled vote count over
    ``run_labels`` (0..3 for the stochastic runs).
    """
    targets, _, _ = llm_vote_targets(votes[votes["model"] == model_label],
                                     roster, run_labels)
    targets = targets.get(model_label, {})
    value_cols = [c for c in feat_df.columns
                 if c not in ("key", "source", "session", "game", "player")]

    rows = []
    by_key = {k: g for k, g in feat_df.groupby("key")}
    for key, counts in targets.items():
        cand = by_key.get(key)
        if cand is None or key not in comp_of_key:
            continue
        for _, r in cand.iterrows():
            row = {c: float(r[c]) for c in value_cols}
            row.update({"key": key, "player": r["player"], ASC_COL: 0.0,
                       "count": float(counts.get(r["player"], 0.0)),
                       "composition_id": comp_of_key[key]})
            rows.append(row)
        circle = {c: 0.0 for c in value_cols}
        circle.update({"key": key, "player": CIRCLE_OPTION, ASC_COL: 1.0,
                       "count": float(counts.get(CIRCLE_OPTION, 0.0)),
                       "composition_id": comp_of_key[key]})
        rows.append(circle)
    frame = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
    frame = frame[frame.groupby("key")["count"].transform("sum") > 0].reset_index(drop=True)
    tot = frame.groupby("key")["count"].sum()
    assert np.allclose(tot.values, 3.0), "vote counts must sum to 3 per game"
    return frame


# ------------------------------------------------------- design matrices ----
def make_design(train_frame, apply_frame, cols):
    """Fit a StandardScaler on TRAIN player rows only (``is_no_werewolf==0``),
    transform ``apply_frame``, then force every circle row's persuasion
    columns back to exactly zero. Appends the (unscaled) ASC column last;
    its index is returned as the sole unpenalised coefficient.
    """
    if not cols:
        X = apply_frame[[ASC_COL]].values.astype(float)
        return X, [0]
    player_mask_tr = train_frame[ASC_COL].values == 0.0
    scaler = StandardScaler().fit(train_frame.loc[player_mask_tr, cols])
    X_feat = scaler.transform(apply_frame[cols])
    circle_mask = apply_frame[ASC_COL].values == 1.0
    X_feat[circle_mask, :] = 0.0
    X = np.column_stack([X_feat, apply_frame[ASC_COL].values.astype(float)])
    return X, [len(cols)]


# ------------------------------------------------------------- fitting ------
def _fit(X, counts, groups, penalty, unpenalized, kind):
    kw = {"l2": penalty} if kind == "l2" else {"l1": penalty}
    return ConditionalLogit(unpenalized=unpenalized, **kw).fit(X, counts, groups)


def choice_loss(frame, q):
    """Mean held-out log loss per choice (weighted by vote share), matching
    the corpus-wide convention used elsewhere in this project."""
    q = np.clip(q, 1e-12, None)
    n = frame["count"].sum()
    return float((frame["count"] * np.log(q)).sum() / n) * -1.0


def per_game_metrics(frame, q):
    """Per-game log loss and top-choice agreement (exact-tie averaged)."""
    q = np.clip(q, 1e-12, None)
    out = []
    for key, g in frame.assign(q=q).groupby("key"):
        share = (g["count"] / g["count"].sum()).values
        loss = float(-(share * np.log(g["q"].values)).sum())
        qmax = g["q"].values.max()
        top = np.isclose(g["q"].values, qmax, rtol=1e-9, atol=1e-12)
        agreement = float(share[top].mean())
        out.append({"key": key, "loss": loss, "agreement": agreement,
                   "n_alt": len(g), "composition_id": g["composition_id"].iloc[0]})
    return pd.DataFrame(out)


def fit_predict_penalized(train, test, cols, kind, penalty_grid=None,
                          inner_seed=0):
    """Inner-CV-tuned ridge/L1 fit on ``train``, prediction on ``test``.

    Inner CV is grouped by composition, restricted to compositions present
    in ``train`` (never touches ``test``). Scaling is refit inside every
    inner fold on that fold's training rows only, and again on the full
    outer-training set for the final fit -- ``test`` never contributes to
    scaling or penalty choice.
    """
    if not cols:
        X_tr, unpen = make_design(train, train, cols)
        X_te, _ = make_design(train, test, cols)
        m = _fit(X_tr, train["count"].values, train["key"].values, 0.0, unpen, "l2")
        return m.predict_proba(X_te, test["key"].values), None

    penalty_grid = penalty_grid_for(kind, penalty_grid)

    comp_of = dict(zip(train["key"], train["composition_id"]))
    train_keys = set(train["key"].unique())
    inner_map = inner_fold_map(train_keys, comp_of, n_folds=N_INNER_FOLDS, seed=inner_seed)
    fold_of = train["key"].map(inner_map)

    best_pen, best_ll = penalty_grid[0], -np.inf
    for pen in penalty_grid:
        lls = []
        for f in range(N_INNER_FOLDS):
            itr, ite = train[fold_of != f], train[fold_of == f]
            if ite.empty or itr.empty:
                continue
            X_itr, unpen = make_design(itr, itr, cols)
            X_ite, _ = make_design(itr, ite, cols)
            m = _fit(X_itr, itr["count"].values, itr["key"].values, pen, unpen, kind)
            q = m.predict_proba(X_ite, ite["key"].values)
            lls.append(-choice_loss(ite, q))
        if lls and np.mean(lls) > best_ll:
            best_ll, best_pen = float(np.mean(lls)), pen

    X_tr, unpen = make_design(train, train, cols)
    X_te, _ = make_design(train, test, cols)
    m = _fit(X_tr, train["count"].values, train["key"].values, best_pen, unpen, kind)
    q = m.predict_proba(X_te, test["key"].values)
    return q, best_pen, m


def run_nested_cv(frame, fold_df, cols, kind="l2", penalty_grid=None,
                  collect_coef=False):
    """Full repeated grouped nested CV for one (model, representation, block)
    cell. Returns (per_game_df with one row per (repeat, game key), fold_meta
    list of per-fold diagnostics[, coef_rows if collect_coef])."""
    rows, fold_meta, coef_rows = [], [], []
    for rep in sorted(fold_df["repeat"].unique()):
        test_keys_by_fold, comp_of_key = outer_splits(fold_df, rep)
        for f, test_keys in test_keys_by_fold.items():
            train = frame[~frame["key"].isin(test_keys)]
            test = frame[frame["key"].isin(test_keys)]
            if test.empty or train.empty:
                continue
            result = fit_predict_penalized(train, test, cols, kind,
                                           penalty_grid, inner_seed=rep * 100 + f)
            q, best_pen = result[0], result[1]
            pg = per_game_metrics(test, q)
            pg["repeat"] = rep
            pg["fold"] = f
            rows.append(pg)
            fold_meta.append({"repeat": rep, "fold": f, "n_test_games": test["key"].nunique(),
                             "selected_penalty": best_pen,
                             "mean_loss": pg["loss"].mean(), "mean_agreement": pg["agreement"].mean()})
            if collect_coef and len(result) > 2 and cols:
                m = result[2]
                for c, b in zip(cols, m.coef_[:len(cols)]):
                    coef_rows.append({"repeat": rep, "fold": f, "feature": c,
                                     "coef": float(b), "penalty": best_pen})
    per_game = pd.concat(rows, ignore_index=True)
    # Every game in ``frame`` must be tested exactly once per repeat. Checked
    # against the frame's own game count rather than a hard-coded 191, so a
    # matched BASE/FT subset (which drops games lacking a complete 3-vote
    # ballot on one side) is still fully verified.
    n_games = frame["key"].nunique()
    assert per_game.groupby("repeat")["key"].nunique().eq(n_games).all(), \
        f"not every game covered exactly once per repeat (expected {n_games})"
    if collect_coef:
        return per_game, pd.DataFrame(fold_meta), pd.DataFrame(coef_rows)
    return per_game, pd.DataFrame(fold_meta)


def select_penalty_full_data(frame, cols, kind="l2", n_folds=5, seed=999,
                             penalty_grid=None):
    """Grouped K-fold CV penalty choice on the WHOLE dataset (for the final,
    full-data coefficient models -- distinct from the nested nested-CV loop,
    which re-selects a penalty inside every outer-training partition)."""
    penalty_grid = penalty_grid_for(kind, penalty_grid)
    comp_of = dict(zip(frame["key"], frame["composition_id"]))
    keys = set(frame["key"].unique())
    fold_map = inner_fold_map(keys, comp_of, n_folds=n_folds, seed=seed)
    fold_of = frame["key"].map(fold_map)
    best_pen, best_ll = penalty_grid[0], -np.inf
    for pen in penalty_grid:
        lls = []
        for f in range(n_folds):
            tr, te = frame[fold_of != f], frame[fold_of == f]
            if te.empty or tr.empty:
                continue
            X_tr, unpen = make_design(tr, tr, cols)
            X_te, _ = make_design(tr, te, cols)
            m = _fit(X_tr, tr["count"].values, tr["key"].values, pen, unpen, kind)
            q = m.predict_proba(X_te, te["key"].values)
            lls.append(-choice_loss(te, q))
        if lls and np.mean(lls) > best_ll:
            best_ll, best_pen = float(np.mean(lls)), pen
    return best_pen


def fit_final(frame, cols, kind, penalty):
    """Fit on ALL rows in ``frame`` with a fixed penalty (standardisation
    uses ``frame``'s own player rows, via :func:`make_design`)."""
    X, unpen = make_design(frame, frame, cols)
    return _fit(X, frame["count"].values, frame["key"].values, penalty, unpen, kind)


def per_game_avg_over_repeats(per_game_df):
    """One row per game: loss/agreement averaged over the 5 repeats."""
    return (per_game_df.groupby(["key", "composition_id"])
            .agg(loss=("loss", "mean"), agreement=("agreement", "mean"))
            .reset_index())
