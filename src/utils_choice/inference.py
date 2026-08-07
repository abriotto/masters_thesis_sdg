"""Inference for the vote choice models.

Cross-validation spread is a stability measure, not a significance test, so:

* **cluster-robust standard errors** (Huber-White, clustered by game), because
  each game contributes several correlated choices and model-based errors would
  be too small;
* **robust Wald tests** for nested feature blocks;
* **stability selection** (Meinshausen & Buhlmann 2010) for which features are
  worth interpreting;
* **game-level bootstrap** intervals for out-of-sample block differences.

A Wald test asks whether a coefficient is reliably non-zero; stability
selection asks whether a feature survives when the model is forced to be
sparse. They answer different questions and can disagree.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from .evaluation import (CV_SEEDS, N_FOLDS, choice_metrics, fit_predict,
                         folds_for)
from .features import CIRCLE_FEATURE, cols_for
from .model import ConditionalLogit

N_BOOT = 2000
STABILITY_B = 200
STABILITY_TARGET_K = 5
STABILITY_THRESHOLD = 0.6      # Meinshausen & Buhlmann suggest 0.6-0.9


def _unpen(cols):
    return [cols.index(CIRCLE_FEATURE)] if CIRCLE_FEATURE in cols else []


def fit_with_vcov(df, cols):
    """MLE plus the game-clustered sandwich covariance."""
    X = StandardScaler().fit_transform(df[cols])
    counts = df["count"].values
    groups = pd.factorize(df["key"])[0]
    m = ConditionalLogit(unpenalized=_unpen(cols)).fit(X, counts, groups)
    q = m.predict_proba(X, groups)
    n_g = np.bincount(groups, weights=counts)
    k = X.shape[1]
    H, S = np.zeros((k, k)), np.zeros((len(n_g), k))
    for g in range(len(n_g)):
        s = groups == g
        Xg, pg = X[s], q[s]
        xbar = pg @ Xg
        H += n_g[g] * ((Xg * pg[:, None]).T @ Xg - np.outer(xbar, xbar))
        S[g] = (counts[s] - n_g[g] * pg) @ Xg
    Hinv = np.linalg.pinv(H)
    return m.coef_, Hinv @ (S.T @ S) @ Hinv, Hinv, len(n_g)


def coef_table(df, cols):
    """Odds ratios with cluster-robust confidence intervals and Wald p-values,
    Bonferroni-corrected within the block (every feature is tested)."""
    b, V, _, n_clusters = fit_with_vcov(df, cols)
    se = np.sqrt(np.diag(V))
    z = b / se
    p = 2 * stats.norm.sf(np.abs(z))
    tab = pd.DataFrame({"feature": cols, "odds_ratio": np.exp(b),
                        "ci_lo": np.exp(b - 1.96 * se),
                        "ci_hi": np.exp(b + 1.96 * se), "z": z, "p": p,
                        "p_bonferroni": np.minimum(1.0, p * len(cols))})
    return tab, V, n_clusters


def block_test(df, small, large="C_both", include_circle=True):
    """Robust Wald test that the coefficients ``large`` adds to ``small`` are
    jointly zero. Wald rather than a likelihood ratio, because LR is not valid
    under a sandwich covariance."""
    cs = cols_for(small, include_circle)
    cl = cols_for(large, include_circle)
    added = [c for c in cl if c not in cs]
    tab, V, _ = coef_table(df, cl)
    idx = [cl.index(a) for a in added]
    b = np.log(tab["odds_ratio"].values[idx])
    W = float(b @ np.linalg.pinv(V[np.ix_(idx, idx)]) @ b)
    return W, len(idx), float(stats.chi2.sf(W, len(idx)))


def stability_selection(df, cols, n_subsamples=STABILITY_B, frac=0.5,
                        target_k=STABILITY_TARGET_K, seed=42):
    """Refit the lasso on random half-samples of games; record how often each
    feature survives.

    The penalty is bisected to retain about ``target_k`` features on the full
    sample. At the CV-optimal penalty nothing is dropped at this n/p ratio, so
    the frequencies would all be 1 and say nothing.
    """
    X = StandardScaler().fit_transform(df[cols])
    counts, keys = df["count"].values, df["key"].values
    asc = _unpen(cols)
    lo, hi = 1e-4, 1e3
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        nz = np.sum(ConditionalLogit(l1=mid, unpenalized=asc).fit(
            X, counts, pd.factorize(keys)[0]).coef_ != 0) - len(asc)
        lo, hi = (mid, hi) if nz > target_k else (lo, mid)
    lam = np.sqrt(lo * hi)

    rng = np.random.default_rng(seed)
    games, hits = np.unique(keys), np.zeros(len(cols))
    for _ in range(n_subsamples):
        pick = set(rng.choice(games, size=int(frac * len(games)), replace=False))
        s = np.array([k in pick for k in keys])
        hits += (ConditionalLogit(l1=lam, unpenalized=asc).fit(
            X[s], counts[s], pd.factorize(keys[s])[0]).coef_ != 0)
    return pd.DataFrame({"feature": cols,
                         "selection_freq": hits / n_subsamples,
                         "lambda": round(lam, 3)}), lam


def oof_loglik_by_game(df, cols, learner="clogit_l2", seeds=CV_SEEDS):
    """Out-of-fold mean log-likelihood per choice for each game, averaged over
    seeds so each game gives the one number the bootstrap resamples."""
    asc_idx = cols.index(CIRCLE_FEATURE) if CIRCLE_FEATURE in cols else None
    per_seed = []
    for seed in seeds:
        fold = df["key"].map(folds_for(df["key"].unique(), seed))
        parts = []
        for f in range(N_FOLDS):
            tr, te = df[fold != f], df[fold == f]
            if te.empty:
                continue
            q = np.clip(fit_predict(learner, tr, te, cols, asc_idx), 1e-12, None)
            t = te.assign(q=q)
            parts.append(t.groupby("key").apply(
                lambda g: float((g["count"] * np.log(g["q"])).sum()
                                / g["count"].sum()), include_groups=False))
        per_seed.append(pd.concat(parts))
    return pd.concat(per_seed, axis=1).mean(axis=1)


def bootstrap_paired_diff(ll_a, ll_b, n_boot=N_BOOT, seed=42):
    """Percentile CI for a difference in held-out log-likelihood, resampling
    games -- the independent unit.

    The CV folds are held fixed, so this captures variation across games but not
    refit variability. That is the standard practical compromise, and the reason
    the interval is a lower bound on the true uncertainty.
    """
    d = (ll_b - ll_a).dropna().values
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def block_bootstrap(df, comparisons, null_ll, include_circle=True, label=""):
    """Bootstrap intervals for a list of (block_a, block_b) comparisons."""
    oof = {}
    rows = []
    for a, b in comparisons:
        for blk in (a, b):
            if blk not in oof:
                oof[blk] = oof_loglik_by_game(df, cols_for(blk, include_circle))
        m, lo, hi = bootstrap_paired_diff(oof[a], oof[b])
        rows.append({"model": label, "comparison": f"{b} - {a}",
                     "delta_mean_ll": round(m, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4),
                     "delta_pseudo_r2": round(m / null_ll, 4),
                     "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def iia_test(df, cols):
    """Independence of irrelevant alternatives -- the assumption the conditional
    logit rests on, and the standard objection to it.

    Dropping an alternative should leave the coefficients consistent. Here the
    dropped alternative is 'No Werewolf', which is also the substantively
    interesting restriction: does having an abstention option change how the
    voter weighs the players against each other?

    Reported two ways: the Hausman-McFadden statistic (using the model-based
    covariances the test assumes) and the coefficients side by side. The
    statistic is not guaranteed positive in finite samples -- a known property,
    reported rather than hidden.
    """
    shared = [c for c in cols if c != CIRCLE_FEATURE]
    b_f, _, V_f, _ = fit_with_vcov(df, cols)
    b_f = pd.Series(b_f, index=cols)[shared].values
    Vf = pd.DataFrame(V_f, index=cols, columns=cols).loc[shared, shared].values

    sub = df[df[CIRCLE_FEATURE] == 0].copy()
    sub = sub[sub.groupby("key")["count"].transform("sum") > 0]
    b_r, _, Vr, _ = fit_with_vcov(sub, shared)

    diff = b_r - b_f
    stat = float(diff @ np.linalg.pinv(Vr - Vf) @ diff)
    p = float(stats.chi2.sf(stat, len(shared))) if stat > 0 else np.nan
    verdict = ("inconclusive (statistic not positive - a known finite-sample "
               "outcome)" if not np.isfinite(p)
               else ("IIA rejected" if p < 0.05 else "no evidence against IIA"))
    tab = pd.DataFrame({"feature": shared, "or_full": np.exp(b_f),
                        "or_restricted": np.exp(b_r),
                        "abs_log_shift": np.abs(diff)})
    summary = pd.DataFrame([{"hausman_chi2": round(stat, 2), "df": len(shared),
                             "p_value": (round(p, 3) if np.isfinite(p) else np.nan),
                             "verdict": verdict,
                             "max_abs_log_shift": round(tab["abs_log_shift"].max(), 3)}])
    return summary, tab
