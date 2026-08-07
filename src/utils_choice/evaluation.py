"""Cross-validation, metrics and baselines for the vote choice models.

Metrics are the conventional ones for discrete choice: held-out log-likelihood
per choice, McFadden's pseudo-R-squared against the equal-shares null, and hit
rate. Splitting is grouped by game, so no game contributes to both training and
test.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .features import CIRCLE_FEATURE
from .model import ConditionalLogit

N_FOLDS = 5
CV_SEEDS = (0, 1, 2)
INNER_FOLDS = 3
RANDOM_STATE = 42
PENALTY_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
LEARNERS = ["clogit", "clogit_l2", "clogit_l1", "gbm"]


def folds_for(keys, seed, n_folds=N_FOLDS):
    ks = list(keys)
    np.random.default_rng(seed).shuffle(ks)
    return {k: i % n_folds for i, k in enumerate(ks)}


def build_frame(targets, ballot, include_circle=True):
    """Ballot rows for one voter (an LLM, or the human village) with the vote
    counts attached. ``targets`` maps a game key to {alternative: count}."""
    rows = []
    by_key = {k: g for k, g in ballot.groupby("key")}
    from .features import ALL_FEATURES
    for key, counts in targets.items():
        cand = by_key.get(key)
        if cand is None:
            continue
        if not include_circle:
            cand = cand[cand[CIRCLE_FEATURE] == 0]
        if cand.empty:
            continue
        for _, c in cand.iterrows():
            rows.append({"key": key, "player": c["player"],
                         "count": float(counts.get(c["player"], 0.0)),
                         **{f: c[f] for f in ALL_FEATURES}})
    frame = pd.DataFrame(rows).sort_values("key").reset_index(drop=True)
    return frame[frame.groupby("key")["count"].transform("sum") > 0].reset_index(drop=True)


def choice_metrics(df, q):
    """(mean log-likelihood per choice, hit rate, null log-likelihood)."""
    q = np.clip(q, 1e-12, None)
    n = df["count"].sum()
    ll = float((df["count"] * np.log(q)).sum() / n)
    size = df.groupby("key")["player"].transform("size").values
    ll_null = float(-(df["count"] * np.log(size)).sum() / n)
    hit = df.loc[df.assign(q=q).groupby("key")["q"].idxmax()]
    return ll, float(hit["count"].sum() / n), ll_null


def fit_predict(learner, tr, te, cols, asc_idx):
    """Fit on ``tr``, return choice probabilities for ``te``. The scaler and any
    penalty tuning use training rows only."""
    scaler = StandardScaler().fit(tr[cols])
    X_tr, X_te = scaler.transform(tr[cols]), scaler.transform(te[cols])
    if learner == "gbm":
        # not a choice model: a pointwise ranker, scored on hit rate only
        totals = tr.groupby("key")["count"].transform("sum").values
        w = np.r_[tr["count"].values, totals - tr["count"].values]
        Xd = np.vstack([X_tr, X_tr])
        y = np.r_[np.ones(len(tr)), np.zeros(len(tr))]
        keep = w > 0
        m = GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100,
                                       max_depth=2, subsample=0.8)
        m.fit(Xd[keep], y[keep], sample_weight=w[keep])
        s = np.clip(m.predict_proba(X_te)[:, 1], 1e-12, None)
        return s / pd.Series(s).groupby(te["key"].values).transform("sum").values

    unpen = [asc_idx] if asc_idx is not None else []
    l2 = l1 = 0.0
    if learner in ("clogit_l2", "clogit_l1"):
        best, best_ll = PENALTY_GRID[0], -np.inf
        ikeys = tr["key"].unique()
        ifold = tr["key"].map(dict(zip(ikeys, np.arange(len(ikeys)) % INNER_FOLDS)))
        for pen in PENALTY_GRID:
            lls = []
            for f in range(INNER_FOLDS):
                a, b = tr[ifold != f], tr[ifold == f]
                if b.empty:
                    continue
                sc = StandardScaler().fit(a[cols])
                kw = {"l2": pen} if learner == "clogit_l2" else {"l1": pen}
                mm = ConditionalLogit(unpenalized=unpen, **kw).fit(
                    sc.transform(a[cols]), a["count"], a["key"])
                lls.append(choice_metrics(
                    b, mm.predict_proba(sc.transform(b[cols]), b["key"]))[0])
            if lls and np.mean(lls) > best_ll:
                best_ll, best = float(np.mean(lls)), pen
        l2, l1 = (best, 0.0) if learner == "clogit_l2" else (0.0, best)

    m = ConditionalLogit(l2=l2, l1=l1, unpenalized=unpen).fit(
        X_tr, tr["count"], tr["key"])
    return m.predict_proba(X_te, te["key"])


def cross_validate(df, cols, learner, collect_perm=False, seeds=CV_SEEDS):
    """Grouped repeated CV. Returns the three metrics with their fold spread,
    and optionally per-feature held-out permutation drops."""
    asc_idx = cols.index(CIRCLE_FEATURE) if CIRCLE_FEATURE in cols else None
    ll_l, hit_l, r2_l = [], [], []
    perm = {f: [] for f in cols}
    for seed in seeds:
        fold = df["key"].map(folds_for(df["key"].unique(), seed))
        for f in range(N_FOLDS):
            tr, te = df[fold != f], df[fold == f]
            if te.empty:
                continue
            q = fit_predict(learner, tr, te, cols, asc_idx)
            ll, hit, ll0 = choice_metrics(te, q)
            ll_l.append(ll); hit_l.append(hit); r2_l.append(1 - ll / ll0)
            if collect_perm:
                rng = np.random.default_rng(RANDOM_STATE + f)
                for feat in cols:
                    tp = te.copy()
                    tp[feat] = rng.permutation(tp[feat].values)
                    perm[feat].append(
                        ll - choice_metrics(tp, fit_predict(learner, tr, tp, cols,
                                                            asc_idx))[0])
    return ({"mean_ll": (np.mean(ll_l), np.std(ll_l)),
             "mcfadden_r2": (np.mean(r2_l), np.std(r2_l)),
             "hit_rate": (np.mean(hit_l), np.std(hit_l))}, perm)


def run_grid(df, blocks, learners=LEARNERS, include_circle=True, label=""):
    """Every (block, learner) cell. ``gbm`` is not a choice model, so its
    likelihood columns are blanked rather than reported as comparable."""
    from .features import cols_for
    rows = []
    for block in blocks:
        for learner in learners:
            res, _ = cross_validate(df, cols_for(block, include_circle), learner)
            rows.append({"model": label, "block": block, "learner": learner,
                         "mean_ll": round(res["mean_ll"][0], 3),
                         "ll_std": round(res["mean_ll"][1], 3),
                         "mcfadden_r2": round(res["mcfadden_r2"][0], 3),
                         "hit_rate": round(res["hit_rate"][0], 3),
                         "hit_std": round(res["hit_rate"][1], 3)})
    out = pd.DataFrame(rows)
    out.loc[out["learner"] == "gbm", ["mean_ll", "ll_std", "mcfadden_r2"]] = np.nan
    return out


# ------------------------------------------------------------- baselines -----
def rule_hit_rate(df, raw, n_draws=25, seed=0):
    """Hit rate of a deterministic rule, averaged over random tie-breaks.

    Ties are common (47% of players are never accused), so a single jitter draw
    moved this by up to 0.04 between runs -- enough to change whether the fitted
    model looks better than the rule.
    """
    rng = np.random.default_rng(seed)
    raw = np.asarray(raw, float)
    out = []
    for _ in range(n_draws):
        s = np.clip(raw + rng.uniform(0, 1e-6, len(raw)), 1e-12, None)
        q = s / pd.Series(s).groupby(df["key"].values).transform("sum").values
        out.append(choice_metrics(df, q)[1])
    return float(np.mean(out))


def reference_points(df, shares, label="", crowd_modal=None):
    """Null model, ceiling and rule baselines.

    ``shares`` is the observed vote distribution per game. The ceiling is the
    probability that two independent votes agree: test-retest reliability for a
    single LLM resampled, inter-voter agreement for a group of people. Both
    bound any model, but they are not the same construct.
    """
    n = df["count"].sum()
    size = df.groupby("key")["player"].transform("size").values
    out = {"model": label,
           "null_ll": float(-(df["count"] * np.log(size)).sum() / n),
           "null_hit_rate": float((df["count"] / size).sum() / n),
           "ceiling_agreement": float(np.mean([sum(v * v for v in s.values())
                                               for s in shares])),
           "ceiling_best_possible": float(np.mean([max(s.values()) for s in shares])),
           "irreducible_entropy": float(np.mean(
               [-sum(v * np.log(v) for v in s.values()) for s in shares])),
           "most_talkative_hit": rule_hit_rate(df, df["n_utterances"]),
           "most_accused_hit": rule_hit_rate(df, df["werewolf_count"])}
    if crowd_modal is not None:
        hit = np.array([1.0 if r["player"] in crowd_modal.get(r["key"], set()) else 0.0
                        for _, r in df.iterrows()])
        out["crowd_modal_hit"] = rule_hit_rate(df, hit)
    return pd.DataFrame([out])
