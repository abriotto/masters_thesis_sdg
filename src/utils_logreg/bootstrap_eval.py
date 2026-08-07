"""Uncertainty for the fixed-split pairwise models.

Lai's protocol fits on train, tunes on val and reports one number on test, which
leaves no way to tell a real difference between models from sampling noise. This
module adds that without disturbing the protocol: the fitted models and their
tuned thresholds are left exactly as they are, and the **test set** is
resampled.

Games are the resampling unit, not rows. The pairs inside a game share a
dialogue and a roster, so resampling rows would treat dependent observations as
independent and give intervals that are far too narrow.

Differences between models are bootstrapped **paired** -- both models are scored
on the same resampled games -- which is what makes the interval a test of the
difference rather than of two separate numbers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def _game_ids(rows):
    return np.array([f"{r['dataset']}|{r['session_key']}|{r['game_id']}" for r in rows])


def _metrics(y, p, threshold):
    out = {"f1": f1_score(y, (p >= threshold).astype(int), zero_division=0)}
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    return out


def bootstrap_test_metrics(test_rows, predictions, thresholds, n_boot=2000, seed=42):
    """Percentile CIs for each model's test metrics, and for every pairwise
    difference between them.

    ``predictions`` maps a model name to its predicted probabilities on
    ``test_rows``; ``thresholds`` maps the same names to the threshold tuned on
    the validation split.
    """
    y = np.array([r["label"] for r in test_rows])
    games = _game_ids(test_rows)
    unique_games = np.unique(games)
    idx_by_game = {g: np.flatnonzero(games == g) for g in unique_games}
    rng = np.random.default_rng(seed)

    names = list(predictions)
    draws = {n: {"f1": [], "auc": []} for n in names}
    for _ in range(n_boot):
        pick = rng.choice(unique_games, size=len(unique_games), replace=True)
        idx = np.concatenate([idx_by_game[g] for g in pick])
        if len(np.unique(y[idx])) < 2:      # a resample with one class only
            continue
        for n in names:
            m = _metrics(y[idx], predictions[n][idx], thresholds[n])
            draws[n]["f1"].append(m["f1"])
            draws[n]["auc"].append(m["auc"])

    rows = []
    for n in names:
        point = _metrics(y, predictions[n], thresholds[n])
        for metric in ("f1", "auc"):
            d = np.array(draws[n][metric], dtype=float)
            rows.append({"model": n, "metric": metric,
                         "value": round(point[metric], 3),
                         "ci_lo": round(float(np.percentile(d, 2.5)), 3),
                         "ci_hi": round(float(np.percentile(d, 97.5)), 3)})
    per_model = pd.DataFrame(rows)

    diffs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for metric in ("f1", "auc"):
                d = (np.array(draws[b][metric], dtype=float)
                     - np.array(draws[a][metric], dtype=float))
                lo, hi = np.percentile(d, [2.5, 97.5])
                point = (_metrics(y, predictions[b], thresholds[b])[metric]
                         - _metrics(y, predictions[a], thresholds[a])[metric])
                diffs.append({"comparison": f"{b} - {a}", "metric": metric,
                              "difference": round(point, 3),
                              "ci_lo": round(float(lo), 3),
                              "ci_hi": round(float(hi), 3),
                              "excludes_zero": bool(lo > 0 or hi < 0)})
    return per_model, pd.DataFrame(diffs)
