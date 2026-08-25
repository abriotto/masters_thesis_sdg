"""Replication of Lai et al. (2023)'s human vote-prediction experiment.

Scope: reproduce the original binary pairwise experiment closely enough to
show that our reconstruction of the corpus is sound. Nothing here feeds the
thesis comparison, which lives in :mod:`utils_choice.human_pairs`.

The construction below is taken from the authors' released implementation
(``SALT-NLP/PersuationGames``, ``baselines/read_data.py`` and
``baselines/main_deduction.py``), not from the paper text, because the paper
specifies neither the pair enumeration nor the hyperparameters:

* **All n^2 ordered pairs per game.** ``i == j`` self-pairs are *not* skipped.
* **Players whose ``startRoles`` entry is ``Moderator`` are skipped**, on both
  the voter and the candidate side. There are exactly three such players in
  the corpus, all in 5-player Ego4D games (2 in val, 1 in test), and each one
  costs ``25 - 16 = 9`` rows. This single rule is what reconciles our counts
  with the paper's 2741 / 427 / 827 -- see :func:`split_row_counts`.
* **Non-votes stay in the design matrix.** ``votingOutcome`` records a voter
  who abstained (voted the centre cards, or was moderating) as the string
  ``"NA"`` / ``"N/A"``; the released code maps those to the impossible index
  6, so such a voter contributes ``n`` all-negative rows rather than being
  dropped. Two entries are out-of-range integers and behave the same way.
* **Hardcoded hyperparameters.** ``C=1.4, class_weight={0:1, 1:4.2}`` is
  hardcoded in ``main_deduction.py`` above a commented-out search. It is a
  *released-code* value, not a paper-reported one.
* **Hard-label AUC.** The released code calls
  ``roc_auc_score(y_true, y_score=model.predict(X))``, i.e. it passes hard 0/1
  predictions where a score is expected, which makes the reported 54.7% a
  balanced accuracy rather than a ROC-AUC. We report both: ``auc_hard`` to
  match their number, ``auc_prob`` (from ``predict_proba``) as the standard
  quantity the thesis uses everywhere else.

Only the **strategies-only** model is reproduced. Lai et al. also report a
"strategies + voter start-role" variant, but ground-truth roles are exactly
what the LLM observer cannot see, so that variant has no counterpart in the
thesis comparison and is not built here. ``startRoles`` is still read, purely
to identify the three moderators.
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from utils_logreg.lai2023_loading import (get_game_id, get_outcome_record,
                                          get_session_key)

#: Lai's 7-dimensional strategy vocabulary. ``No Strategy`` is an explicit
#: annotation label in the corpus (10,793 utterances) and is part of their
#: representation, so it is kept here and only here.
LAI_STRATEGIES = ("No Strategy", "Identity Declaration", "Accusation",
                  "Interrogation", "Call for Action", "Defense", "Evidence")

MODERATOR_ROLE = "Moderator"

#: Paper-reported figures (Lai et al. 2023), for the replication table only.
PAPER_REPORTED = {
    "Lai Strategy": {"f1": 0.322, "auc": 0.546},
    "Random prediction": {"f1": 0.286, "auc": 0.500},
}
PAPER_ROW_COUNTS = {"train": 2741, "val": 427, "test": 827}


def strategy_distribution(dialogue, players):
    """Per-player distribution over the 7 strategies; all-zero if silent."""
    counts = {p: Counter() for p in players}
    for utt in dialogue:
        speaker = utt.get("speaker")
        if speaker not in counts:
            continue
        for label in (utt.get("annotation") or []):
            if label in LAI_STRATEGIES:
                counts[speaker][label] += 1
    dists = {}
    for p, c in counts.items():
        total = sum(c[s] for s in LAI_STRATEGIES)
        dists[p] = ([c[s] / total for s in LAI_STRATEGIES] if total
                    else [0.0] * len(LAI_STRATEGIES))
    return dists


def _is_index(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def build_lai_rows(game, dataset, outcome_index):
    """All n^2 ordered (voter, candidate) rows for one game, moderators removed."""
    outcome = get_outcome_record(game, dataset, outcome_index)
    names = list(outcome["playerNames"])
    start_roles = list(outcome.get("startRoles", []))
    votes = list(outcome["votingOutcome"])
    dists = strategy_distribution(game["Dialogue"], names)

    keep = [i for i in range(len(names))
            if i >= len(start_roles) or start_roles[i] != MODERATOR_ROLE]

    rows = []
    for i in keep:
        voter = names[i]
        v = votes[i] if i < len(votes) else None
        for j in keep:
            candidate = names[j]
            rows.append({
                "dataset": dataset,
                "session_key": get_session_key(game, dataset),
                "game_id": get_game_id(game),
                "voter": voter, "candidate": candidate,
                "label": int(_is_index(v) and v == j),
                "feature": dists[voter] + dists[candidate],
            })
    return rows


def build_lai_split_rows(annot_splits, outcome_index):
    """``{split: [row, ...]}`` over both the YouTube and Ego4D subsets."""
    split_rows = {"train": [], "val": [], "test": []}
    for dataset, by_split in annot_splits.items():
        for split, games in by_split.items():
            for game in games:
                split_rows[split].extend(
                    build_lai_rows(game, dataset, outcome_index))
    return split_rows


def rows_to_xy(rows):
    X = np.asarray([r["feature"] for r in rows], dtype=float)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    return X, y


def split_row_counts(split_rows):
    """Row / positive / prevalence counts per split, against the paper's."""
    return pd.DataFrame([
        {"split": s, "rows": len(rows),
         "positives": int(sum(r["label"] for r in rows)),
         "prevalence": float(np.mean([r["label"] for r in rows])),
         "paper_rows": PAPER_ROW_COUNTS[s],
         "matches_paper": len(rows) == PAPER_ROW_COUNTS[s]}
        for s, rows in split_rows.items()])


def feature_names():
    return ([f"voter_{s}" for s in LAI_STRATEGIES]
            + [f"candidate_{s}" for s in LAI_STRATEGIES])


def run_lai_model(split_rows, C, class_weight, random_state=42):
    """Fixed-hyperparameter logistic regression, evaluated on every split.

    ``auc_hard`` reproduces the released code's call; ``auc_prob`` is the
    standard probability-based ROC-AUC.
    """
    X = {s: rows_to_xy(rows)[0] for s, rows in split_rows.items()}
    y = {s: rows_to_xy(rows)[1] for s, rows in split_rows.items()}

    model = LogisticRegression(C=C, class_weight=class_weight, max_iter=2000,
                               random_state=random_state)
    model.fit(X["train"], y["train"])
    assert np.all(model.n_iter_ < 2000), "logistic regression did not converge"

    out = {"model": model, "coef": model.coef_[0],
           "feature_names": feature_names()}
    for split in ("train", "val", "test"):
        hard = model.predict(X[split])
        prob = model.predict_proba(X[split])[:, 1]
        assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
        out[split] = {
            "n_rows": len(y[split]), "prevalence": float(y[split].mean()),
            "f1": f1_score(y[split], hard, zero_division=0),
            "auc_prob": roc_auc_score(y[split], prob),
            "auc_hard": roc_auc_score(y[split], hard),
        }
    return out


def replication_table(results, split_rows):
    """One compact table: paper vs ours, plus the split sizes."""
    n = {s: len(rows) for s, rows in split_rows.items()}
    rows = []
    for name, res in results.items():
        paper = PAPER_REPORTED[name]
        rows.append({
            "model": name,
            "paper_f1": paper["f1"], "our_f1": round(res["test"]["f1"], 4),
            "paper_auc": paper["auc"],
            "our_auc_prob": round(res["test"]["auc_prob"], 4),
            "our_auc_hard_as_released": round(res["test"]["auc_hard"], 4),
            "n_features": len(res["feature_names"]),
            "n_train": n["train"], "n_val": n["val"], "n_test": n["test"],
            "test_prevalence": round(res["test"]["prevalence"], 4),
        })
    paper = PAPER_REPORTED["Random prediction"]
    rows.append({"model": "Random prediction (paper)", "paper_f1": paper["f1"],
                 "our_f1": np.nan, "paper_auc": paper["auc"],
                 "our_auc_prob": np.nan, "our_auc_hard_as_released": np.nan,
                 "n_features": np.nan, "n_train": n["train"], "n_val": n["val"],
                 "n_test": n["test"], "test_prevalence": np.nan})
    return pd.DataFrame(rows)
