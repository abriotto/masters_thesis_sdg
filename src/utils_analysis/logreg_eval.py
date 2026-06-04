import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def find_best_threshold(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def eval_prob_model(model, X, y, threshold=0.5):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": threshold,
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc_prob": safe_auc(y, y_prob),
        "accuracy": accuracy_score(y, y_pred),
        "positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }, y_prob, y_pred


def eval_group_ranking(rows, prob, group_cols=("dataset", "session_key", "game_id")):
    """
    Optional ranking evaluation.

    For LLM rows, this asks:
        in each game, is the highest-probability candidate the LLM's chosen_vote?

    For human rows, if group_cols includes voter:
        in each (game, voter), is the highest-probability candidate the human vote?

    This is not a replacement for binary F1/AUC. It is just a useful companion.
    """
    df = pd.DataFrame([
        {
            "dataset": r.get("dataset"),
            "session_key": r.get("session_key"),
            "game_id": r.get("game_id"),
            "voter": r.get("voter"),
            "candidate": r.get("candidate"),
            "chosen_vote": r.get("chosen_vote"),
            "label": r.get("label"),
        }
        for r in rows
    ]).copy()

    df["prob"] = prob

    group_cols = list(group_cols)
    pred_idx = df.groupby(group_cols)["prob"].idxmax()
    pred_df = df.loc[pred_idx].copy()

    # Works for both human and LLM because the true positive row has label == 1.
    top1_acc = float((pred_df["label"] == 1).mean())

    df["rank"] = df.groupby(group_cols)["prob"].rank(
        ascending=False,
        method="min",
    )

    true_rows = df[df["label"] == 1].copy()
    mrr = float((1.0 / true_rows["rank"]).mean()) if len(true_rows) else np.nan

    return {
        "top1_accuracy": top1_acc,
        "mrr": mrr,
        "num_groups": int(len(pred_df)),
    }, pred_df, df


def run_tuned_logreg_with_classweight_search(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_names,
    c_grid=(0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 5.0),
    class_weight_grid=(None, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}),
    solver="liblinear",
    optimize_for="f1",
):
    """
    Generic binary logistic-regression training.

    optimize_for:
        "f1"  -> choose C/class_weight/threshold by validation F1
        "auc" -> choose C/class_weight by validation AUC; threshold still tuned by F1
    """
    search_rows = []
    best = None

    for cw in class_weight_grid:
        for C in c_grid:
            model = LogisticRegression(
                class_weight=cw,
                C=C,
                max_iter=3000,
                random_state=42,
                solver=solver,
            )
            model.fit(X_train, y_train)

            val_prob = model.predict_proba(X_val)[:, 1]
            best_t, best_val_f1 = find_best_threshold(y_val, val_prob)
            val_auc = safe_auc(y_val, val_prob)

            record = {
                "C": C,
                "class_weight": str(cw),
                "val_best_threshold": best_t,
                "val_f1": best_val_f1,
                "val_auc_prob": val_auc,
            }
            search_rows.append(record)

            if optimize_for == "f1":
                score = (best_val_f1, val_auc)
            elif optimize_for == "auc":
                score = (val_auc, best_val_f1)
            else:
                raise ValueError("optimize_for must be 'f1' or 'auc'")

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "C": C,
                    "class_weight": cw,
                    "threshold": best_t,
                    "val_f1": best_val_f1,
                    "val_auc_prob": val_auc,
                    "model": model,
                }

    best_model = best["model"]
    threshold = best["threshold"]

    train_metrics, train_prob, train_pred = eval_prob_model(best_model, X_train, y_train, threshold)
    val_metrics, val_prob, val_pred = eval_prob_model(best_model, X_val, y_val, threshold)
    test_metrics, test_prob, test_pred = eval_prob_model(best_model, X_test, y_test, threshold)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": best_model.coef_[0],
    }).sort_values("coef", ascending=False)

    search_df = pd.DataFrame(search_rows).sort_values(
        ["val_f1", "val_auc_prob"],
        ascending=False,
    ).reset_index(drop=True)

    return {
        "best_C": best["C"],
        "best_class_weight": best["class_weight"],
        "best_threshold": threshold,
        "model": best_model,
        "search_df": search_df,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "coef_df": coef_df,
        "train_prob": train_prob,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "train_pred": train_pred,
        "val_pred": val_pred,
        "test_pred": test_pred,
    }


def metrics_table(result):
    return pd.DataFrame([
        {"split": "train", **result["train_metrics"]},
        {"split": "val", **result["val_metrics"]},
        {"split": "test", **result["test_metrics"]},
    ])
