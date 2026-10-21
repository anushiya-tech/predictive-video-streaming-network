"""Rigorous model calibration and comparison across candidate models.
Evaluates on validation data only (excluding protected 59-trace holdout).
Does not overwrite production models.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, StandardScaler


def compute_calibration_metrics(y_true_indices: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Compute multi-class Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true_indices

    # Multi-class Brier score: mean squared error between probabilities and one-hot ground truth
    n_classes = y_prob.shape[1]
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true_indices)), y_true_indices] = 1.0
    brier_score = float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            acc_bin = float(np.mean(accuracies[in_bin]))
            conf_bin = float(np.mean(confidences[in_bin]))
            diff = float(np.abs(acc_bin - conf_bin))
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            bin_stats.append({
                "bin": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "count": int(np.sum(in_bin)),
                "prop": prop_in_bin,
                "accuracy": acc_bin,
                "confidence": conf_bin,
                "gap": diff,
            })

    return {
        "brier_score": brier_score,
        "ece": float(ece),
        "mce": float(mce),
        "bins": bin_stats,
    }


def run_calibration_evaluation():
    print("Loading data for calibration evaluation...")
    features = pd.read_csv("data/features/features.csv")
    labels = pd.read_csv("data/labels/labels.csv")
    joined = features.merge(labels, on=["trace_id", "timestamp_s"], how="inner")
    feature_columns = [
        col for col in features
        if col not in {"trace_id", "timestamp_s", "split", "is_synthetic"}
    ]

    holdout_path = Path("data/processed/final_holdout.json")
    holdout = set(json.loads(holdout_path.read_text())["trace_ids"]) if holdout_path.exists() else set()

    train_df = joined[(joined.split == "train") & (~joined.trace_id.isin(holdout))].copy()
    val_df = joined[joined.split == "validation"].copy()

    # 3-class network target
    train_df["network_condition_3class"] = train_df["network_condition"].replace({"Severely Degrading": "Degrading"})
    val_df["network_condition_3class"] = val_df["network_condition"].replace({"Severely Degrading": "Degrading"})

    tasks = {
        "buffer_risk": {
            "target": "buffer_risk",
            "production_artifact": "models/buffer_risk_best.joblib",
        },
        "network_condition_3class": {
            "target": "network_condition_3class",
            "production_artifact": None,
        },
        "network_condition_4class": {
            "target": "network_condition",
            "production_artifact": "models/network_condition_best.joblib",
        },
    }

    results = {}

    for task_name, task_info in tasks.items():
        print(f"\n=================== Evaluating Calibration for {task_name} ===================")
        target_col = task_info["target"]
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        classes = sorted(y_train.unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_val_idx = np.array([class_to_idx[c] for c in y_val])

        candidates = {
            "majority": Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
            "logistic_balanced": Pipeline([
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]),
            "random_forest_balanced": Pipeline([
                ("impute", SimpleImputer()),
                ("model", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)),
            ]),
            "hist_gradient": Pipeline([
                ("impute", SimpleImputer()),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]),
        }

        # If a production artifact exists, load it directly as one of the candidates to evaluate exactly
        if task_info["production_artifact"] and Path(task_info["production_artifact"]).exists():
            art = joblib.load(task_info["production_artifact"])
            candidates["production_model"] = art["model"]

        results[task_name] = {
            "classes": classes,
            "val_samples": len(val_df),
            "models": {},
        }

        for model_name, pipeline in candidates.items():
            # If not pre-trained production model, fit on clean train
            if model_name != "production_model":
                pipeline.fit(train_df[feature_columns], y_train)

            # Predict and proba
            y_pred = pipeline.predict(val_df[feature_columns])
            y_prob = pipeline.predict_proba(val_df[feature_columns])

            # Ensure proba columns map correctly to sorted classes
            if hasattr(pipeline, "classes_"):
                p_classes = list(pipeline.classes_)
            elif hasattr(pipeline, "named_steps") and hasattr(pipeline.named_steps["model"], "classes_"):
                p_classes = list(pipeline.named_steps["model"].classes_)
            else:
                p_classes = classes

            if p_classes != classes:
                # Reorder proba columns to match classes
                reorder_idx = [p_classes.index(c) for c in classes]
                y_prob = y_prob[:, reorder_idx]

            acc = float(accuracy_score(y_val, y_pred))
            macro_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
            macro_prec = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
            macro_rec = float(recall_score(y_val, y_pred, average="macro", zero_division=0))
            ll = float(log_loss(y_val, y_prob, labels=classes))

            cal = compute_calibration_metrics(y_val_idx, y_prob)

            results[task_name]["models"][model_name] = {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "precision": macro_prec,
                "recall": macro_rec,
                "log_loss": ll,
                "brier_score": cal["brier_score"],
                "ece": cal["ece"],
                "mce": cal["mce"],
                "calibration_bins": cal["bins"],
            }

            print(f"[{task_name}] {model_name:25s} -> Acc: {acc*100:.2f}%, F1: {macro_f1:.4f}, LL: {ll:.4f}, Brier: {cal['brier_score']:.4f}, ECE: {cal['ece']:.4f}")

    out_path = Path("reports/model_calibration_metrics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved calibration results to {out_path}")
    return results


if __name__ == "__main__":
    run_calibration_evaluation()
