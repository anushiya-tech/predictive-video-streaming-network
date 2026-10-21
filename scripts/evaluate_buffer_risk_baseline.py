"""Evaluate current-buffer-only baseline vs existing buffer-risk model on validation data.
Does NOT touch protected holdout or overwrite production models.
"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_baselines():
    features = pd.read_csv("data/features/features.csv")
    labels = pd.read_csv("data/labels/labels.csv")
    data = features.merge(labels, on=["trace_id", "timestamp_s"])

    val = data[data.split == "validation"].copy()
    y_val = val["buffer_risk"]
    classes = sorted(y_val.unique())

    dist_val = val["buffer_risk"].value_counts().to_dict()
    print("Validation distribution:", dist_val)

    # 1. Majority baseline
    maj_class = val["buffer_risk"].mode()[0]
    maj_pred = [maj_class] * len(val)

    # 2. Current-buffer-only heuristic baseline:
    # Uses the same static buffer thresholds (High < 3s, Medium < 8s, Low >= 8s)
    def buffer_heuristic(buf: float) -> str:
        if buf < 3.0:
            return "High"
        elif buf < 8.0:
            return "Medium"
        else:
            return "Low"

    buf_pred = val["buffer_s"].map(buffer_heuristic)

    # 3. Existing production buffer-risk model
    art = joblib.load("models/buffer_risk_best.joblib")
    model = art["model"]
    feat_cols = art["features"]
    model_pred = model.predict(val[feat_cols])

    results = {
        "validation_samples": len(val),
        "validation_distribution": dist_val,
        "classes": classes,
        "baselines": {},
    }

    eval_targets = [
        ("majority", maj_pred, "Majority Baseline"),
        ("current_buffer_heuristic", buf_pred, "Current-Buffer-Only Heuristic"),
        ("production_random_forest", model_pred, "Production Model (Random Forest)"),
    ]

    for key, pred, name in eval_targets:
        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
        prec = float(precision_score(y_val, pred, average="macro", zero_division=0))
        rec = float(recall_score(y_val, pred, average="macro", zero_division=0))
        rep = classification_report(y_val, pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_val, pred, labels=classes).tolist()

        results["baselines"][key] = {
            "name": name,
            "accuracy": acc,
            "macro_f1": f1,
            "precision": prec,
            "recall": rec,
            "per_class": {
                c: {
                    "precision": float(rep[c]["precision"]),
                    "recall": float(rep[c]["recall"]),
                    "f1-score": float(rep[c]["f1-score"]),
                    "support": int(rep[c]["support"]),
                }
                for c in classes
            },
            "confusion_matrix": cm,
        }
        print(f"=== {name} ===")
        print(f"Accuracy: {acc*100:.2f}%, Macro-F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        for c in classes:
            print(f"  Class {c:6s}: Prec: {rep[c]['precision']:.4f}, Rec: {rep[c]['recall']:.4f}, F1: {rep[c]['f1-score']:.4f}, Support: {rep[c]['support']}")
        print("Confusion Matrix:")
        print(cm)
        print()

    out_path = Path("reports/buffer_risk_baseline_metrics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_path}")
    return results


if __name__ == "__main__":
    evaluate_baselines()
