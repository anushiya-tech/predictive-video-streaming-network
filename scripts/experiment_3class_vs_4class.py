"""Rigorous 3-class vs 4-class network-condition experiment.
Evaluates on validation partition only (clean train vs validation).
Does NOT touch models/ or modify protected holdouts.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_experiment():
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

    # Create 3-class target without mutating original labels
    train_df["network_condition_3class"] = train_df["network_condition"].replace({"Severely Degrading": "Degrading"})
    val_df["network_condition_3class"] = val_df["network_condition"].replace({"Severely Degrading": "Degrading"})

    # Class distributions
    dist_4class_train = train_df["network_condition"].value_counts().to_dict()
    dist_4class_val = val_df["network_condition"].value_counts().to_dict()
    dist_3class_train = train_df["network_condition_3class"].value_counts().to_dict()
    dist_3class_val = val_df["network_condition_3class"].value_counts().to_dict()

    print("=== 4-Class Train Distribution ===", dist_4class_train)
    print("=== 4-Class Val Distribution ===", dist_4class_val)
    print("=== 3-Class Train Distribution ===", dist_3class_train)
    print("=== 3-Class Val Distribution ===", dist_3class_val)

    candidate_defs = {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic_balanced": lambda: Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "logistic_unweighted": lambda: Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "random_forest_balanced": lambda: Pipeline([
            ("impute", SimpleImputer()),
            ("model", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "random_forest_unweighted": lambda: Pipeline([
            ("impute", SimpleImputer()),
            ("model", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
        ]),
        "hist_gradient": lambda: Pipeline([
            ("impute", SimpleImputer()),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]),
    }

    results = {
        "class_distributions": {
            "4class": {"train": dist_4class_train, "validation": dist_4class_val},
            "3class": {"train": dist_3class_train, "validation": dist_3class_val},
        },
        "experiments": {},
    }

    for task_name, target_col in [("4class", "network_condition"), ("3class", "network_condition_3class")]:
        results["experiments"][task_name] = {}
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        classes = sorted(y_train.unique())

        print(f"\n=================== Running {task_name.upper()} Experiments ===================")
        for model_name, make_model in candidate_defs.items():
            model = make_model()
            model.fit(train_df[feature_columns], y_train)
            preds = model.predict(val_df[feature_columns])

            acc = float(accuracy_score(y_val, preds))
            macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
            macro_prec = float(precision_score(y_val, preds, average="macro", zero_division=0))
            macro_rec = float(recall_score(y_val, preds, average="macro", zero_division=0))

            report = classification_report(y_val, preds, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_val, preds, labels=classes).tolist()

            results["experiments"][task_name][model_name] = {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "precision": macro_prec,
                "recall": macro_rec,
                "per_class": {
                    cls: {
                        "precision": float(report[cls]["precision"]),
                        "recall": float(report[cls]["recall"]),
                        "f1-score": float(report[cls]["f1-score"]),
                        "support": int(report[cls]["support"]),
                    }
                    for cls in classes
                },
                "classes": classes,
                "confusion_matrix": cm,
            }
            print(f"[{task_name}] {model_name:25s} -> Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Prec: {macro_prec:.4f}, Rec: {macro_rec:.4f}")

    out_path = Path("reports/network_3class_vs_4class_metrics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {out_path}")
    return results


if __name__ == "__main__":
    run_experiment()
