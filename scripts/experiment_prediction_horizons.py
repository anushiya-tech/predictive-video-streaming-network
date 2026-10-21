"""Rigorous prediction horizon experiment for 3-class network condition.
Evaluates horizons: 1s, 3s, 5s, 10s on train (excluding protected holdout) vs validation.
Preserves existing labels and models untouched.
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


def generate_horizon_labels(player_state: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Generate 3-class network condition labels for a specific future horizon."""
    records = []
    for trace_id, group in player_state.sort_values(["trace_id", "timestamp_s"]).groupby("trace_id", sort=False):
        tp = group["throughput_mbps"].values
        n = len(tp)
        if n < horizon + 1:
            continue
        past_cum = np.cumsum(np.insert(tp, 0, 0))
        # Past reference: past 5-second mean (or up to 5s at trace start)
        past_means = np.array([tp[max(0, i - 4) : i + 1].mean() for i in range(n - horizon)])
        # Future window: horizon steps starting at i+1
        future_means = (past_cum[1 + horizon : n + 1] - past_cum[1 : n - horizon + 1]) / horizon
        ratios = (future_means / np.maximum(0.01, past_means)) - 1.0

        # 3-class formulation: Degrading + Severely Degrading merged into Degrading
        trends = np.where(
            ratios >= 0.15,
            "Improving",
            np.where(ratios <= -0.15, "Degrading", "Stable"),
        )
        sub = pd.DataFrame({
            "trace_id": trace_id,
            "timestamp_s": group["timestamp_s"].values[: n - horizon],
            "target": trends,
        })
        records.append(sub)
    return pd.concat(records, ignore_index=True)


def run_experiment():
    print("Loading data...")
    player_state = pd.read_csv("data/processed/player_state.csv")
    features = pd.read_csv("data/features/features.csv")
    feature_columns = [
        col for col in features
        if col not in {"trace_id", "timestamp_s", "split", "is_synthetic"}
    ]

    holdout_path = Path("data/processed/final_holdout.json")
    holdout = set(json.loads(holdout_path.read_text())["trace_ids"]) if holdout_path.exists() else set()

    horizons = [1, 3, 5, 10]
    candidate_defs = {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic_balanced": lambda: Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest_balanced": lambda: Pipeline([
            ("impute", SimpleImputer()),
            ("model", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "hist_gradient": lambda: Pipeline([
            ("impute", SimpleImputer()),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]),
    }

    results = {
        "horizons": {},
    }

    for h in horizons:
        print(f"\n=================== Evaluating Horizon {h}s ===================")
        labels_h = generate_horizon_labels(player_state, horizon=h)
        joined = features.merge(labels_h, on=["trace_id", "timestamp_s"], how="inner")

        train_df = joined[(joined.split == "train") & (~joined.trace_id.isin(holdout))].copy()
        val_df = joined[joined.split == "validation"].copy()

        dist_train = train_df["target"].value_counts().to_dict()
        dist_val = val_df["target"].value_counts().to_dict()
        print(f"[{h}s] Train dist:", dist_train)
        print(f"[{h}s] Val dist:  ", dist_val)

        classes = sorted(train_df["target"].unique())
        y_train = train_df["target"]
        y_val = val_df["target"]

        results["horizons"][f"{h}s"] = {
            "horizon_seconds": h,
            "train_distribution": dist_train,
            "val_distribution": dist_val,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "classes": classes,
            "models": {},
        }

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

            results["horizons"][f"{h}s"]["models"][model_name] = {
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
                "confusion_matrix": cm,
            }
            print(f"[{h}s] {model_name:25s} -> Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}, Prec: {macro_prec:.4f}, Rec: {macro_rec:.4f}")

    out_path = Path("reports/network_prediction_horizon_metrics.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics to {out_path}")
    return results


if __name__ == "__main__":
    run_experiment()
