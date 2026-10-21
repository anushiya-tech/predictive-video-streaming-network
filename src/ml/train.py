"""Train trace-split classical models and select by validation macro-F1."""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = {"network_condition": "network_condition", "buffer_risk": "buffer_risk", "recommended_quality": "recommended_quality"}


def train_network_condition() -> dict:
    """Train validated 3-class HistGradientBoosting model for network_condition."""
    features = pd.read_csv("data/features/features.csv")
    labels = pd.read_csv("data/labels/labels.csv")
    joined = features.merge(labels, on=["trace_id", "timestamp_s"], how="inner")
    feature_columns = [col for col in features if col not in {"trace_id", "timestamp_s", "split", "is_synthetic"}]
    holdout_path = Path("data/processed/final_holdout.json")
    holdout = set(json.loads(holdout_path.read_text())["trace_ids"]) if holdout_path.exists() else set()
    train = joined[(joined.split == "train") & ~joined.trace_id.isin(holdout)].copy()
    validation = joined[joined.split == "validation"].copy()

    # 3-class mapping
    train["network_condition"] = train["network_condition"].replace({"Severely Degrading": "Degrading"})
    validation["network_condition"] = validation["network_condition"].replace({"Severely Degrading": "Degrading"})

    model = Pipeline([("impute", SimpleImputer()), ("model", HistGradientBoostingClassifier(random_state=42))])
    started = time.perf_counter()
    model.fit(train[feature_columns], train["network_condition"])
    predicted = model.predict(validation[feature_columns])
    probabilities = model.predict_proba(validation[feature_columns])
    metrics = {
        "Model": "network_condition:hist_gradient",
        "Accuracy": float(accuracy_score(validation["network_condition"], predicted)),
        "Macro F1": float(f1_score(validation["network_condition"], predicted, average="macro", zero_division=0)),
        "Precision": float(precision_score(validation["network_condition"], predicted, average="macro", zero_division=0)),
        "Recall": float(recall_score(validation["network_condition"], predicted, average="macro", zero_division=0)),
        "Log Loss": float(log_loss(validation["network_condition"], probabilities, labels=model.classes_)),
        "Inference ms": float((time.perf_counter() - started) * 1000 / max(1, len(validation))),
        "classes": list(model.classes_),
    }
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": feature_columns, "target": "network_condition", "metrics": metrics, "seed": 42}, "models/network_condition_best.joblib")
    return metrics


def train_all() -> pd.DataFrame:
    features = pd.read_csv("data/features/features.csv")
    labels = pd.read_csv("data/labels/labels.csv")
    joined = features.merge(labels, on=["trace_id", "timestamp_s"], how="inner")
    feature_columns = [col for col in features if col not in {"trace_id", "timestamp_s", "split", "is_synthetic"}]
    holdout_path = Path("data/processed/final_holdout.json")
    holdout = set(json.loads(holdout_path.read_text())["trace_ids"]) if holdout_path.exists() else set()
    rows = []
    Path("models").mkdir(exist_ok=True)
    for target in TARGETS:
        train = joined[(joined.split == "train") & ~joined.trace_id.isin(holdout)].copy()
        validation = joined[joined.split == "validation"].copy()
        if target == "network_condition":
            train[target] = train[target].replace({"Severely Degrading": "Degrading"})
            validation[target] = validation[target].replace({"Severely Degrading": "Degrading"})

        candidates = {
            "majority": DummyClassifier(strategy="most_frequent"),
            "logistic": Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]),
            "random_forest": Pipeline([("impute", SimpleImputer()), ("model", RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1))]),
            "hist_gradient": Pipeline([("impute", SimpleImputer()), ("model", HistGradientBoostingClassifier(random_state=42))]),
        }
        best = None
        for name, model in candidates.items():
            started = time.perf_counter()
            model.fit(train[feature_columns], train[target])
            predicted = model.predict(validation[feature_columns])
            probabilities = model.predict_proba(validation[feature_columns])
            metrics = {
                "Model": f"{target}:{name}",
                "Accuracy": accuracy_score(validation[target], predicted),
                "Macro F1": f1_score(validation[target], predicted, average="macro", zero_division=0),
                "Precision": precision_score(validation[target], predicted, average="macro", zero_division=0),
                "Recall": recall_score(validation[target], predicted, average="macro", zero_division=0),
                "AUC": "",
                "Log Loss": log_loss(validation[target], probabilities, labels=model.classes_),
                "Inference ms": (time.perf_counter() - started) * 1000 / max(1, len(validation)),
            }
            rows.append(metrics)
            if best is None or metrics["Macro F1"] > best[0]:
                best = (metrics["Macro F1"], model, metrics)
        joblib.dump({"model": best[1], "features": feature_columns, "target": target, "metrics": best[2], "seed": 42}, f"models/{target}_best.joblib")
    pd.DataFrame(rows).to_csv("reports/model_comparison.csv", index=False)
    return pd.DataFrame(rows)

