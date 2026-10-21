import argparse, json, sys
from pathlib import Path
import joblib, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if not arguments.final:
        raise SystemExit("Evaluation is test-set protected; use --final.")
    output = Path("reports/final_test_metrics.json")
    if output.exists() and not arguments.force:
        raise SystemExit("Final test evaluation already exists; use --force with a logged reason.")

    features = pd.read_csv("data/features/features.csv")
    labels = pd.read_csv("data/labels/labels.csv")
    data = features.merge(labels, on=["trace_id", "timestamp_s"])
    test = data[data.split == "test"]
    holdout_path = Path("data/processed/final_holdout.json")
    if holdout_path.exists():
        holdout_ids = json.loads(holdout_path.read_text())["trace_ids"]
        test = data[data.trace_id.isin(holdout_ids)].copy()

    results = {
        "holdout_samples": len(test),
        "holdout_traces": int(test["trace_id"].nunique()),
        "targets": {},
    }

    for target in ["network_condition", "buffer_risk", "recommended_quality"]:
        artifact = joblib.load(f"models/{target}_best.joblib")
        model = artifact["model"]
        feat_cols = artifact["features"]
        model_classes = list(model.classes_)

        y_true = test[target].copy()
        if target == "network_condition" and len(model_classes) == 3 and "Severely Degrading" not in model_classes:
            y_true = y_true.replace({"Severely Degrading": "Degrading"})

        y_pred = model.predict(test[feat_cols])
        classes = sorted(y_true.unique())

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

        results["targets"][target] = {
            "model_type": type(model.named_steps["model"] if hasattr(model, "named_steps") else model).__name__,
            "classes": classes,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "precision": macro_prec,
            "recall": macro_rec,
            "per_class": {
                c: {
                    "precision": float(report[c]["precision"]),
                    "recall": float(report[c]["recall"]),
                    "f1-score": float(report[c]["f1-score"]),
                    "support": int(report[c]["support"]),
                }
                for c in classes
            },
            "confusion_matrix": cm,
        }

    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

