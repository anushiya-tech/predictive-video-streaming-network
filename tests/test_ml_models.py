from pathlib import Path
import joblib
from src.ml.predict import predict

def test_models_exist_and_load():
    for target in ["network_condition", "buffer_risk", "recommended_quality"]:
        path = Path(f"models/{target}_best.joblib")
        assert path.exists(), f"Missing model: {path}"
        artifact = joblib.load(path)
        assert "model" in artifact
        assert "features" in artifact
        assert len(artifact["features"]) > 0


def test_model_classes():
    nc_artifact = joblib.load("models/network_condition_best.joblib")
    nc_classes = list(nc_artifact["model"].classes_)
    assert sorted(nc_classes) == ["Degrading", "Improving", "Stable"]

    br_artifact = joblib.load("models/buffer_risk_best.joblib")
    br_classes = list(br_artifact["model"].classes_)
    assert sorted(br_classes) == ["High", "Low", "Medium"]


def test_predict_function_returns_valid_probabilities():
    nc_artifact = joblib.load("models/network_condition_best.joblib")
    features = {f: 1.0 for f in nc_artifact["features"]}
    res = predict("network_condition", features)

    assert "prediction" in res
    assert res["prediction"] in {"Degrading", "Improving", "Stable"}
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 1e-4
