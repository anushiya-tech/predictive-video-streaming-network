"""Validated inference from registered joblib models."""
from __future__ import annotations
from pathlib import Path
import joblib,pandas as pd
def predict(target: str, measurements: dict[str,float]) -> dict[str,object]:
 path=Path(f"models/{target}_best.joblib")
 if not path.is_file(): raise FileNotFoundError(f"Model is missing: {path}. Run scripts/train_models.py first.")
 artifact=joblib.load(path); missing=[name for name in artifact["features"] if name not in measurements]
 if missing: raise ValueError(f"Invalid input: missing feature values {missing}")
 data=pd.DataFrame([{name:measurements[name] for name in artifact["features"]}]); probabilities=artifact["model"].predict_proba(data)[0]; index=int(probabilities.argmax())
 return {"prediction":artifact["model"].classes_[index],"confidence":float(probabilities[index]),"probabilities":dict(zip(artifact["model"].classes_,map(float,probabilities)))}

