"""Explicitly labelled autocorrelated stress traces; never used in final testing."""
from __future__ import annotations
import numpy as np, pandas as pd
from src.data.schema import validate_trace_frame
def generate_stress_trace(kind: str, seconds: int = 300, seed: int = 42) -> pd.DataFrame:
    if seconds < 2: raise ValueError("seconds must be at least 2")
    rng = np.random.default_rng(seed); values = np.empty(seconds); values[0] = 12.0
    targets = {"stable": 12, "degradation": 3, "sudden_drop": 2, "recovery": 18, "congestion": 5}.get(kind)
    if targets is None: raise ValueError(f"Unsupported synthetic scenario: {kind}")
    for index in range(1, seconds):
        regime = targets if kind != "sudden_drop" or index > seconds // 3 else 12
        if kind == "degradation": regime = max(2, 15 - 12 * index / seconds)
        values[index] = max(.1, .82 * values[index - 1] + .18 * regime + rng.normal(0, .6))
    return validate_trace_frame(pd.DataFrame({"trace_id": f"synthetic_{kind}_{seed}", "source": "synthetic", "timestamp_s": np.arange(seconds), "throughput_mbps": values, "latency_ms": np.nan, "loss_pct": np.nan, "jitter_ms": np.nan, "is_synthetic": True}))

