"""Common trace schema validation."""
from __future__ import annotations
import pandas as pd
TRACE_COLUMNS = ["trace_id", "source", "timestamp_s", "throughput_mbps", "latency_ms", "loss_pct", "jitter_ms", "is_synthetic"]
def validate_trace_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in TRACE_COLUMNS if column not in frame]
    if missing: raise ValueError(f"Trace data missing required columns: {missing}")
    output = frame[TRACE_COLUMNS].copy()
    if output["trace_id"].isna().any() or output["timestamp_s"].isna().any(): raise ValueError("trace_id and timestamp_s cannot be missing")
    if (output["throughput_mbps"] < 0).any(): raise ValueError("throughput_mbps cannot be negative")
    return output

