"""Load preprocessed [timestamp, Mbps] research traces."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.data.schema import TRACE_COLUMNS, validate_trace_frame
def load_mahimahi_trace(path: str | Path, source: str = "mahimahi") -> pd.DataFrame:
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f"Trace file not found: {path}")
    frame = pd.read_csv(path, sep=r"\s+|,", engine="python", header=None, comment="#")
    if frame.shape[1] == 1: frame[1] = frame.index.astype(float)
    if frame.shape[1] < 2: raise ValueError(f"{path}: expected timestamp and throughput columns")
    output = pd.DataFrame({"trace_id": path.stem, "source": source, "timestamp_s": pd.to_numeric(frame.iloc[:, 0], errors="coerce"), "throughput_mbps": pd.to_numeric(frame.iloc[:, 1], errors="coerce"), "latency_ms": float("nan"), "loss_pct": float("nan"), "jitter_ms": float("nan"), "is_synthetic": False})
    return validate_trace_frame(output.dropna(subset=["timestamp_s", "throughput_mbps"]))

