"""Load Norway/Belgium six-column mobile measurement logs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.data.schema import validate_trace_frame
def load_mobile_log(path: str | Path, source: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f"Trace file not found: {path}")
    data = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    if data.shape[1] < 6: raise ValueError(f"{path}: expected six columns in mobile log")
    elapsed_ms = pd.to_numeric(data.iloc[:, 5], errors="coerce")
    received_bytes = pd.to_numeric(data.iloc[:, 4], errors="coerce")
    output = pd.DataFrame({"trace_id": path.stem, "source": source, "timestamp_s": pd.to_numeric(data.iloc[:, 0], errors="coerce"), "throughput_mbps": received_bytes * 8 / elapsed_ms / 1000, "latency_ms": float("nan"), "loss_pct": float("nan"), "jitter_ms": float("nan"), "is_synthetic": False})
    return validate_trace_frame(output.replace([float("inf"), -float("inf")], float("nan")).dropna(subset=["timestamp_s", "throughput_mbps"]))

