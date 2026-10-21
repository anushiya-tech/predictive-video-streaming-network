import hashlib
import json
from pathlib import Path
import pandas as pd

HOLDOUT_FILE = Path("data/processed/final_holdout.json")
EXPECTED_SHA = "89a83dcc3e15f3c109240a0f013a8bfcd8480f238cafb6e45d22974b2a581823"


def test_holdout_file_exists():
    assert HOLDOUT_FILE.exists(), f"Holdout file missing: {HOLDOUT_FILE}"


def test_holdout_trace_count():
    data = json.loads(HOLDOUT_FILE.read_text())
    traces = data.get("trace_ids", [])
    assert len(traces) == 59, f"Expected 59 traces, found {len(traces)}"


def test_holdout_sha256_integrity():
    data = json.loads(HOLDOUT_FILE.read_text())
    traces = data["trace_ids"]
    calc_sha = hashlib.sha256("\n".join(traces).encode()).hexdigest()
    assert calc_sha == EXPECTED_SHA, f"Holdout hash modified: {calc_sha} != {EXPECTED_SHA}"
    assert data.get("sha256") == EXPECTED_SHA, "Stored hash metadata mismatch"


def test_zero_leakage_into_validation():
    data = json.loads(HOLDOUT_FILE.read_text())
    holdout_set = set(data["trace_ids"])

    traces_df = pd.read_csv("data/processed/traces.csv")
    val_traces = set(traces_df[traces_df.split == "validation"]["trace_id"])

    overlap = holdout_set.intersection(val_traces)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping traces: {overlap}"
