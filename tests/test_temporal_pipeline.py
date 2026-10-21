import pandas as pd

from src.data.labels import create_labels
from src.data.schema import TRACE_COLUMNS, validate_trace_frame
from src.features.build import build_features
from src.streaming.simulator import add_player_state


def _trace(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"trace_id": "trace", "source": "fixture", "timestamp_s": range(len(values)), "throughput_mbps": values, "latency_ms": [float("nan")] * len(values), "loss_pct": [float("nan")] * len(values), "jitter_ms": [float("nan")] * len(values), "is_synthetic": [False] * len(values)})


def test_future_changes_do_not_change_features() -> None:
    baseline = add_player_state(_trace([5, 5, 5, 5, 5, 5, 5]))
    changed = baseline.copy()
    changed.loc[5:, "throughput_mbps"] = 100
    left = build_features(baseline).iloc[4]
    right = build_features(changed).iloc[4]
    assert left.filter(like="throughput").equals(right.filter(like="throughput"))


def test_future_changes_do_change_labels() -> None:
    steady = add_player_state(_trace([5] * 12))
    degraded = add_player_state(_trace([5] * 5 + [.2] * 7))
    assert create_labels(steady).iloc[1].network_condition != create_labels(degraded).iloc[1].network_condition


def test_schema_rejects_negative_throughput() -> None:
    invalid = _trace([1, -1])
    try:
        validate_trace_frame(invalid)
    except ValueError as error:
        assert "negative" in str(error)
    else:
        raise AssertionError("Expected validation failure")
