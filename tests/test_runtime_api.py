from fastapi.testclient import TestClient
from app.backend.main import app

client = TestClient(app)
SCENARIOS = ["stable", "sudden_drop", "recovery", "degradation", "congestion", "mobile"]
VALID_QUALITIES = {"360p", "480p", "720p", "1080p"}
VALID_CONDITIONS = {"Stable", "Improving", "Degrading"}
VALID_RISKS = {"Low", "Medium", "High"}
VALID_REASONS = {"hold", "fallback", "predicted_high_risk", "evidence_based_up_switch"}


def test_simulation_scenarios_endpoint():
    res = client.get("/api/simulation/scenarios")
    assert res.status_code == 200
    assert res.json() == SCENARIOS


def test_baselines_endpoint():
    res = client.get("/api/baselines")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 10
    for row in data:
        assert "method" in row
        assert "environment" in row
        assert "rebuffer_seconds" in row
        assert "avg_bitrate_mbps" in row
        assert "switches" in row
        assert "qoe" in row


def test_video_status_endpoint():
    res = client.get("/api/video/status")
    assert res.status_code == 200
    data = res.json()
    assert data["ffmpeg_available"] is False
    assert "qualities" in data


def test_all_six_simulation_scenarios():
    for sc in SCENARIOS:
        res = client.post("/api/simulation/scenario", json={"scenario": sc})
        assert res.status_code == 200
        assert res.json()["scenario"] == sc
        assert res.json()["ticks"] == 60

        hist = client.get("/api/network/history").json()
        assert len(hist) == 60

        curr = client.get("/api/network/current").json()
        assert curr["tick"] == 59
        assert curr["quality"] in VALID_QUALITIES
        assert curr["decision"]["reason"] in VALID_REASONS

        pred = client.get("/api/prediction/current").json()
        assert pred["network_condition"] in VALID_CONDITIONS
        assert pred["buffer_risk"] in VALID_RISKS
        assert 0.0 <= pred["network_confidence"] <= 1.0
        assert 0.0 <= pred["buffer_confidence"] <= 1.0
        assert pred["horizon_seconds"] == 5
