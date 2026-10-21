import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.backend.main import app


def main():
    client = TestClient(app)

    # 1. Test GET /
    res_root = client.get("/")
    assert res_root.status_code == 200, f"GET / failed with {res_root.status_code}"
    assert "<!DOCTYPE html>" in res_root.text or "<html" in res_root.text
    print("[PASS] GET / served HTML frontend")

    # 2. Test GET /api/simulation/scenarios
    res_sc = client.get("/api/simulation/scenarios")
    assert res_sc.status_code == 200
    scenarios = res_sc.json()
    expected_scenarios = ["stable", "sudden_drop", "recovery", "degradation", "congestion", "mobile"]
    assert scenarios == expected_scenarios, f"Scenarios mismatch: {scenarios}"
    print("[PASS] GET /api/simulation/scenarios returns expected 6 scenarios")

    # 3. Test GET /api/baselines
    res_base = client.get("/api/baselines")
    assert res_base.status_code == 200
    baselines = res_base.json()
    assert len(baselines) == 10
    for b in baselines:
        assert "method" in b
        assert "environment" in b
        assert "rebuffer_seconds" in b
        assert "avg_bitrate_mbps" in b
        assert "switches" in b
        assert "qoe" in b
    print("[PASS] GET /api/baselines returned valid 10 baseline records")

    # 4. Test GET /api/video/status
    res_vid = client.get("/api/video/status")
    assert res_vid.status_code == 200
    vdata = res_vid.json()
    assert vdata["ffmpeg_available"] is False
    assert "qualities" in vdata
    print("[PASS] GET /api/video/status reports correct video status")

    # 5. Test each scenario with POST /api/simulation/scenario and check history/current/prediction
    valid_qualities = {"360p", "480p", "720p", "1080p"}
    valid_conditions = {"Stable", "Improving", "Degrading"}
    valid_risks = {"Low", "Medium", "High"}

    for sc in scenarios:
        post_res = client.post("/api/simulation/scenario", json={"scenario": sc})
        assert post_res.status_code == 200, f"POST scenario {sc} failed: {post_res.text}"
        data = post_res.json()
        assert data["scenario"] == sc
        assert data["ticks"] == 60

        # GET /api/network/history
        hist_res = client.get("/api/network/history")
        assert hist_res.status_code == 200
        hist = hist_res.json()
        assert len(hist) == 60, f"Expected 60 ticks for {sc}, got {len(hist)}"

        for t, item in enumerate(hist):
            assert item["tick"] == t
            assert "bandwidth_mbps" in item and item["bandwidth_mbps"] >= 0
            assert "buffer_s" in item and 0 <= item["buffer_s"] <= 35
            assert "stall_s" in item and item["stall_s"] >= 0
            assert item["quality"] in valid_qualities
            assert item["bitrate_mbps"] > 0
            pred = item["prediction"]
            assert pred["type"] == "PREDICTED"
            assert pred["network_condition"] in valid_conditions, f"Invalid condition: {pred['network_condition']}"
            assert pred["buffer_risk"] in valid_risks, f"Invalid risk: {pred['buffer_risk']}"
            assert 0.0 <= pred["network_confidence"] <= 1.0
            assert 0.0 <= pred["buffer_confidence"] <= 1.0
            assert pred["horizon_seconds"] == 5
            dec = item["decision"]
            assert dec["quality"] in valid_qualities
            assert dec["reason"] in {"hold", "fallback", "predicted_high_risk", "evidence_based_up_switch"}

        # GET /api/network/current
        curr_res = client.get("/api/network/current")
        assert curr_res.status_code == 200
        curr = curr_res.json()
        assert curr["tick"] == 59
        assert curr == hist[-1]

        # GET /api/prediction/current
        pred_res = client.get("/api/prediction/current")
        assert pred_res.status_code == 200
        pcurr = pred_res.json()
        assert pcurr == curr["prediction"]

        print(f"[PASS] Scenario '{sc}': 60 ticks, quality={curr['quality']}, buffer={curr['buffer_s']}s, risk={pcurr['buffer_risk']}, cond={pcurr['network_condition']}")

    print("ALL BACKEND API AND SCENARIO VERIFICATIONS PASSED!")

if __name__ == "__main__":
    main()
