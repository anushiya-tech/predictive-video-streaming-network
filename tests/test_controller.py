import random
from src.adaptation.controller import PredictiveController, LADDER

valid_qualities = {q for q, _ in LADDER}
valid_bitrates = {b for _, b in LADDER}


def test_1_controller_initialization():
    c = PredictiveController()
    dec = c.decide(bandwidth=10.0, buffer_s=5.0, risk="Low", confidence=0.9, trend="Stable")
    assert dec["quality"] == "360p"
    assert dec["bitrate_mbps"] == 0.8
    assert dec["reason"] == "hold"


def test_2_normal_hold():
    c = PredictiveController()
    # Buffer is sufficient but evidence not yet reached 3
    d1 = c.decide(10.0, 15.0, "Low", 0.8, "Stable")
    assert d1["quality"] == "360p"
    assert d1["reason"] == "hold"
    # Degrading trend resets evidence and holds
    d2 = c.decide(10.0, 15.0, "Low", 0.8, "Degrading")
    assert d2["quality"] == "360p"
    assert d2["reason"] == "hold"
    assert c.evidence == 0


def test_3_evidence_based_up_switch():
    c = PredictiveController()
    c.decide(10.0, 20.0, "Low", 0.9, "Improving")
    c.decide(10.0, 20.0, "Low", 0.9, "Improving")
    d3 = c.decide(10.0, 20.0, "Low", 0.9, "Improving")
    assert d3["quality"] == "480p"
    assert d3["bitrate_mbps"] == 1.4
    assert d3["reason"] == "evidence_based_up_switch"
    assert c.cooldown == 6


def test_4_cooldown():
    c = PredictiveController()
    c.decide(10.0, 20.0, "Low", 0.9, "Improving")
    c.decide(10.0, 20.0, "Low", 0.9, "Improving")
    c.decide(10.0, 20.0, "Low", 0.9, "Improving")  # upswitch to 480p, cooldown=6
    # For 5 ticks, up-switch cannot trigger even if evidence accumulates
    for _ in range(5):
        d = c.decide(10.0, 20.0, "Low", 0.9, "Improving")
        assert d["quality"] == "480p"


def test_5_predicted_high_risk_downswitch():
    c = PredictiveController()
    c.index = 2  # 720p
    c.cooldown = 0
    dec = c.decide(1.0, 5.0, risk="High", confidence=0.75, trend="Degrading")
    assert dec["quality"] == "480p"
    assert dec["bitrate_mbps"] == 1.4
    assert dec["reason"] == "predicted_high_risk"
    assert c.cooldown == 6


def test_6_low_buffer_fallback():
    c = PredictiveController()
    c.index = 3  # 1080p
    dec = c.decide(10.0, 1.2, risk="Low", confidence=0.9, trend="Stable")
    assert dec["quality"] == "360p"
    assert dec["bitrate_mbps"] == 0.8
    assert dec["reason"] == "fallback"


def test_7_randomized_input_robustness():
    c = PredictiveController()
    random.seed(42)
    for _ in range(100):
        bw = random.uniform(0.1, 20.0)
        buf = random.uniform(0.0, 35.0)
        risk = random.choice(["Low", "Medium", "High"])
        conf = random.uniform(0.0, 1.0)
        trend = random.choice(["Stable", "Improving", "Degrading"])
        d = c.decide(bw, buf, risk, conf, trend)
        assert d["quality"] in valid_qualities
        assert d["bitrate_mbps"] in valid_bitrates


def test_8_invalid_edge_network_conditions():
    c = PredictiveController()
    # Unknown string or edge conditions must not crash the controller
    d1 = c.decide(5.0, 12.0, risk="UnknownRisk", confidence=0.8, trend="Volatile")
    assert d1["quality"] in valid_qualities
    assert d1["reason"] == "hold"


def test_9_zero_near_zero_throughput():
    c = PredictiveController()
    c.index = 2  # 720p
    d0 = c.decide(0.0, 15.0, risk="High", confidence=0.8, trend="Degrading")
    assert d0["quality"] == "480p"
    assert d0["reason"] == "predicted_high_risk"
    d_near = c.decide(0.0001, 15.0, risk="Low", confidence=0.8, trend="Stable")
    assert d_near["quality"] in valid_qualities


def test_10_very_large_buffer():
    c = PredictiveController()
    d_large = c.decide(15.0, 1000.0, risk="Low", confidence=0.95, trend="Stable")
    assert d_large["quality"] in valid_qualities
    assert d_large["bitrate_mbps"] in valid_bitrates


def test_11_very_low_buffer():
    c = PredictiveController()
    c.index = 3  # 1080p
    d_zero = c.decide(10.0, 0.0, risk="Low", confidence=0.9, trend="Stable")
    assert d_zero["quality"] == "360p"
    assert d_zero["reason"] == "fallback"
    d_neg = c.decide(10.0, -1.5, risk="Low", confidence=0.9, trend="Stable")
    assert d_neg["quality"] == "360p"
    assert d_neg["reason"] == "fallback"


def test_12_confidence_threshold_boundaries():
    c = PredictiveController()
    c.index = 2  # 720p
    c.cooldown = 0
    # Just below 0.70 threshold: should NOT trigger predicted_high_risk
    d_sub = c.decide(1.0, 5.0, risk="High", confidence=0.699, trend="Degrading")
    assert d_sub["quality"] == "720p"
    assert d_sub["reason"] == "hold"

    # Exactly at or above 0.70 threshold: SHOULD trigger predicted_high_risk
    d_at = c.decide(1.0, 5.0, risk="High", confidence=0.700, trend="Degrading")
    assert d_at["quality"] == "480p"
    assert d_at["reason"] == "predicted_high_risk"

    # Boundary at 1.0
    c.cooldown = 0
    d_max = c.decide(1.0, 5.0, risk="High", confidence=1.000, trend="Degrading")
    assert d_max["quality"] == "360p"
    assert d_max["reason"] == "predicted_high_risk"
