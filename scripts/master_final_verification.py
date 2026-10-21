import hashlib
import json
import subprocess
import sys
from pathlib import Path
import joblib
import pandas as pd
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.backend.main import app
from src.adaptation.controller import PredictiveController, LADDER


def verify_models():
    print("--- [1/8] Verifying Production Models ---")
    for target in ["network_condition", "buffer_risk", "recommended_quality"]:
        p = REPO_ROOT / f"models/{target}_best.joblib"
        assert p.exists(), f"Missing model artifact: {p}"
        artifact = joblib.load(p)
        assert "model" in artifact and "features" in artifact
        m = artifact["model"]
        classes = list(m.classes_)
        print(f"  {target}: {type(m.named_steps['model'] if hasattr(m, 'named_steps') else m).__name__}, classes={classes}")
    print("  Models verified successfully.\n")


def verify_holdout_hash():
    print("--- [2/8] Verifying Protected Holdout Integrity ---")
    holdout_path = REPO_ROOT / "data/processed/final_holdout.json"
    content = holdout_path.read_text()
    data = json.loads(content)
    traces = data["trace_ids"]
    assert len(traces) == 59, f"Expected 59 traces, got {len(traces)}"
    expected_sha = "89a83dcc3e15f3c109240a0f013a8bfcd8480f238cafb6e45d22974b2a581823"
    calc_sha = hashlib.sha256("\n".join(traces).encode()).hexdigest()
    assert calc_sha == expected_sha, f"SHA mismatch: {calc_sha} != {expected_sha}"
    assert data.get("sha256") == expected_sha, "Stored sha256 mismatch"
    print(f"  Holdout traces: 59, SHA-256: {calc_sha} [EXACT MATCH]\n")


def verify_holdout_evaluation():
    print("--- [3/8] Running & Verifying Final Holdout Evaluation ---")
    # Run evaluation script with --final --force
    cmd = [sys.executable, str(REPO_ROOT / "scripts/evaluate_models.py"), "--final", "--force"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"evaluate_models.py failed: {res.stderr}"

    report_path = REPO_ROOT / "reports/final_test_metrics.json"
    metrics = json.loads(report_path.read_text())
    assert metrics["holdout_traces"] == 59
    assert metrics["holdout_samples"] == 6711

    nc = metrics["targets"]["network_condition"]
    assert nc["classes"] == ["Degrading", "Improving", "Stable"]
    assert nc["accuracy"] >= 0.57, f"Unexpected nc accuracy: {nc['accuracy']}"
    assert nc["macro_f1"] >= 0.52, f"Unexpected nc macro_f1: {nc['macro_f1']}"

    br = metrics["targets"]["buffer_risk"]
    assert br["classes"] == ["High", "Low", "Medium"]
    assert br["accuracy"] >= 0.93, f"Unexpected br accuracy: {br['accuracy']}"
    assert br["macro_f1"] >= 0.89, f"Unexpected br macro_f1: {br['macro_f1']}"

    print(f"  Holdout network_condition: Acc={nc['accuracy']:.4f}, Macro-F1={nc['macro_f1']:.4f}")
    print(f"  Holdout buffer_risk:       Acc={br['accuracy']:.4f}, Macro-F1={br['macro_f1']:.4f}\n")


def verify_controller():
    print("--- [4/8] Verifying Predictive Controller ---")
    c = PredictiveController()
    dec = c.decide(10.0, 5.0, "Low", 0.9, "Stable")
    assert dec["quality"] == "360p"
    # test evidence-based upswitch
    c.decide(10.0, 20.0, "Low", 0.9, "Improving")
    c.decide(10.0, 20.0, "Low", 0.9, "Improving")
    d_up = c.decide(10.0, 20.0, "Low", 0.9, "Improving")
    assert d_up["quality"] == "480p"
    assert d_up["reason"] == "evidence_based_up_switch"
    # test high risk downswitch
    c.cooldown = 0
    d_down = c.decide(1.0, 5.0, "High", 0.85, "Degrading")
    assert d_down["quality"] == "360p"
    assert d_down["reason"] == "predicted_high_risk"
    print("  Controller logic verified successfully.\n")


def verify_backend_scenarios():
    print("--- [5/8] Verifying Backend APIs and 6 Scenarios ---")
    client = TestClient(app)
    scenarios = ["stable", "sudden_drop", "recovery", "degradation", "congestion", "mobile"]
    for sc in scenarios:
        res = client.post("/api/simulation/scenario", json={"scenario": sc})
        assert res.status_code == 200
        hist = client.get("/api/network/history").json()
        assert len(hist) == 60
        last = hist[-1]
        assert last["prediction"]["network_condition"] in {"Stable", "Improving", "Degrading"}
        assert last["prediction"]["buffer_risk"] in {"High", "Medium", "Low"}
        assert 0.0 <= last["prediction"]["network_confidence"] <= 1.0
        assert 0.0 <= last["prediction"]["buffer_confidence"] <= 1.0
        assert last["quality"] in {"360p", "480p", "720p", "1080p"}
    print("  All 6 scenarios verified (60 ticks each, valid predictions and qualities).\n")


def verify_dashboard_js():
    print("--- [6/8] Verifying Dashboard & JavaScript Syntax ---")
    html_path = REPO_ROOT / "app/frontend/index.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Predictive Video Streaming Network" in html
    start = html.find("<script>") + len("<script>")
    end = html.find("</script>", start)
    js_code = html[start:end]

    tmp_js = REPO_ROOT / ".pytest_tmp/final_js_check.js"
    tmp_js.write_text(js_code, encoding="utf-8")
    res = subprocess.run(["node", "--check", str(tmp_js)], capture_output=True, text=True)
    if tmp_js.exists():
        tmp_js.unlink()
    assert res.returncode == 0, f"JS syntax check failed: {res.stderr}"
    print("  Dashboard HTML and JavaScript verified (Node.js syntax check passed: 0 errors).\n")


def verify_baselines_and_calibration():
    print("--- [7/8] Verifying Baselines and Calibration Reports ---")
    b_path = REPO_ROOT / "reports/baseline_comparison.csv"
    assert b_path.exists()
    df_b = pd.read_csv(b_path)
    assert len(df_b) == 10
    assert set(df_b["method"].unique()) == {"Fixed 1080p", "Rule-based", "Throughput", "Buffer-based", "Proposed"}

    c_path = REPO_ROOT / "reports/model_calibration_metrics.json"
    assert c_path.exists()
    cal = json.loads(c_path.read_text())
    assert "network_condition_3class" in cal
    assert "buffer_risk" in cal
    print("  Baselines and calibration verified.\n")


def verify_pytest():
    print("--- [8/8] Verifying Test Suite (pytest) ---")
    cmd = [sys.executable, "-m", "pytest", "-q", "--basetemp=.pytest_tmp"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"pytest failed:\n{res.stdout}\n{res.stderr}"
    print(f"  Pytest passed: {res.stdout.strip().splitlines()[-1]}\n")


def main():
    print("==================================================")
    print("RUNNING FINAL MASTER VERIFICATION SUITE")
    print("==================================================")
    verify_models()
    verify_holdout_hash()
    verify_holdout_evaluation()
    verify_controller()
    verify_backend_scenarios()
    verify_dashboard_js()
    verify_baselines_and_calibration()
    verify_pytest()
    print("==================================================")
    print("ALL MASTER VERIFICATION STEPS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    main()

