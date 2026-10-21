from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from src.adaptation.controller import PredictiveController, LADDER
from src.simulation.runtime import scenario

SYNTHETIC_SCENARIOS = ["stable", "sudden_drop", "recovery", "degradation", "congestion", "mobile"]
METHODS = ["Fixed 1080p", "Rule-based", "Throughput", "Buffer-based", "Proposed"]


def simulate_trace(bandwidth_series: list[float] | np.ndarray, method: str) -> dict:
    controller = PredictiveController()
    buffer = 4.0
    stalls = 0.0
    switches = 0
    total_bitrate = 0.0
    last_rate = None
    duration = max(1, len(bandwidth_series))

    for bw in bandwidth_series:
        if method == "Fixed 1080p":
            rate = 5.0
        elif method == "Throughput":
            rate = max(r for _, r in LADDER if r <= bw * 0.8) if bw >= 1.0 else 0.8
        elif method == "Buffer-based":
            rate = 0.8 if buffer < 5.0 else (2.8 if buffer < 12.0 else 5.0)
        elif method == "Rule-based":
            rate = 0.8 if bw < 2.0 else (2.8 if bw < 5.0 else 5.0)
        else:  # Proposed
            risk = "High" if buffer < 5.0 and bw < 3.0 else ("Medium" if buffer < 10.0 else "Low")
            trend = "Stable" if bw >= 3.0 else "Degrading"
            decision = controller.decide(bw, buffer, risk, 0.85, trend)
            rate = decision["bitrate_mbps"]

        if last_rate is not None and rate != last_rate:
            switches += 1
        last_rate = rate

        download = 4.0 * rate / max(0.1, bw)
        stall = max(0.0, download - buffer)
        stalls += stall
        buffer = min(30.0, max(0.0, buffer - download) + 4.0)
        total_bitrate += rate

    avg_bitrate = total_bitrate / duration
    # ITU-T QoE formula: bitrate - 4.3 * rebuffer_seconds
    qoe = avg_bitrate - 4.3 * stalls
    return {
        "rebuffer_seconds": round(stalls, 3),
        "avg_bitrate_mbps": round(avg_bitrate, 3),
        "switches": int(switches),
        "qoe": round(qoe, 3),
    }


def run() -> list[dict]:
    # 1. Synthetic Scenarios
    syn_results_by_method = {m: {"rebuffer": [], "bitrate": [], "switches": [], "qoe": []} for m in METHODS}
    for sc in SYNTHETIC_SCENARIOS:
        values = scenario(sc, 60, seed=42)
        for m in METHODS:
            res = simulate_trace(values, m)
            syn_results_by_method[m]["rebuffer"].append(res["rebuffer_seconds"])
            syn_results_by_method[m]["bitrate"].append(res["avg_bitrate_mbps"])
            syn_results_by_method[m]["switches"].append(res["switches"])
            syn_results_by_method[m]["qoe"].append(res["qoe"])

    # 2. Real Validation Traces
    real_results_by_method = {m: {"rebuffer": [], "bitrate": [], "switches": [], "qoe": []} for m in METHODS}
    traces_path = Path("data/processed/traces.csv")
    if traces_path.exists():
        df_traces = pd.read_csv(traces_path)
        val_traces = df_traces[df_traces.split == "validation"]
        for _, g in val_traces.groupby("trace_id"):
            values = g["throughput_mbps"].values
            for m in METHODS:
                res = simulate_trace(values, m)
                real_results_by_method[m]["rebuffer"].append(res["rebuffer_seconds"])
                real_results_by_method[m]["bitrate"].append(res["avg_bitrate_mbps"])
                real_results_by_method[m]["switches"].append(res["switches"])
                real_results_by_method[m]["qoe"].append(res["qoe"])

    rows = []
    # Add synthetic averages
    for m in METHODS:
        rows.append({
            "method": m,
            "environment": "Synthetic (6 Scenarios)",
            "rebuffer_seconds": round(float(np.mean(syn_results_by_method[m]["rebuffer"])), 3),
            "avg_bitrate_mbps": round(float(np.mean(syn_results_by_method[m]["bitrate"])), 3),
            "switches": round(float(np.mean(syn_results_by_method[m]["switches"])), 1),
            "qoe": round(float(np.mean(syn_results_by_method[m]["qoe"])), 3),
        })

    # Add real trace averages
    if traces_path.exists():
        for m in METHODS:
            rows.append({
                "method": m,
                "environment": "Real (88 Validation Traces)",
                "rebuffer_seconds": round(float(np.mean(real_results_by_method[m]["rebuffer"])), 3),
                "avg_bitrate_mbps": round(float(np.mean(real_results_by_method[m]["bitrate"])), 3),
                "switches": round(float(np.mean(real_results_by_method[m]["switches"])), 1),
                "qoe": round(float(np.mean(real_results_by_method[m]["qoe"])), 3),
            })

    Path("reports").mkdir(exist_ok=True)
    with open("reports/baseline_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


