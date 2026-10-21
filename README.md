# Predictive Video Streaming Network

An end-to-end Machine Learning-enhanced Adaptive Bitrate (ABR) video streaming platform that anticipates throughput fluctuations and buffer starvation risks, proactively selecting video qualities to prevent playback stalls and eliminate bitrate oscillation.

---

## Features
- **Proactive ABR Controller:** Uses forward-looking ML predictions to prevent stalls and buffer collapses before they happen.
- **3-Class Network Condition Model:** Predicts future 5-second network state (`Stable`, `Improving`, `Degrading`) using `HistGradientBoostingClassifier` (57.99% holdout accuracy / 0.5296 macro-F1).
- **Buffer Risk Prediction:** Predicts imminent buffer starvation risk (`Low`, `Medium`, `High`) using `RandomForestClassifier` (93.19% holdout accuracy / 0.8954 macro-F1 / 90.88% high-risk recall).
- **Temporal Hygiene:** Strict past-only feature construction and future-only label projection with complete-trace split isolation.
- **Protected 59-Trace Holdout:** Frozen holdout dataset protected by SHA-256 hash (`89a83dcc...`) never used for training or tuning.
- **Interactive Telemetry Dashboard:** Glassmorphic real-time UI with dual canvas charts, video simulation, ABR ladder, prediction telemetry, and baseline comparisons.
- **FastAPI Backend:** Modular REST API serving simulation streaming, online feature extraction, and real-time model inference.
- **Multi-Trace Benchmarking:** Evaluated against 4 standard ABR algorithms (Fixed 1080p, Rule-based, Throughput-based, Buffer-based) across 6 synthetic stress scenarios and 88 real cellular validation traces.

---

## Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.14.2)
- Node.js (optional, for frontend syntax checks)

### Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## Running the Application

### 1. Start the FastAPI Server & Dashboard
```powershell
python run.py
```
Open your browser to:
```
http://127.0.0.1:8000
```

### 2. Available Simulation Scenarios
The dashboard and backend support 6 simulated network environments:
- `stable`: High consistent 12.0 Mbps broadband connection.
- `sudden_drop`: Abrupt drop from 10.0 Mbps to 0.5 Mbps simulating a dead zone.
- `recovery`: Rebound from 0.8 Mbps bottleneck back to 8.0 Mbps.
- `degradation`: Gradual stepped decline from 8.0 Mbps down to 1.0 Mbps.
- `congestion`: Sinusoidal bandwidth oscillation between 1.0 and 5.0 Mbps.
- `mobile`: High-variance cellular transit conditions.

---

## Running Tests

Run the full pytest suite with temporary directory isolation:
```powershell
pytest -v --basetemp=.pytest_tmp
```

Run backend API scenario verification:
```powershell
python scripts/verify_backend_api.py
```

---

## Model Evaluation and Calibration

### Final Holdout Evaluation (Protected 59 Traces)
```powershell
python scripts/evaluate_models.py --final --force
```
Outputs per-class precision, recall, F1, accuracy, and confusion matrices to `reports/final_test_metrics.json`.

### Model Calibration Evaluation
```powershell
python scripts/evaluate_model_calibration.py
```
Generates Brier score, ECE, MCE, and reliability bin statistics in `reports/model_calibration_metrics.json` and `reports/model_calibration.md`.

### Baseline Benchmarking
```powershell
python scripts/run_baselines.py
```
Generates comparison table across synthetic scenarios and 88 real traces in `reports/baseline_comparison.csv`.

---

## Model Architecture & Bitrate Ladder

### Bitrate Ladder
- `360p`: 0.8 Mbps
- `480p`: 1.4 Mbps
- `720p`: 2.8 Mbps
- `1080p`: 5.0 Mbps

### Predictive Controller Logic
- **Critical Fallback:** Switches to `360p` immediately if buffer $< 1.5$ seconds.
- **Predictive Down-Switch:** Shifts down one level if `buffer_risk == "High"` with confidence $\ge 0.70$.
- **Evidence-Based Up-Switch:** Requires `network_condition` $\in \{\text{"Stable"}, \text{"Improving"}\}$ with confidence $\ge 0.70$ and buffer $\ge 10.0$ seconds sustained over 3 consecutive ticks.
- **Hysteresis Cooldown:** Enforces a 6-second refractory period following any quality switch to eliminate flapping.

---

## Limitations
1. **Simulated Video Stream:** In environments where `ffmpeg` is not pre-installed, the dashboard displays high-fidelity simulated canvas representations matching current resolution and playback status rather than live transcoded MP4/HLS segments.
2. **Raw Dataset Metrics:** The open-source FCC and Norway network traces report throughput measurements; supplementary latency, packet loss, and jitter are handled gracefully via NaN imputation in the feature pipeline.
