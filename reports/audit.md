# Final Audit — 2026-09-28

## Dependency map

`data/raw` → `scripts/prepare_dataset.py` → `data/processed/traces.csv` → `src/streaming/simulator.py` / `scripts/create_features.py` → `data/features/features.csv` → `src/data/labels.py` / `scripts/create_labels.py` → `src/ml/train.py` → `models/*_best.joblib` → `src/ml/predict.py` → `src/simulation/runtime.py` → `src/adaptation/controller.py` → `app/backend/main.py` → `app/frontend/index.html` → reports.

## Evidence and decisions

| Stage | Status | Evidence | Decision |
|---|---|---|---|
| Real data | working | `data/processed/traces.csv`, 101,851 rows / 587 traces | Keep. |
| Preprocessing and split | working with limitation | `scripts/prepare_dataset.py`, `data/processed/split.json` | Keep old split as before-comparison only. |
| Features / labels | weak | `src/features/build.py`, `src/data/labels.py`; throughput-only real data and current simulated player state | Keep foundation; add stricter temporal tests before new training. |
| Models | weak | `reports/final_test_metrics.json`: trend macro-F1 0.3901, buffer risk 0.8960 | Do not claim target achievement; redesign requires an unconsumed holdout or nested grouped CV. |
| Simulator / controller | incomplete | `src/simulation/runtime.py`, `src/adaptation/controller.py` | Retain as simulated ABR demo; it is not a real player or HLS/DASH implementation. |
| Baselines | unrealistic | `reports/baseline_comparison.csv` covers one generated sudden-drop scenario | Replace comparison methodology before drawing performance claims. |
| API / dashboard | incomplete | `app/backend/main.py`, `app/frontend/index.html` | API works for a local simulated demo but has no database or real model inference in its loop. |
| Video | incomplete | no `videos/` rendition assets; `ffmpeg` absent in Phase 0 | Do not claim DASH/HLS. |

## Existing command evidence

The immediately preceding run recorded `pytest -q` as 6 passing tests, created `reports/final_test_metrics.json`, and created `reports/baseline_comparison.csv`. The protected old split has already been evaluated, so it must not be used for new model selection.

## Leakage review

`build_features` uses windows ending at the current row, while `create_labels` uses rows beginning at `index + 1`; existing tests cover one future-mutation case. However, buffer-risk labels are strongly related to current player buffer and the evaluation lacks a current-buffer-only baseline. The current workflow also trains on whole-trace splits, which prevents trace overlap. No evidence supports treating simulated latency, loss, or jitter as real measurements: they are absent from all 101,851 processed real rows.

## Required replacement rationale

Any renewed final model comparison must define a newly frozen whole-trace holdout and retrain excluding it, or use nested trace-grouped CV. The current results remain an honest before-comparison only.
