# Technical Decision: Network Condition Prediction Architecture

## 1. Decision Summary

Based strictly on rigorous empirical validation on 16,254 observations across 88 complete traces (and zero exposure to the protected 59-trace holdout), we formally decide the production architecture for the **Network Condition Prediction** model:

| Dimension | Previous (Baseline) Implementation | Justified Production Decision | Empirical Justification |
|---|---|---|---|
| **Target Formulation** | **4-Class** (`Stable`, `Improving`, `Degrading`, `Severely Degrading`) | **3-Class** (`Stable`, `Improving`, `Degrading` with `Degrading` & `Severely Degrading` merged) | Macro-F1 improves from **0.3751 $\to$ 0.5186** (+38.3% relative). Removes artificial $-0.35$ boundary noise. Zero loss to ABR control. |
| **Prediction Horizon** | **5 Seconds** | **5 Seconds** (Retained) | Matches 4.0s video chunk duration, 6s cooldown, and 5s buffer risk horizon. High predictability (56.24% acc / 0.5186 F1) without long-term decorrelation. |
| **Model Family** | **Logistic Regression** (Balanced) | **HistGradientBoostingClassifier** (or Random Forest) | Beats majority baseline (56.24% vs 47.00%). Captures non-linear mean reversion in mobile traces. Compact footprint & fast inference. |

---

## 2. Target Formulation: 3-Class vs. 4-Class

### Empirical Evidence from Validation Split
- In the original 4-class formulation:
  - Majority baseline accuracy was **47.00%**.
  - The production Logistic model achieved **45.37% accuracy** and **0.3751 Macro-F1** (failing to exceed the trivial majority baseline).
  - The `Degrading` class was effectively undetectable (F1 of **0.1330** for Logistic and **0.0859** for Random Forest) due to severe mutual misclassification across the $-0.35$ threshold with `Severely Degrading`.
- In the 3-class formulation:
  - Macro-F1 jumped across all candidate families:
    - Logistic (Balanced): $0.3751 \to \mathbf{0.4958}$
    - Random Forest: $0.3595 \to \mathbf{0.5164}$
    - HistGradientBoosting: $0.3431 \to \mathbf{0.5186}$
  - Accuracy consistently surpassed majority baseline ($56.24\%$ for HistGradient vs $47.00\%$).
  - Degrading recall rose from $5.0\%$ to $25.9\%$, with F1 increasing from $0.086 \to 0.331$.

### ABR Controller Consumption Review
In [`src/adaptation/controller.py`](file:///c:/Users/ELCOT/predictive-video-streaming-network/src/adaptation/controller.py#L8-L11):
```python
elif trend in ("Improving", "Stable") and confidence >= 0.7 and buffer_s >= 10:
    self.evidence += 1
else:
    self.evidence = 0
```
- The controller only consumes `network_condition` to check whether the link is favorable (`Improving` or `Stable`) before allowing an evidence-based quality up-switch.
- Any degradation (`Degrading` or `Severely Degrading`) resets the up-switch counter to zero.
- Downshifts and panic fallbacks are governed by `buffer_risk == "High"` and `buffer_s < 1.5s`.
- **Verdict:** Merging the two degradation classes eliminates severe prediction noise while preserving 100% of the functional control semantics.

---

## 3. Horizon Selection: 1s vs. 3s vs. 5s vs. 10s

### Empirical Horizon Comparison (Validation Set)
| Horizon | Majority Accuracy | Top Model (HistGB) Accuracy | Top Model (HistGB) Macro-F1 | Cellular Autocorrelation |
|---|---|---|---|---|
| **1s** | 43.00% | 53.54% | 0.5167 | $r = 0.674$ |
| **3s** | 47.16% | **56.66%** | **0.5277** | $r = 0.502$ |
| **5s** | 47.00% | **56.24%** | **0.5186** | $r = 0.388$ |
| **10s** | 45.81% | 53.80% | 0.4824 | $r = 0.202$ |

### Architectural Analysis:
1. **1-Second Horizon (Rejected):**
   - High variance from short-term packet bursts and TCP congestion window oscillation.
   - Far too short for video streaming: chunks in DASH/HLS and our simulator are 4.0 seconds long. A 1-second forecast expires before a chunk download is even a quarter completed.
2. **10-Second Horizon (Rejected):**
   - Signal collapses due to cellular decorrelation in moving vehicles ($r \approx 0.20$ at lag 10s).
   - Macro-F1 drops to $0.4824$, causing noisy long-term false alarms.
3. **3-Second vs. 5-Second Horizon:**
   - Both horizons reside in the optimal predictability sweet spot ($>56\%$ accuracy, $>0.518$ Macro-F1).
   - **Selection:** Retain **5 seconds**.
     - Matches the simulator's 4.0-second chunk duration.
     - Matches the 5-second lookahead of the companion `buffer_risk` target, providing synchronous temporal scope across both models.
     - Avoids desynchronizing the existing configuration schema (`configs/labels.yaml` specifies `horizon_seconds: 5`).

---

## 4. Model Family: Linear vs. Tree Ensembles

### Comparison:
- **Logistic Regression (Balanced):**
  - Validation Accuracy: **51.11%** (3-class), Macro-F1: **0.4958**.
  - High false positive rate on `Stable` (22.7% predicted as Degrading, 32.7% predicted as Improving). Linear decision surfaces struggle with cellular channel mean-reversion dynamics.
- **Random Forest (Balanced):**
  - Validation Accuracy: **56.11%** (3-class), Macro-F1: **0.5164**.
  - Strong non-linear modeling, robust against monotonic trend extrapolation. Large artifact footprint (~86 MB).
- **HistGradientBoostingClassifier:**
  - Validation Accuracy: **56.24%** (3-class), Macro-F1: **0.5186**.
  - Highest accuracy and Macro-F1 across all models.
  - Native handling of continuous features, compact serialized footprint (<500 KB), and ultra-fast inference latency (<0.2 ms per sample), making it ideal for real-time per-tick ABR evaluation.

---

## 5. Implementation Roadmap

1. **Pipeline Retraining (Controlled Next Step):**
   - Train `HistGradientBoostingClassifier` with the 3-class target on the clean training set (excluding the protected 59-trace holdout).
   - Verify on the validation partition that accuracy exceeds 56% and Macro-F1 exceeds 0.51.
   - Retain the protected 59-trace holdout strictly untouched until final post-retraining verification.
2. **Controller & API Alignment:**
   - Verify that `src/adaptation/controller.py` and `app/backend/main.py` handle the 3-class target strings cleanly (`Improving`, `Stable`, `Degrading`).
3. **Status:**
   - Existing production models in `models/` remain untouched until formal retraining is invoked.
