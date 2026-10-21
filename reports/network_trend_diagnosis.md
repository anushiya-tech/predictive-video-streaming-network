# Network Condition Prediction Diagnosis

## 1. Executive Summary

Evaluation of the current production model (`LogisticRegression` with `class_weight='balanced'`) on the protected 59-trace holdout yielded **47.49% accuracy** and **0.4032 macro-F1**. 
On the existing validation set (16,254 observations across 88 complete traces), the model achieves **45.37% accuracy** and **0.3751 macro-F1**, which fails to exceed even the trivial majority-class accuracy (**47.00%**).

This diagnostic investigates why this task is challenging and identifies structural, statistical, and modeling root causes.

---

## 2. Current Label Definition and Prediction Horizon

In [`src/data/labels.py`](file:///c:/Users/ELCOT/predictive-video-streaming-network/src/data/labels.py#L4-L18):
- **Horizon ($H$):** 5 seconds ($t+1$ to $t+5$).
- **Past Reference Window:** Past 5 seconds ($t-4$ to $t$).
- **Ratio Metric:**
  $$\text{ratio} = \frac{\frac{1}{5}\sum_{i=1}^{5} \text{throughput}_{t+i}}{\max\left(0.01, \frac{1}{5}\sum_{j=0}^{4} \text{throughput}_{t-j}\right)} - 1$$
- **Thresholds:**
  - $\text{ratio} \ge +0.15 \implies \text{\textbf{Improving}}$
  - $\text{ratio} \le -0.35 \implies \text{\textbf{Severely Degrading}}$
  - $-0.35 < \text{ratio} \le -0.15 \implies \text{\textbf{Degrading}}$
  - $-0.15 < \text{ratio} < +0.15 \implies \text{\textbf{Stable}}$

---

## 3. Class Distribution

The dataset consists of 101,851 real rows across 587 traces. The clean training set (excluding the protected 59-trace holdout) and the validation set show closely aligned, imbalanced 4-class distributions:

| Class | Train (excl. holdout) Count | Train % | Validation Count | Validation % |
|---|---|---|---|---|
| **Stable** | 28,457 | 48.06% | 7,640 | 47.00% |
| **Improving** | 15,905 | 26.86% | 4,378 | 26.93% |
| **Degrading** | 9,709 | 16.40% | 2,862 | 17.61% |
| **Severely Degrading** | 5,144 | 8.69% | 1,374 | 8.45% |
| **Total** | **59,215** | **100.0%** | **16,254** | **100.0%** |

---

## 4. Baseline Evaluation (Validation Set)

Evaluated strictly on the 16,254 validation rows (no holdout data used):

| Baseline | Accuracy | Macro-F1 | Precision | Recall | Description |
|---|---|---|---|---|---|
| **Majority Class** | **47.00%** | 0.1599 | 0.1175 | 0.2500 | Always predicts `Stable` |
| **Lag-5 Elapsed Condition** | **37.01%** | 0.2677 | 0.2677 | 0.2677 | Predicts condition observed over past 5s vs preceding 5s (strictly past information) |
| **Past-Trend Feature** | **37.01%** | 0.2677 | 0.2677 | 0.2677 | Same as Lag-5 ratio mapped to the 4 classes |
| **Lag-1 Label Persistence** | **71.28%** | 0.6843 | 0.6845 | 0.6842 | Predicts ground-truth label from $t-1$ (*Theoretical ceiling only; not realizable in real-time*) |
| **Current Model (Logistic)** | **45.37%** | **0.3751** | 0.3835 | 0.4275 | Balanced Logistic Regression from `models/network_condition_best.joblib` |
| **Candidate (Random Forest)** | **54.42%** | **0.3595** | 0.4224 | 0.3821 | Balanced Random Forest (from `reports/model_comparison.csv`) |
| **Candidate (HistGradient)** | **54.17%** | **0.3431** | 0.3773 | 0.3757 | HistGradientBoosting (from `reports/model_comparison.csv`) |

### Key Observations:
1. The **Lag-5 trend persistence baseline (37.01% accuracy)** performs far worse than majority class (47.00%), proving that mobile throughput does not exhibit simple trend momentum over 5-second horizons.
2. The **Lag-1 persistence baseline (71.28%)** is high solely because sliding 5-second windows at $t$ and $t-1$ overlap by 4 seconds ($t+1$ to $t+4$). However, the true label of $t-1$ is unknown at time $t$ because its future horizon has not completed.
3. The current production model (Logistic Regression) has **lower accuracy (45.37%) than the majority-class baseline (47.00%)**.

---

## 5. Current Model Validation Performance Breakdown

### Classification Report (Validation Set):
```
                    precision    recall  f1-score   support

         Degrading     0.2003    0.0996    0.1330      2862
         Improving     0.4665    0.7574    0.5774      4378
Severely Degrading     0.2055    0.4381    0.2797      1374
            Stable     0.6616    0.4151    0.5101      7640

          accuracy                         0.4537     16254
         macro avg     0.3835    0.4275    0.3751     16254
      weighted avg     0.4893    0.4537    0.4424     16254
```

### Confusion Matrix:
| Ground Truth \ Predicted | Stable | Improving | Degrading | Severely Degrading |
|---|---|---|---|---|
| **Stable (7,640)** | 3,171 (41.5%) | 2,450 (32.1%) | 914 (12.0%) | 1,105 (14.5%) |
| **Improving (4,378)** | 473 (10.8%) | 3,316 (75.7%) | 113 (2.6%) | 476 (10.9%) |
| **Degrading (2,862)** | 923 (32.2%) | 907 (31.7%) | 285 (10.0%) | 747 (26.1%) |
| **Severely Degrading (1,374)** | 226 (16.4%) | 435 (31.7%) | 111 (8.1%) | 602 (43.8%) |

**Core Failure Modes:**
- **Severe false alarms on `Stable`:** Over 58% of truly stable intervals are falsely predicted as changing (2,450 predicted as Improving, 2,019 predicted as Degrading or Severely Degrading).
- **Inability to distinguish `Degrading`:** For actual `Degrading` events, the model achieves only **10.0% recall** and **20.0% precision**, misclassifying 32.2% as `Stable` and 31.7% as `Improving`.

---

## 6. Likely Causes of Weak Performance (Supported by Repository Evidence)

### 1. Rapid Autocorrelation Decay in Mobile Traces
Measured autocorrelation of `throughput_mbps` across validation traces:
- Lag 1s: **0.6740**
- Lag 2s: **0.5884**
- Lag 3s: **0.5016**
- Lag 4s: **0.4373**
- Lag 5s: **0.3878**
- Lag 10s: **0.2016**

In driving, bus, tram, and metro environments, channel fading and handovers decorrelate throughput rapidly. Predicting average throughput 5 seconds ahead based purely on past throughput approaches the signal noise floor.

### 2. Strong Empirical Mean Reversion (Contrarian Transitions)
Analyzing transitions between the past 5s trend and the subsequent 5s label reveals strong mean-reversion rather than persistence:
- When past 5s trend was **Severely Degrading**: **51.8%** of the subsequent 5s ground-truth intervals were **Improving**, and only **9.3%** remained Severely Degrading.
- When past 5s trend was **Degrading**: **37.4%** rebounded to **Improving**, **34.8%** leveled to **Stable**, and only **17.4%** continued Degrading.
- When past 5s trend was **Improving**: **47.0%** leveled to **Stable**, and only **25.4%** continued Improving.

Linear trend models that extrapolate positive slope into "Improving" and negative slope into "Degrading" predict the exact opposite of what physics (burst completion and channel recovery) dictates.

### 3. Low Median Throughput Amplifying Ratio Sensitivity
From `features.csv` summary statistics:
- 25th percentile throughput: **0.65 Mbps**
- Median throughput: **1.04 Mbps**
- Mean throughput: **1.24 Mbps**

Because the label uses a relative ratio:
$$\text{ratio} = \frac{\bar{T}_{\text{future}}}{\bar{T}_{\text{past}}} - 1$$
at 0.8 Mbps, an unnoticeable noise fluctuation of $\pm 0.12$ Mbps ($\pm 15\%$) flips a label between `Stable`, `Improving`, and `Degrading`. The ratio definition treats a 120 kbps fluctuation on a 0.8 Mbps link identically to a 1.5 Mbps swing on a 10 Mbps link.

### 4. Missing Physical and Network Telemetry
In `data/processed/traces.csv`, fields `latency_ms`, `loss_pct`, and `jitter_ms` are **100% NaN** across all 101,851 rows. The model has zero signal regarding transport-layer RTT spikes, packet drop surges, or bufferbloat, which are the standard leading indicators of cellular network degradation before throughput collapses.

### 5. Metric Artifact of Model Selection Rule
In [`src/ml/train.py`](file:///c:/Users/ELCOT/predictive-video-streaming-network/src/ml/train.py#L28), models were selected purely on `Macro F1`:
```python
if best is None or metrics["Macro F1"] > best[0]:
    best = (metrics["Macro F1"], model, metrics)
```
`LogisticRegression(class_weight="balanced")` achieved Macro F1 **0.3751** vs Random Forest's **0.3595**, solely because balanced weighting aggressively penalized minority errors, forcing predictions into `Severely Degrading` and `Degrading`. However, this collapsed precision to ~20% and reduced overall accuracy to **45.37%** (below Random Forest's **54.42%** and below majority baseline **47.00%**).

---

## 7. Recommended Next Experiment

To systematically address the weakness without altering the protected holdout or breaking frozen configurations:

1. **Direct Throughput Regression or 3-Class Coarsening:**
   - Explore predicting absolute expected throughput $\hat{T}_{t+5}$ directly via regression (and deriving trend from predicted magnitude), OR merge `Degrading` and `Severely Degrading` into a single `Degrading` class (creating a 3-class target: `Improving`, `Stable`, `Degrading`), avoiding the noisy boundary at $-0.35$.
2. **Mean-Reversion & Non-Linear Tree Ensembles:**
   - Incorporate mean-reversion features (e.g. current throughput relative to long-term rolling median $T_t / \text{median}_{30}(T)$) to allow trees/gradient boosters to learn rebound dynamics.
3. **Threshold Calibration on Validation Set:**
   - Use validation probabilities from `HistGradientBoosting` or `RandomForest` with tuned classification thresholds optimizing a composite of Macro-F1 and Accuracy, ensuring accuracy remains comfortably above the 47.00% majority baseline.
