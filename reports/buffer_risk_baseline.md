# Buffer Risk Prediction: Baseline Benchmark & Evaluation

## 1. Executive Summary

In [`reports/audit.md`](file:///c:/Users/ELCOT/predictive-video-streaming-network/reports/audit.md), an important concern was raised regarding whether the high performance of the `buffer_risk` model ($>90\%$ accuracy) was merely an artifact of reading the current player buffer state (`buffer_s`), which is trivially known at decision time.

To resolve this question rigorously, this evaluation benchmarks:
1. **Majority Class Baseline:** Predicts the most frequent class (`High`).
2. **Current-Buffer-Only Heuristic:** Classifies risk directly from the current buffer occupancy at decision time $t$ using the exact label thresholds:
   - $\text{buffer}_t < 3\text{s} \implies \text{High}$
   - $3\text{s} \le \text{buffer}_t < 8\text{s} \implies \text{Medium}$
   - $\text{buffer}_t \ge 8\text{s} \implies \text{Low}$
3. **Production Model (Random Forest):** The trained production pipeline in [`models/buffer_risk_best.joblib`](file:///c:/Users/ELCOT/predictive-video-streaming-network/models/buffer_risk_best.joblib) using all past features (buffer state, throughput rolling statistics, bandwidth margin, depletion rate).

All models were evaluated strictly on the **16,254 rows of the validation partition** (88 complete traces). The protected 59-trace holdout was **not used**.

### Primary Findings
- **The Current-Buffer Heuristic fails completely on `High` risk:** It achieves **0.00% recall** and **0.00% precision** on `High` risk events (F1 = 0.0000).
- **The Production Model achieves 92.01% recall and 89.55% precision on `High` risk** (F1 = 0.9077).
- **Conclusion:** The ML model is **not** trivially copying the current buffer. It successfully learns the dynamic depletion relationship between upcoming throughput and video bitrate, detecting buffer underruns before they happen.

---

## 2. Background: The Buffer Risk Target Definition

In [`src/data/labels.py`](file:///c:/Users/ELCOT/predictive-video-streaming-network/src/data/labels.py#L13-L14):
$$\text{projected} = \text{buffer}_t - \sum_{i=1}^{5} \max\left(0, \frac{\text{bitrate}_t}{\max(0.05, \text{throughput}_{t+i})} - 1\right)$$
$$\text{risk} = \begin{cases} \text{\textbf{High}} & \text{if projected} < 3.0 \\ \text{\textbf{Medium}} & \text{if } 3.0 \le \text{projected} < 8.0 \\ \text{\textbf{Low}} & \text{if projected} \ge 8.0 \end{cases}$$

At decision time $t$:
- A player might have a seemingly healthy buffer of $4.5$ seconds ($\text{buffer}_t \ge 3.0$, so the static heuristic guesses `Medium`).
- However, if future network throughput drops below the current bitrate, the player burns buffer faster than chunks arrive. Over the next 5 seconds, the buffer collapses below 3 seconds ($\text{projected} < 3.0 \implies \text{High Risk}$).
- To detect this, a model must predict whether future throughput can sustain the bitrate.

---

## 3. Validation Performance Comparison

Evaluated on 16,254 validation samples across 88 complete traces:

| Metric | Majority Baseline | Current-Buffer Heuristic | Production Model (Random Forest) |
|---|---|---|---|
| **Accuracy** | 44.92% | 54.04% | **90.63%** (+36.59% over heuristic) |
| **Macro-F1** | 0.2066 | 0.4902 | **0.8940** (+0.4038 over heuristic) |
| **Precision** | 0.1497 | 0.4299 | **0.8988** |
| **Recall** | 0.3333 | 0.6517 | **0.8905** |

---

## 4. Detailed Per-Class Breakdown

### Validation Class Distribution
- **High Risk:** 7,301 samples (44.92%)
- **Low Risk:** 5,196 samples (31.97%)
- **Medium Risk:** 3,757 samples (23.11%)
- **Total:** 16,254 samples

### Per-Class Metrics:

#### 1. Majority Class Baseline
- **High:** Precision: 0.4492, Recall: 1.0000, F1: 0.6199 (Support: 7,301)
- **Low:** Precision: 0.0000, Recall: 0.0000, F1: 0.0000 (Support: 5,196)
- **Medium:** Precision: 0.0000, Recall: 0.0000, F1: 0.0000 (Support: 3,757)

#### 2. Current-Buffer-Only Heuristic
- **High:** Precision: **0.0000**, Recall: **0.0000**, F1: **0.0000** (Support: 7,301)
- **Low:** Precision: 0.9583, Recall: 1.0000, F1: **0.9787** (Support: 5,196)
- **Medium:** Precision: 0.3312, Recall: 0.9550, F1: **0.4919** (Support: 3,757)

#### 3. Production Random Forest Model
- **High:** Precision: **0.8955**, Recall: **0.9201**, F1: **0.9077** (Support: 7,301)
- **Low:** Precision: **0.9689**, Recall: **0.9956**, F1: **0.9821** (Support: 5,196)
- **Medium:** Precision: **0.8321**, Recall: **0.7559**, F1: **0.7922** (Support: 3,757)

---

## 5. Confusion Matrix Analysis

### Current-Buffer-Only Heuristic Confusion Matrix
| Ground Truth \ Predicted | High | Low | Medium |
|---|---|---|---|
| **High** (7,301) | **0** (0.0%) | 57 (0.8%) | **7,244** (99.2%) |
| **Low** (5,196) | 0 (0.0%) | **5,196** (100.0%) | 0 (0.0%) |
| **Medium** (3,757) | 0 (0.0%) | 169 (4.5%) | **3,588** (95.5%) |

*Diagnosis:* 
Because the video player behaviour policy maintains buffer levels above 3 seconds during stable periods, $\text{buffer}_t$ is almost never $< 3$ seconds at the start of a tick. Consequently, the naive heuristic predicts `Medium` for 7,244 intervals that subsequently suffer `High` risk! It is blind to upcoming stall hazards.

### Production Random Forest Confusion Matrix
| Ground Truth \ Predicted | High | Low | Medium |
|---|---|---|---|
| **High** (7,301) | **6,718** (92.0%) | 32 (0.4%) | 551 (7.5%) |
| **Low** (5,196) | 1 (0.0%) | **5,173** (99.6%) | 22 (0.4%) |
| **Medium** (3,757) | 783 (20.8%) | 134 (3.6%) | **2,840** (75.6%) |

*Diagnosis:*
The trained model accurately catches **6,718 out of 7,301 upcoming High Risk events (92.0% true positive rate)** while maintaining a precision of 89.6%. By combining `bandwidth_margin`, `buffer_depletion_rate`, and rolling throughput trend features with `buffer_s`, the model successfully anticipates buffer collapses before they manifest.

---

## 6. Conclusion
The experimental audit confirms that:
1. Predicting buffer risk requires joint modeling of player state and network trajectory; relying purely on current buffer occupancy yields an unusable model with zero detection capability for impending high-risk stalls.
2. The production Random Forest model delivers genuine predictive power, exceeding the current-buffer heuristic by **+36.6% accuracy** and **+0.4038 Macro-F1**.
