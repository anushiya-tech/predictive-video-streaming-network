# Network Condition: 3-Class vs. 4-Class Experiment

## 1. Executive Summary

This experiment rigorously investigates whether reformulating the network condition prediction target from **4 classes** (`Stable`, `Improving`, `Degrading`, `Severely Degrading`) to **3 classes** (`Stable`, `Improving`, `Degrading` by merging `Degrading` and `Severely Degrading`) improves model learnability and predictive fidelity.

### Key Findings
1. **Macro-F1 substantially increased across all model families:**
   - **Logistic Regression (Balanced):** $0.3751 \to \mathbf{0.4958}$ (+32.2% relative)
   - **Random Forest (Balanced):** $0.3595 \to \mathbf{0.5164}$ (+43.6% relative)
   - **HistGradientBoosting:** $0.3431 \to \mathbf{0.5186}$ (+51.1% relative)
2. **Accuracy consistently surpassed the majority baseline:**
   - The majority class baseline accuracy is **47.00%** in both formulations.
   - In 4-class, the existing production Logistic Regression achieved only **45.37%** (below majority).
   - In 3-class, all non-dummy models exceed the baseline: Logistic reaches **51.11%** (balanced) / **54.25%** (unweighted), Random Forest reaches **56.11%**, and HistGradientBoosting reaches **56.24%**.
3. **The Root Mechanism:** Merging eliminates the noisy and fragile $-0.35$ ratio boundary that caused extensive confusion between `Degrading` and `Severely Degrading`.
4. **The Trade-off & System Impact:** The system loses the distinction between moderate decline and severe collapse. However, inside [`src/adaptation/controller.py`](file:///c:/Users/ELCOT/predictive-video-streaming-network/src/adaptation/controller.py), the adaptation logic only checks `if trend in ("Improving", "Stable")` for up-switching evidence; it never differentiated between `Degrading` and `Severely Degrading`. Down-switching is driven by `buffer_risk == "High"`. Thus, the loss of granularity carries zero practical cost for the existing ABR controller.
5. **Production Status:** The production models and pipelines remain **completely unchanged** at this stage pending review.

---

## 2. Methodology & Leakage Prevention

- **Datasets:** The training set strictly excludes the protected 59-trace holdout ([`data/processed/final_holdout.json`](file:///c:/Users/ELCOT/predictive-video-streaming-network/data/processed/final_holdout.json)).
- **Validation Partition:** Identical validation set (16,254 rows from 88 complete traces) evaluated for both 4-class and 3-class formulations.
- **Labels:** Original labels in `data/labels/labels.csv` were left unaltered. The 3-class target was created in-memory by mapping:
  $$\text{Degrading} + \text{Severely Degrading} \longrightarrow \text{Degrading}$$
- **Seed & Tunables:** Random seed fixed to `42` across all models.
- **Protected Holdout:** Completely untouched (not evaluated, not tuned against).

---

## 3. Class Distributions

### Train Partition (Excluding 59 Protected Holdout Traces, 59,215 rows)
| Class | 4-Class Count | 4-Class % | 3-Class Count | 3-Class % |
|---|---|---|---|---|
| **Stable** | 28,457 | 48.06% | 28,457 | 48.06% |
| **Improving** | 15,905 | 26.86% | 15,905 | 26.86% |
| **Degrading** | 9,709 | 16.40% | 14,853 | 25.08% |
| **Severely Degrading** | 5,144 | 8.69% | — | — |

### Validation Partition (88 Complete Traces, 16,254 rows)
| Class | 4-Class Count | 4-Class % | 3-Class Count | 3-Class % |
|---|---|---|---|---|
| **Stable** | 7,640 | 47.00% | 7,640 | 47.00% |
| **Improving** | 4,378 | 26.93% | 4,378 | 26.93% |
| **Degrading** | 2,862 | 17.61% | 4,236 | 26.06% |
| **Severely Degrading** | 1,374 | 8.45% | — | — |

*Note: In the 3-class formulation, the class distribution becomes noticeably more balanced: ~47% Stable, ~27% Improving, ~26% Degrading.*

---

## 4. Head-to-Head Performance Comparison (Validation Set)

| Model | Target | Accuracy | Macro-F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Majority Baseline** | 4-Class | 47.00% | 0.1599 | 0.1175 | 0.2500 |
| | **3-Class** | **47.00%** | **0.2132** | 0.1567 | 0.3333 |
| **Logistic Regression (Balanced)** | 4-Class | 45.37% | 0.3751 | 0.3835 | 0.4275 |
| *(Current Model Candidate)* | **3-Class** | **51.11%** | **0.4958** | **0.5087** | **0.5257** |
| **Logistic Regression (Unweighted)** | 4-Class | 53.39% | 0.2957 | 0.3327 | 0.3346 |
| | **3-Class** | **54.25%** | **0.4330** | **0.5404** | **0.4594** |
| **Random Forest (Balanced)** | 4-Class | 54.42% | 0.3595 | 0.4224 | 0.3821 |
| | **3-Class** | **56.11%** | **0.5164** | **0.5323** | **0.5235** |
| **Random Forest (Unweighted)** | 4-Class | 54.29% | 0.3641 | 0.4280 | 0.3844 |
| | **3-Class** | **55.80%** | **0.5162** | **0.5286** | **0.5237** |
| **HistGradientBoosting** | 4-Class | 54.17% | 0.3431 | 0.3773 | 0.3757 |
| | **3-Class** | **56.24%** | **0.5186** | **0.5330** | **0.5272** |

---

## 5. Detailed Per-Class Breakdown

### 4-Class Models (Validation Set)

#### Current Logistic Regression (Balanced)
- **Degrading:** Precision: 0.2003, Recall: 0.0996, F1: **0.1330** (Support: 2,862)
- **Improving:** Precision: 0.4665, Recall: 0.7574, F1: **0.5774** (Support: 4,378)
- **Severely Degrading:** Precision: 0.2055, Recall: 0.4381, F1: **0.2797** (Support: 1,374)
- **Stable:** Precision: 0.6616, Recall: 0.4151, F1: **0.5101** (Support: 7,640)

#### Random Forest (Balanced)
- **Degrading:** Precision: 0.3062, Recall: 0.0500, F1: **0.0859** (Support: 2,862)
- **Improving:** Precision: 0.5081, Recall: 0.6425, F1: **0.5675** (Support: 4,378)
- **Severely Degrading:** Precision: 0.2899, Recall: 0.0793, F1: **0.1246** (Support: 1,374)
- **Stable:** Precision: 0.5853, Recall: 0.7565, F1: **0.6600** (Support: 7,640)

---

### 3-Class Models (Validation Set)

#### Logistic Regression (Balanced)
- **Degrading (Merged):** Precision: 0.4066, Recall: 0.3520, F1: **0.3773** (Support: 4,236) — *(Recall increased from 10.0% to 35.2%, F1 jumped from 0.133 to 0.377)*
- **Improving:** Precision: 0.4594, Recall: 0.7796, F1: **0.5781** (Support: 4,378)
- **Stable:** Precision: 0.6601, Recall: 0.4455, F1: **0.5320** (Support: 7,640)

#### Random Forest (Balanced)
- **Degrading (Merged):** Precision: 0.4601, Recall: 0.2587, F1: **0.3312** (Support: 4,236) — *(Recall jumped from 5.0% to 25.9%, F1 jumped from 0.086 to 0.331)*
- **Improving:** Precision: 0.5312, Recall: 0.6122, F1: **0.5688** (Support: 4,378)
- **Stable:** Precision: 0.6054, Recall: 0.6995, F1: **0.6491** (Support: 7,640)

#### HistGradientBoosting
- **Degrading (Merged):** Precision: 0.4583, Recall: 0.2594, F1: **0.3313** (Support: 4,236)
- **Improving:** Precision: 0.5294, Recall: 0.6313, F1: **0.5759** (Support: 4,378)
- **Stable:** Precision: 0.6112, Recall: 0.6908, F1: **0.6486** (Support: 7,640)

---

## 6. Confusion Matrices Analysis

### 4-Class Logistic Regression (Balanced)
| Ground Truth \ Predicted | Degrading | Improving | Severely Degrading | Stable |
|---|---|---|---|---|
| **Degrading** (2,862) | **285** | 907 | **747** | 923 |
| **Improving** (4,378) | 113 | **3,316** | 476 | 473 |
| **Severely Degrading** (1,374) | **111** | 435 | **602** | 226 |
| **Stable** (7,640) | 914 | 2,450 | 1,105 | **3,171** |

*Analysis:* Actual `Degrading` events were predicted as `Severely Degrading` 747 times (26.1% of the time), while actual `Severely Degrading` was predicted as `Degrading` 111 times. The boundary between -15% and -35% introduces heavy misclassifications between adjacent negative states.

### 3-Class Logistic Regression (Balanced)
| Ground Truth \ Predicted | Degrading (Merged) | Improving | Stable |
|---|---|---|---|
| **Degrading** (4,236) | **1,491** (35.2%) | 1,517 (35.8%) | 1,228 (29.0%) |
| **Improving** (4,378) | 440 (10.1%) | **3,413** (78.0%) | 525 (12.0%) |
| **Stable** (7,640) | 1,736 (22.7%) | 2,500 (32.7%) | **3,404** (44.6%) |

*Analysis:* In 3-class, true negative detections surged from 285 to 1,491. While false positives from `Stable` remain an inherent challenge of short-term mobile throughput variance, the model now captures over a third of all degradation events with >40% precision.

### 3-Class HistGradientBoosting
| Ground Truth \ Predicted | Degrading (Merged) | Improving | Stable |
|---|---|---|---|
| **Degrading** (4,236) | **1,099** (25.9%) | 981 (23.2%) | 2,156 (50.9%) |
| **Improving** (4,378) | 413 (9.4%) | **2,764** (63.1%) | 1,201 (27.4%) |
| **Stable** (7,640) | 886 (11.6%) | 1,476 (19.3%) | **5,278** (69.1%) |

*Analysis:* HistGradientBoosting achieves a cleaner compromise: stable recall rises to **69.1%** and improving recall reaches **63.1%**, while degrading precision is **45.8%**, yielding the highest accuracy (**56.24%**) and highest Macro-F1 (**0.5186**).

---

## 7. Trade-off Analysis & Recommendation

### Does 3-Class actually improve the problem?
**Yes.** The 3-class formulation provides a genuine, measurable improvement across every statistical dimension:
1. Macro-F1 increases from ~0.37 to **0.50 – 0.52**.
2. Degrading class F1 increases from ~0.13 to **0.33 – 0.38**.
3. All models now surpass the 47.00% majority accuracy baseline.
4. Overfitting to the arbitrary $-0.35$ threshold is removed.

### What is the trade-off?
The system loses fine-grained separation between "mild degradation" ($-15\%$ to $-35\%$) and "severe degradation" ($< -35\%$).

### Practical Impact on Video Streaming:
Inspection of [`src/adaptation/controller.py`](file:///c:/Users/ELCOT/predictive-video-streaming-network/src/adaptation/controller.py#L5-L12) demonstrates that this trade-off has **zero negative functional impact**:
```python
if buffer_s < 1.5: 
    self.index = 0; reason = "fallback"
elif risk == "High" and confidence >= 0.7 and self.index > 0: 
    self.index -= 1; self.cooldown = 6; reason = "predicted_high_risk"
elif trend in ("Improving", "Stable") and confidence >= 0.7 and buffer_s >= 10: 
    self.evidence += 1
else: 
    self.evidence = 0
```
- The controller only queries `trend in ("Improving", "Stable")` as evidence for up-switching.
- For down-switching or fallback, the controller relies on `buffer_risk == "High"` and `buffer_s < 1.5`.
- Whether network condition was `Degrading` or `Severely Degrading`, both reset up-switch evidence to zero in the exact same manner.

### Recommendation:
The 3-class formulation is statistically sound, reduces label noise, and aligns with the actual controller consumption pattern.
**Status:** In accordance with the prompt constraints, the production model remains **unchanged** for now.
