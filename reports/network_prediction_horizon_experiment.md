# Network Condition: Prediction Horizon Experiment

## 1. Executive Summary

Following the confirmation that the **3-class target** (`Stable`, `Improving`, `Degrading` with `Degrading` and `Severely Degrading` merged) markedly improves learnability and eliminates artificial boundary noise, this experiment investigates the **temporal lookahead horizon** ($H$) for the network condition target.

Four candidate prediction horizons were systematically evaluated on the validation partition:
- **1 second:** Micro-horizon, near-instantaneous next-second trend.
- **3 seconds:** Short-horizon, matching typical multi-second media segments.
- **5 seconds:** Current standard horizon, matching the project's buffer risk and quality horizon.
- **10 seconds:** Extended lookahead horizon.

### Key Empirical Findings
1. **The 3-second horizon achieves the highest empirical metrics:**
   - **HistGradientBoosting:** **56.66% accuracy** and **0.5277 macro-F1** (highest overall across all horizons).
   - **Random Forest:** **55.89% accuracy** and **0.5183 macro-F1**.
2. **The 5-second horizon performs virtually on par with 3s:**
   - **HistGradientBoosting:** **56.24% accuracy** and **0.5186 macro-F1**.
   - **Random Forest:** **56.11% accuracy** and **0.5164 macro-F1**.
3. **The 1-second horizon suffers from transport-level noise:**
   - Accuracy drops to **53.00% – 53.54%** and majority accuracy drops to 43.00% because instantaneous 1-second packet fluctuations flap unpredictably.
4. **The 10-second horizon degrades due to cellular decorrelation:**
   - Accuracy drops to **53.80% – 54.63%** and macro-F1 falls sharply to **0.4824 – 0.4938** as radio channel autocorrelation collapses ($r \approx 0.20$ at lag 10s).
5. **Trade-off Verdict:** While 3s yields slightly higher raw validation F1 (+0.009 over 5s), the **5-second horizon** remains the optimal choice for the end-to-end system because it directly matches the 4-second chunk duration, controller cooldown (6s), and the 5-second buffer risk horizon without needing multi-horizon synchronization.

---

## 2. Methodology & Leakage Prevention

1. **Past-Only Features:** All features at timestamp $t$ are derived strictly from history up to and including $t$ ($t \le \text{current}$).
2. **Strict Future-Only Labels:** For horizon $H \in \{1, 3, 5, 10\}$, future labels are computed from observations strictly in $[t+1, t+H]$. The future window begins at $t+1$, ensuring $0\%$ overlap with the feature observation window.
3. **Partition Independence:** Traces are split strictly by complete `trace_id` (zero trace leakage between train and validation).
4. **Protected Holdout Untouched:** The 59-trace protected holdout ([`data/processed/final_holdout.json`](file:///c:/Users/ELCOT/predictive-video-streaming-network/data/processed/final_holdout.json)) is excluded from all model training and was **not evaluated**.
5. **Existing Files Unaltered:** Original `data/labels/labels.csv` and all production models in `models/` were kept intact.

---

## 3. Class Distributions Across Horizons

The target uses the 3-class ratio relative to the past 5-second reference throughput:
$$\text{ratio}_H = \frac{\frac{1}{H}\sum_{i=1}^{H} \text{throughput}_{t+i}}{\max\left(0.01, \bar{T}_{\text{past, 5s}}\right)} - 1$$
$\text{Improving} \ (\text{ratio} \ge +0.15)$, $\text{Degrading} \ (\text{ratio} \le -0.15)$, $\text{Stable} \ (\text{otherwise})$.

### Validation Partition (88 Complete Traces)
| Horizon | Total Samples | Stable (%) | Degrading (%) | Improving (%) | Majority Accuracy |
|---|---|---|---|---|---|
| **1s** | 16,606 | 7,140 (**43.00%**) | 4,828 (29.07%) | 4,638 (27.93%) | 43.00% |
| **3s** | 16,430 | 7,749 (**47.16%**) | 4,341 (26.42%) | 4,340 (26.42%) | 47.16% |
| **5s** | 16,254 | 7,640 (**47.00%**) | 4,236 (26.06%) | 4,378 (26.93%) | 47.00% |
| **10s** | 15,814 | 7,245 (**45.81%**) | 4,134 (26.14%) | 4,435 (28.05%) | 45.81% |

### Training Partition (Excluding 59 Protected Holdout Traces)
| Horizon | Total Samples | Stable (%) | Degrading (%) | Improving (%) |
|---|---|---|---|---|
| **1s** | 60,619 | 27,304 (**45.04%**) | 16,853 (27.80%) | 16,462 (27.16%) |
| **3s** | 59,917 | 29,176 (**48.69%**) | 15,159 (25.30%) | 15,582 (26.01%) |
| **5s** | 59,215 | 28,457 (**48.06%**) | 14,853 (25.08%) | 15,905 (26.86%) |
| **10s** | 57,460 | 26,729 (**46.52%**) | 14,526 (25.28%) | 16,205 (28.20%) |

*Note: As horizon increases from 1s to 10s, total available training rows gently decrease (60,619 to 57,460) as the last $H$ seconds of each finite trace cannot form a complete future window.*

---

## 4. Head-to-Head Benchmark Across Horizons (Validation Partition)

| Horizon | Model | Accuracy | Macro-F1 | Precision | Recall | vs. Majority Acc |
|---|---|---|---|---|---|---|
| **1s** | Majority Baseline | 43.00% | 0.2005 | 0.1433 | 0.3333 | — |
| | Logistic (Balanced) | 50.55% | 0.5050 | 0.5029 | 0.5103 | +7.55% |
| | Random Forest (Balanced) | 53.00% | 0.5157 | 0.5269 | 0.5115 | +10.00% |
| | **HistGradientBoosting** | **53.54%** | **0.5167** | **0.5330** | **0.5125** | **+10.54%** |
|---|---|---|---|---|---|---|
| **3s** | Majority Baseline | 47.16% | 0.2137 | 0.1572 | 0.3333 | — |
| | Logistic (Balanced) | 52.37% | 0.5157 | 0.5179 | 0.5346 | +5.21% |
| | Random Forest (Balanced) | 55.89% | 0.5183 | 0.5344 | 0.5186 | +8.73% |
| | **HistGradientBoosting** | **56.66%** | **0.5277** | **0.5449** | **0.5281** | **+9.50%** |
|---|---|---|---|---|---|---|
| **5s** | Majority Baseline | 47.00% | 0.2132 | 0.1567 | 0.3333 | — |
| | Logistic (Balanced) | 51.11% | 0.4958 | 0.5087 | 0.5257 | +4.11% |
| | Random Forest (Balanced) | 56.11% | 0.5164 | 0.5323 | 0.5235 | +9.11% |
| | **HistGradientBoosting** | **56.24%** | **0.5186** | **0.5330** | **0.5272** | **+9.24%** |
|---|---|---|---|---|---|---|
| **10s** | Majority Baseline | 45.81% | 0.2095 | 0.1527 | 0.3333 | — |
| | Logistic (Balanced) | 49.95% | 0.4730 | 0.4941 | 0.5129 | +4.14% |
| | **Random Forest (Balanced)** | **54.63%** | **0.4938** | **0.5092** | **0.5150** | **+8.82%** |
| | HistGradientBoosting | 53.80% | 0.4824 | 0.4953 | 0.5061 | +7.99% |

---

## 5. Per-Class Performance Breakdown (Best Model per Horizon)

### 1s Horizon — HistGradientBoosting
- **Degrading:** Precision: 0.4851, Recall: 0.4078, F1: **0.4431** (Support: 4,828)
- **Improving:** Precision: 0.4770, Recall: 0.5513, F1: **0.5115** (Support: 4,638)
- **Stable:** Precision: 0.6074, Recall: 0.6111, F1: **0.6092** (Support: 7,140)

### 3s Horizon — HistGradientBoosting *(Top F1)*
- **Degrading:** Precision: 0.4777, Recall: 0.3013, F1: **0.3696** (Support: 4,341)
- **Improving:** Precision: 0.5376, Recall: 0.6191, F1: **0.5755** (Support: 4,340)
- **Stable:** Precision: 0.6193, Recall: 0.6638, F1: **0.6408** (Support: 7,749)

### 5s Horizon — HistGradientBoosting *(Top Generalization)*
- **Degrading:** Precision: 0.4583, Recall: 0.2594, F1: **0.3313** (Support: 4,236)
- **Improving:** Precision: 0.5294, Recall: 0.6313, F1: **0.5759** (Support: 4,378)
- **Stable:** Precision: 0.6112, Recall: 0.6908, F1: **0.6486** (Support: 7,640)

### 10s Horizon — Random Forest (Balanced)
- **Degrading:** Precision: 0.4357, Recall: 0.2975, F1: **0.3535** (Support: 4,134)
- **Improving:** Precision: 0.4957, Recall: 0.5894, F1: **0.5385** (Support: 4,435)
- **Stable:** Precision: 0.5962, Recall: 0.6580, F1: **0.6256** (Support: 7,245)

---

## 6. Comprehensive Trade-off Analysis

### 1. Prediction Usefulness
- **1-Second Lookahead:** A 1-second horizon is nearly useless for adaptive video streaming. Video clients request media in discrete chunks (typically 2 to 4 seconds, or 4 seconds in our simulator). A 1-second forecast expires long before a chunk finishes downloading, leaving the player with no advance warning of mid-chunk stalls.
- **10-Second Lookahead:** While theoretically attractive for strategic buffer management, predicting 10 seconds ahead based purely on past throughput is inherently inaccurate (Macro-F1 drops below 0.50). Decisions made on low-confidence 10s predictions risk causing unwarranted downshifts or aggressive upshifts.
- **3s vs. 5s Lookahead:** Both horizons provide immediate, actionable lookahead for chunk-level adaptation. A 3–5 second horizon covers the exact time window required to fetch the next video segment.

### 2. Prediction Difficulty & Signal Decorrelation
- In cellular networks (Norway dataset traces from cars, trains, buses, and ferries), throughput autocorrelation drops steeply with time:
  $$\text{Lag 1s: } 0.674 \implies \text{Lag 3s: } 0.502 \implies \text{Lag 5s: } 0.388 \implies \text{Lag 10s: } 0.202$$
- At 1 second, high autocorrelation is offset by high-frequency TCP slow-start and burst noise (majority accuracy drops to 43%).
- At 3 seconds, the forward average acts as a low-pass filter, smoothing micro-bursts while autocorrelation remains strong ($r \approx 0.50$), producing the highest Macro-F1 (**0.5277**).
- At 5 seconds, predictability remains very solid (**56.24% accuracy**, **0.5186 Macro-F1**), trailing 3s by less than 1 percentage point.
- At 10 seconds, channel conditions have often completely changed due to vehicle mobility and base station handovers; models degrade to $\sim 0.48$ Macro-F1.

### 3. Stability of the Target
- The 1s target is unstable: short-term packet bursts cause the ratio to oscillate violently between `Improving` and `Degrading` tick by tick.
- The 3s and 5s targets aggregate over multi-second windows, producing smooth, coherent trend regimes that prevent oscillatory bitrate switching ("quality flapping").

### 4. Alignment with Video Streaming Architecture
In [`src/adaptation/controller.py`](file:///c:/Users/ELCOT/predictive-video-streaming-network/src/adaptation/controller.py) and [`app/backend/main.py`](file:///c:/Users/ELCOT/predictive-video-streaming-network/app/backend/main.py):
- Video chunk duration is **4.0 seconds**:
  $$\text{download\_time} = \frac{4 \times \text{bitrate}}{\text{throughput}}$$
- Controller hysteresis cooldown is **6 seconds**.
- The `buffer_risk` model evaluates projected buffer drain over a **5-second horizon**.
- Keeping the network condition horizon aligned at **5 seconds** (or 3–5s) ensures that both predictions (`network_condition` and `buffer_risk`) reference the same temporal scale, preventing conflicting multi-scale control signals.

---

## 7. Conclusions & Strategic Recommendation

1. **Empirical Summary:**
   - Both **3s** ($0.5277$ Macro-F1 / $56.66\%$ Acc) and **5s** ($0.5186$ Macro-F1 / $56.24\%$ Acc) form the optimal sweet spot.
   - 1s suffers from instantaneous burst noise; 10s suffers from long-range decorrelation.
2. **Recommendation:**
   - Retain the **5-second horizon** (or select between 3s and 5s) combined with the **3-class formulation** and **HistGradientBoosting / Random Forest**.
   - This maintains seamless architectural alignment with the 5s `buffer_risk` target, requires zero changes to the configuration schema (`labels.horizon_seconds: 5`), and preserves the existing 4-second chunk simulation dynamics.
3. **Production Model Status:**
   - Production models in [`models/`](file:///c:/Users/ELCOT/predictive-video-streaming-network/models/) remain **completely unchanged**.
   - The protected 59-trace holdout remains **untouched and unconsumed**.
