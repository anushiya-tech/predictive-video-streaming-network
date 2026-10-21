# Experiment log

## Renewed protected evaluation

The former test split was already reported and is retained in `reports/before_test_metrics.json`. A new 59-trace holdout was frozen in `data/processed/final_holdout.json` before retraining. Models now exclude those traces during fitting; the forced evaluator run replaces `final_test_metrics.json` with the newly defined final evaluation.

## Phase 1 validation

- Dataset: 101,851 real one-second rows, 587 full traces; train/validation/test trace counts are 410/88/89.
- Frozen horizon: 5 s, from configs/labels.yaml.
- Models compared: majority, logistic regression, random forest, HistGradientBoosting.
- Selection criterion: validation macro-F1, documented in docs/MODEL_SELECTION.md.
- Network condition: best validation accuracy 0.5453 and macro-F1 0.3781 (logistic regression selected by macro-F1). The target is missed. Likely causes are trace volatility, broad future-trend classes, and lack of latency/loss/jitter in this real throughput-only source. Thresholds were not changed.
- Buffer risk: best validation accuracy 0.9055 and macro-F1 0.8929 (random forest selected). The macro-F1 target is narrowly missed; no test set was used.
- Recommended quality: best validation macro-F1 0.7969 (HistGradientBoosting).
