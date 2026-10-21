# Final verification

- `python scripts/run_baselines.py`: exit 0.
- `python scripts/evaluate_models.py --final`: exit 0; protected final metrics saved in `reports/final_test_metrics.json`.
- `pytest -q`: exit 0; 6 tests passed.
- `python -c "from app.backend.main import app"`: exit 0.

Final test metrics: network condition accuracy 0.4528/macro-F1 0.3901; buffer risk accuracy 0.9104/macro-F1 0.8960; recommended quality accuracy 0.9588/macro-F1 0.6845. The stated 95% and 0.90 macro-F1 targets were not met for network trend or buffer risk. Trace volatility and absent latency/loss/jitter fields are material limitations. Baseline results are generated in `reports/baseline_comparison.csv`.

