# Model selection

Only training traces fit models. Validation macro-F1 selects a model; ties use recall on the risk/degradation class, then log loss, then inference latency. Test traces are protected for the final evaluation command.

