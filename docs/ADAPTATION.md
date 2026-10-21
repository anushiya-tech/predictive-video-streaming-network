# Predictive adaptation policy

The runtime computes the same temporal feature frame used offline, loads the saved trend and buffer-risk models, and passes their predictions and calibrated model probabilities to `PredictiveController`. A critical-buffer fallback selects 360p. A High-risk prediction above the controller confidence threshold steps down one rung; up-switches require three confident Stable/Improving observations, adequate buffer, and no cooldown. The simulation and controller decisions are labelled SIMULATED / PREDICTED in the API.

