# Frozen labels

All labels use only samples strictly after time *t* for a five-second horizon. Network condition is the future mean throughput divided by the five-second current mean minus one: Improving >= 0.15; Degrading <= -0.15; Severely Degrading <= -0.35; otherwise Stable. Buffer risk uses simulator look-ahead with current bitrate: High if projected buffer < 3 s, Medium if < 8 s, otherwise Low. Recommended quality is the highest ladder bitrate at or below 85% of future mean throughput that also leaves a projected buffer of at least 3 s. Thresholds are frozen in `configs/labels.yaml`.

