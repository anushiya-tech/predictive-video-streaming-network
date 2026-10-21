# Limitations

- Real source traces provide throughput only; latency, loss, and jitter remain missing rather than fabricated.
- ffmpeg is unavailable, so the representation ladder and buffer reaction are simulated, not HLS/DASH playback.
- The dashboard is local-only and its network observations are simulated scenarios.
- The existing trend model is weak; new holdout comparison is not yet performed and the old test result is a before-comparison only.
- No CDN, multi-user load, authentication, cloud deployment, or production video source is implemented.

