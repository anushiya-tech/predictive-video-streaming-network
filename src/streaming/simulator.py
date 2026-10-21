"""Chunk-level player-state simulator driven by real network traces."""
from __future__ import annotations
import numpy as np, pandas as pd
LADDER = [( "360p", .8), ("480p", 1.4), ("720p", 2.8), ("1080p", 5.0)]
def add_player_state(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed); output = frame.sort_values(["trace_id", "timestamp_s"]).copy()
    states = []
    for _, trace in output.groupby("trace_id", sort=False):
        buffer_s, bitrate, last_switch, switches = 4., .8, 0, 0
        policy = ["fixed", "throughput", "buffer", "epsilon"][int(rng.integers(4))]
        for position, row in enumerate(trace.itertuples(index=False)):
            throughput = max(float(row.throughput_mbps), .05)
            if policy == "throughput": candidate = max((rate for _, rate in LADDER if rate <= throughput * .8), default=.8)
            elif policy == "buffer": candidate = .8 if buffer_s < 5 else (2.8 if buffer_s < 12 else 5.0)
            elif policy == "epsilon" and rng.random() < .15: candidate = LADDER[int(rng.integers(4))][1]
            else: candidate = bitrate
            switched = candidate != bitrate; switches += int(switched); last_switch = 0 if switched else last_switch + 1; bitrate = candidate
            download_s = 4 * bitrate / throughput; stall_s = max(0., download_s - buffer_s)
            buffer_s = min(30., max(0., buffer_s - download_s) + 4)
            states.append((buffer_s, bitrate, next(name for name, rate in LADDER if rate == bitrate), switches, last_switch, policy, stall_s))
    output[["buffer_s","current_bitrate_mbps","resolution","switch_count","time_since_switch_s","behavior_policy","stall_s"]] = states
    return output
