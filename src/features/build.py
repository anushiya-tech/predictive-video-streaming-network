"""Past-only temporal feature construction shared by offline and live paths."""
from __future__ import annotations
import numpy as np, pandas as pd
def build_features(frame: pd.DataFrame, windows: tuple[int,...]=(1,3,5,10)) -> pd.DataFrame:
    ordered = frame.sort_values(["trace_id","timestamp_s"]).copy()
    if "split" not in ordered:
        ordered["split"] = ""
    result=ordered[["trace_id","timestamp_s","split","is_synthetic","buffer_s","current_bitrate_mbps","switch_count","time_since_switch_s"]].copy()
    grouped=ordered.groupby("trace_id", sort=False)["throughput_mbps"]
    for window in windows:
        rolling=grouped.rolling(window, min_periods=1)
        result[f"throughput_mean_{window}"]=rolling.mean().reset_index(level=0,drop=True)
        result[f"throughput_std_{window}"]=rolling.std(ddof=0).reset_index(level=0,drop=True)
        result[f"throughput_min_{window}"]=rolling.min().reset_index(level=0,drop=True)
        result[f"throughput_max_{window}"]=rolling.max().reset_index(level=0,drop=True)
        first=grouped.shift(window-1).fillna(ordered["throughput_mbps"])
        result[f"throughput_slope_{window}"]=(ordered["throughput_mbps"]-first)/max(1,window-1)
        result[f"throughput_pct_change_{window}"]=(ordered["throughput_mbps"]-first)/first.clip(lower=.001)
    result["buffer_depletion_rate"]=-ordered.groupby("trace_id")["buffer_s"].diff().fillna(0)
    result["bandwidth_margin"]=ordered["throughput_mbps"]/ordered["current_bitrate_mbps"].clip(lower=.01)
    result["buffer_safety_margin"]=ordered["buffer_s"]-10
    result["stability_score"]=1/(1+result["throughput_std_5"])
    return result.replace([np.inf,-np.inf],np.nan).fillna(0)
