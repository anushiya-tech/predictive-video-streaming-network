"""Future-only labels with frozen configuration thresholds."""
from __future__ import annotations
import pandas as pd
def create_labels(frame: pd.DataFrame, horizon: int=5, safety: float=.85) -> pd.DataFrame:
    rows=[]
    for _, trace in frame.sort_values(["trace_id","timestamp_s"]).groupby("trace_id",sort=False):
        trace=trace.reset_index(drop=True)
        for index,row in trace.iterrows():
            future=trace.iloc[index+1:index+1+horizon]
            if len(future) < horizon: continue
            ratio=future.throughput_mbps.mean()/max(.01,trace.iloc[max(0,index-4):index+1].throughput_mbps.mean())-1
            trend="Improving" if ratio >= .15 else "Severely Degrading" if ratio <= -.35 else "Degrading" if ratio <= -.15 else "Stable"
            projected=row.buffer_s-sum(max(0,row.current_bitrate_mbps/max(.05,value)-1) for value in future.throughput_mbps)
            risk="High" if projected < 3 else "Medium" if projected < 8 else "Low"
            feasible=[(name,rate) for name,rate in [("360p",.8),("480p",1.4),("720p",2.8),("1080p",5)] if rate <= future.throughput_mbps.mean()*safety]
            quality=feasible[-1][0] if feasible and projected>=3 else "360p"
            rows.append({"trace_id":row.trace_id,"timestamp_s":row.timestamp_s,"network_condition":trend,"buffer_risk":risk,"recommended_quality":quality})
    return pd.DataFrame(rows)

