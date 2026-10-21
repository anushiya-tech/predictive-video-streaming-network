"""Clean and trace-split real dataset files."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import load_mahimahi_trace
def main() -> None:
    raw=Path("data/raw"); files=[path for path in raw.rglob("*") if path.is_file() and path.suffix.lower() in {".txt",".log",".csv"}]
    frames=[]
    for path in files:
        try: frames.append(load_mahimahi_trace(path, "real_world_traces"))
        except (ValueError,UnicodeDecodeError): continue
    if not frames: raise RuntimeError("No real trace files found. Run download_datasets.py or follow its manual instructions.")
    data=pd.concat(frames,ignore_index=True); data=data[(data.throughput_mbps>=0)&(data.throughput_mbps<=1000)].drop_duplicates()
    data["timestamp_s"]=data.groupby("trace_id").timestamp_s.transform(lambda series: (series-series.min()).round())
    data=data.groupby(["trace_id","timestamp_s"],as_index=False).agg({"source":"first","throughput_mbps":"mean","latency_ms":"first","loss_pct":"first","jitter_ms":"first","is_synthetic":"first"})
    traces=sorted(data.trace_id.unique()); train_end=int(len(traces)*.7); val_end=int(len(traces)*.85)
    mapping={trace:("train" if index<train_end else "validation" if index<val_end else "test") for index,trace in enumerate(traces)}
    data["split"]=data.trace_id.map(mapping); Path("data/processed").mkdir(parents=True,exist_ok=True); data.to_csv("data/processed/traces.csv",index=False)
    Path("data/processed/split.json").write_text(json.dumps(mapping,indent=2))
    print(f"Prepared {len(data)} real rows from {len(traces)} traces.")
if __name__ == "__main__": main()
