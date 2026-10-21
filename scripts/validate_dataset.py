from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
def main():
 data=pd.read_csv("data/processed/traces.csv"); split=json.loads(Path("data/processed/split.json").read_text())
 memberships={name:{trace for trace,assigned in split.items() if assigned==name} for name in ("train","validation","test")}
 overlap=any(memberships[left]&memberships[right] for left in memberships for right in memberships if left<right)
 report={"real_sample_count":int(len(data)),"feature_count":int(pd.read_csv("data/features/features.csv").shape[1]) if Path("data/features/features.csv").exists() else 0,"missing_values":data.isna().sum().to_dict(),"duplicates":int(data.duplicated().sum()),"time_range_s":[float(data.timestamp_s.min()),float(data.timestamp_s.max())],"unique_traces_by_source":data.groupby("source").trace_id.nunique().to_dict(),"split_trace_counts":{key:len(value) for key,value in memberships.items()},"zero_trace_overlap":not overlap,"outliers_over_1000_mbps":int((data.throughput_mbps>1000).sum()),"field_availability":{column:int(data[column].notna().sum()) for column in ["throughput_mbps","latency_ms","loss_pct","jitter_ms"]}}
 if Path("data/labels/labels.csv").exists(): report["class_distribution"]=pd.read_csv("data/labels/labels.csv").network_condition.value_counts().to_dict()
 Path("reports").mkdir(exist_ok=True); Path("reports/dataset_validation.json").write_text(json.dumps(report,indent=2)); Path("reports/dataset_validation.md").write_text("# Dataset validation\n\n"+"\n".join(f"- {key}: {value}" for key,value in report.items()))
 if overlap: raise RuntimeError("Trace overlap detected across splits")
 print(json.dumps(report,indent=2))
if __name__=="__main__": main()

