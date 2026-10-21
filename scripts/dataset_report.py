import json
from pathlib import Path
import pandas as pd
data=pd.read_csv("data/processed/traces.csv")
counts=data.groupby("trace_id").size()
report={"observations":len(data),"traces":int(counts.size),"observations_per_trace":{"min":int(counts.min()),"max":int(counts.max()),"mean":float(counts.mean()),"median":float(counts.median())},"throughput_mbps":data.throughput_mbps.describe().to_dict(),"missing":data.isna().sum().to_dict(),"duplicates":int(data.duplicated().sum()),"sources":data.groupby("source").trace_id.nunique().to_dict(),"sampling_interval_s":float(data.groupby("trace_id").timestamp_s.diff().dropna().median())}
Path("reports/dataset_report.json").write_text(json.dumps(report,indent=2))
Path("reports/dataset_report.md").write_text("# Dataset report\n\n"+"\n".join(f"- {k}: {v}" for k,v in report.items()))
print(json.dumps(report,indent=2))

