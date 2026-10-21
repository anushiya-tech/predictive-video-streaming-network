"""Freeze a new whole-trace holdout before renewed experiments."""
import hashlib,json
from pathlib import Path
import pandas as pd
data=pd.read_csv("data/processed/traces.csv")
eligible=sorted(data.loc[data.split=="train","trace_id"].unique())
holdout=eligible[::7]
payload={"seed":42,"source_partition":"previous_train_only","trace_ids":holdout}
payload["sha256"]=hashlib.sha256("\n".join(holdout).encode()).hexdigest()
Path("data/processed/final_holdout.json").write_text(json.dumps(payload,indent=2))
print(f"Frozen {len(holdout)} traces; sha256={payload['sha256']}")

