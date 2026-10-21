import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.labels import create_labels
def main():
 state=pd.read_csv("data/processed/player_state.csv"); labels=create_labels(state)
 Path("data/labels").mkdir(parents=True,exist_ok=True); labels.to_csv("data/labels/labels.csv",index=False); print(f"Created {len(labels)} future-only label rows.")
if __name__=="__main__": main()
