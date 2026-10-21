import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.streaming.simulator import add_player_state
from src.features.build import build_features
def main():
 data=pd.read_csv("data/processed/traces.csv"); state=add_player_state(data)
 state.to_csv("data/processed/player_state.csv",index=False); features=build_features(state)
 Path("data/features").mkdir(parents=True,exist_ok=True); features.to_csv("data/features/features.csv",index=False); print(f"Created {len(features)} past-only feature rows.")
if __name__=="__main__": main()
