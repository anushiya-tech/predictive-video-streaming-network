import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ml.train import train_all
def main():
 report=train_all(); figure,axis=plt.subplots(figsize=(10,5)); axis.bar(report["Model"],report["Macro F1"]); axis.tick_params(axis="x",rotation=90); axis.set_ylabel("Validation macro-F1"); figure.tight_layout(); figure.savefig("reports/model_comparison.png"); print(report.to_string(index=False))
if __name__=="__main__": main()
