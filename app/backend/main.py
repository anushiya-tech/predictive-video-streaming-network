from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from src.simulation.runtime import scenario
from src.adaptation.controller import PredictiveController
from src.evaluation.baselines import run
from src.streaming.simulator import add_player_state
from src.features.build import build_features
from src.ml.predict import predict
import pandas as pd
app=FastAPI(title="Predictive Video Streaming Network"); history=[]; controller=PredictiveController(); active=False
@app.get("/api/network/current")
def current(): return history[-1] if history else {"error":"simulation not started"}
@app.get("/api/network/history")
def network_history(): return history
@app.get("/api/prediction/current")
def prediction(): return history[-1].get("prediction",{}) if history else {"error":"simulation not started"}
@app.get("/api/baselines")
def baselines(): return run()
@app.post("/api/simulation/scenario")
def start(payload:dict):
 global history,active,controller
 name=payload.get("scenario","stable")
 try: values=scenario(name,60)
 except ValueError as e: raise HTTPException(422,str(e))
 history=[]; controller=PredictiveController(); buffer=4.
 for tick,bw in enumerate(values):
  observed=values[:tick+1]
  frame=pd.DataFrame({"trace_id":"live","source":"simulated","timestamp_s":range(len(observed)),"throughput_mbps":observed,"latency_ms":float("nan"),"loss_pct":float("nan"),"jitter_ms":float("nan"),"is_synthetic":True,"split":""})
  state=add_player_state(frame); state.loc[state.index[-1],"buffer_s"]=buffer
  feature=build_features(state).iloc[-1].to_dict()
  try:
   trend_output=predict("network_condition",feature); risk_output=predict("buffer_risk",feature)
  except (ValueError,FileNotFoundError) as error: raise HTTPException(503,f"Model inference unavailable: {error}")
  trend=trend_output["prediction"]; risk=risk_output["prediction"]; confidence=float(risk_output["confidence"])
  decision = controller.decide(bw, buffer, risk, confidence, trend)
  download = 4.0 * decision["bitrate_mbps"] / max(0.1, bw)
  stall = max(0.0, download - buffer)
  buffer = min(30.0, max(0.0, buffer - download) + 4.0)
  history.append({
      "tick": tick,
      "bandwidth_mbps": round(bw, 3),
      "measurement_type": "SIMULATED",
      "buffer_s": round(buffer, 3),
      "stall_s": round(stall, 3),
      "quality": decision["quality"],
      "bitrate_mbps": decision["bitrate_mbps"],
      "prediction": {
          "network_condition": trend,
          "network_confidence": round(float(trend_output["confidence"]), 4),
          "buffer_risk": risk,
          "buffer_confidence": round(confidence, 4),
          "horizon_seconds": 5,
          "type": "PREDICTED",
      },
      "decision": decision,
  })
 active = True
 return {"scenario": name, "ticks": len(history)}

@app.get("/api/simulation/scenarios")
def get_scenarios():
 return ["stable", "sudden_drop", "recovery", "degradation", "congestion", "mobile"]

@app.post("/api/simulation/stop")
def stop():
 global active
 active = False
 return {"active": False}

@app.get("/api/video/status")
def video():
 return {"ffmpeg_available": False, "source": "simulated representations", "qualities": ["360p", "480p", "720p", "1080p"]}

app.mount("/", StaticFiles(directory="app/frontend", html=True), name="frontend")

