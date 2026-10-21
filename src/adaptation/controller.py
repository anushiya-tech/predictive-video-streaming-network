from __future__ import annotations
LADDER=[("360p",.8),("480p",1.4),("720p",2.8),("1080p",5.)]
class PredictiveController:
 def __init__(self): self.index=0; self.cooldown=0; self.evidence=0
 def decide(self, bandwidth:float, buffer_s:float, risk:str="Low", confidence:float=0.0, trend:str="Stable"):
  reason="hold"; self.cooldown=max(0,self.cooldown-1)
  if buffer_s<1.5: self.index=0; reason="fallback"
  elif risk=="High" and confidence>=.7 and self.index>0: self.index-=1; self.cooldown=6; reason="predicted_high_risk"
  elif trend in ("Improving","Stable") and confidence>=.7 and buffer_s>=10: self.evidence+=1
  else: self.evidence=0
  if self.evidence>=3 and self.cooldown==0 and self.index<len(LADDER)-1: self.index+=1; self.cooldown=6; self.evidence=0; reason="evidence_based_up_switch"
  return {"quality":LADDER[self.index][0],"bitrate_mbps":LADDER[self.index][1],"reason":reason}

