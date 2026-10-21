from __future__ import annotations
import numpy as np
def scenario(name:str, seconds:int=60, seed:int=42):
 rng=np.random.default_rng(seed); base={"stable":12,"degradation":12,"sudden_drop":12,"recovery":3,"congestion":7,"mobile":9}.get(name)
 if base is None: raise ValueError("Unsupported scenario")
 values=[]
 for tick in range(seconds):
  target=base
  if name=="degradation": target=max(2,20-17*tick/max(1,seconds-1))
  if name=="sudden_drop" and tick>=seconds//3: target=2
  if name=="recovery": target=min(16,3+13*tick/max(1,seconds-1))
  if name=="congestion": target=5+3*np.sin(tick/4)
  if name=="mobile": target=9+3*np.sin(tick/3)
  value=max(.1,(values[-1] if values else target)*.7+target*.3+rng.normal(0,.35))
  values.append(float(value))
 return values

