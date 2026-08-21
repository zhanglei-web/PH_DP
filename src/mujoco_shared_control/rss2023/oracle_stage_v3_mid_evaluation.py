from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.oracle_stage_v3_mid_model import V3MidConfig,V3MidDiffusion
from mujoco_shared_control.rss2023.oracle_stage_evaluation import evaluate_episode
from mujoco_shared_control.rss2023.global_evaluation import summarize
class V3MidPredictor:
 def __init__(self,checkpoint,normalization,device_name='auto'):
  self.device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name);p=torch.load(Path(checkpoint),map_location=self.device,weights_only=False);c=V3MidConfig();self.model=V3MidDiffusion(c).to(self.device).eval();self.model.load_state_dict(p['model']);
  with np.load(normalization) as s:self.om=s['observation_mean'];self.os=s['observation_std'];self.am=s['action_mean'];self.ast=s['action_std']
  self.action_spec=ExpertActionSpec();self.generator=None
 def reset_sampling(self,seed):self.generator=torch.Generator(device=self.device).manual_seed(seed)
 def sample(self,o,stage):
  x=np.r_[o.astype('f4'),np.eye(5,dtype='f4')[stage]];x=torch.from_numpy((x-self.om)/self.os).to(self.device).unsqueeze(0);a=self.model.assist(x,torch.zeros((1,7),device=self.device),1.,generator=self.generator).squeeze(0).cpu().numpy();return np.asarray(a*self.ast+self.am,np.float64)
def run_evaluation(checkpoint,normalization,output,formal_seeds=range(2000000,2000100),device='auto'):
 output.mkdir(parents=True,exist_ok=True);p=V3MidPredictor(checkpoint,normalization,device);rows=[]
 for i,s in enumerate(formal_seeds):
  rows.append(evaluate_episode(p,s,8000000+s)); print(f'eval {i+1}/{len(formal_seeds)}',flush=True)
 r={'policy':'Oracle Stage V3-Mid','checkpoint':str(Path(checkpoint).resolve()),'summary':summarize(rows),'rows':rows};(output/'evaluation_report.json').write_text(json.dumps(r,indent=2)+'\n');return r
