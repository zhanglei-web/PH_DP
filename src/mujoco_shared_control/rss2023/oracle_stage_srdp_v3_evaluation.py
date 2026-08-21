"""Closed-loop V3 evaluation using the frozen Oracle current-stage source."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.oracle_stage_srdp_v3_model import SRDPStageDiffusion,SRDPStageDiffusionConfig
from mujoco_shared_control.rss2023.oracle_stage_evaluation import evaluate_episode
from mujoco_shared_control.rss2023.global_evaluation import summarize
class SRDPStagePredictor:
 def __init__(self,checkpoint,normalization,*,device_name='auto'):
  self.device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name);p=torch.load(Path(checkpoint),map_location=self.device,weights_only=False);keys=SRDPStageDiffusionConfig.__dataclass_fields__;cfg=SRDPStageDiffusionConfig(**{k:v for k,v in p['diffusion_config'].items() if k in keys});self.model=SRDPStageDiffusion(cfg).to(self.device).eval();self.model.load_state_dict(p['model'])
  with np.load(normalization,allow_pickle=False) as s:self.observation_mean=np.asarray(s['observation_mean'],np.float32);self.observation_std=np.asarray(s['observation_std'],np.float32);self.action_mean=np.asarray(s['action_mean'],np.float32);self.action_std=np.asarray(s['action_std'],np.float32)
  if self.observation_mean.shape!=(48,) or not np.array_equal(self.observation_mean[43:],np.zeros(5,np.float32)) or not np.array_equal(self.observation_std[43:],np.ones(5,np.float32)):raise ValueError('invalid V3 normalization')
  self.action_spec=ExpertActionSpec();self.generator=None
 def reset_sampling(self,seed):self.generator=torch.Generator(device=self.device).manual_seed(seed)
 @torch.inference_mode()
 def sample(self,observation_43,active_stage):
  x=np.concatenate((observation_43.astype(np.float32),np.eye(5,dtype=np.float32)[active_stage]));x=torch.from_numpy((x-self.observation_mean)/self.observation_std).to(self.device).unsqueeze(0);a=self.model.assist(x,torch.zeros((1,7),device=self.device),1.,generator=self.generator).squeeze(0).cpu().numpy();return np.asarray(a*self.action_std+self.action_mean,np.float64)
def run_evaluation(checkpoint,normalization,output,*,formal_seeds=range(2_000_000,2_000_100),device='auto'):
 output.mkdir(parents=True,exist_ok=True);p=SRDPStagePredictor(checkpoint,normalization,device_name=device);rows=[]
 for i,seed in enumerate(formal_seeds):rows.append(evaluate_episode(p,seed,8_000_000+seed));print(f'formal {i+1}/{len(formal_seeds)}',flush=True)
 r={'policy':'Oracle Stage SRDP-style V3','checkpoint':str(Path(checkpoint).resolve()),'normalization':str(Path(normalization).resolve()),'policy_observation':'physical43_plus_stage_condition32','policy_uses_phase':True,'policy_uses_milestones':False,'future_stage_leakage':False,'summary':summarize(rows),'rows':rows};(output/'evaluation_report.json').write_text(json.dumps(r,indent=2)+'\n');return r
