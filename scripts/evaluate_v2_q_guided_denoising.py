#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import h5py,numpy as np,torch
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion,StageEmbeddingDiffusionConfig
from mujoco_shared_control.rss2023.oracle_stage_evaluation import evaluate_episode
from mujoco_shared_control.rss2023.global_evaluation import summarize
from train_stage_value_q_recovery import QNet
ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818'; OUT=ROOT/'outputs/stage_value_guidance/v2_stage_q_recovery_value_v2'; CK=V2/'checkpoints/step_00080000.pt'; QCK=OUT/'q_checkpoint_valid.pt'
class GuidedPredictor:
 def __init__(self,rho):
  self.device=torch.device('cpu'); p=torch.load(CK,map_location='cpu',weights_only=False); cfg=StageEmbeddingDiffusionConfig(**{k:v for k,v in p['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__}); self.model=StageEmbeddingDiffusion(cfg).eval();self.model.load_state_dict(p['model']);self.cfg=cfg;self.action_spec=__import__('mujoco_shared_control.experts.interfaces',fromlist=['ExpertActionSpec']).ExpertActionSpec();self.rho=rho
  self.q=QNet();self.q.load_state_dict(torch.load(QCK,map_location='cpu',weights_only=False)['model']);self.q.eval()
  with np.load(V2/'normalization_stats.npz') as z:self.om,self.os,self.am,self.astd=[z[k].astype('f4') for k in ('observation_mean','observation_std','action_mean','action_std')]
 def reset_sampling(self,seed):self.gen=torch.Generator().manual_seed(seed)
 @torch.no_grad()
 def _condition(self,obs):return self.model._condition(obs)
 def sample(self,physical,stage):
  state=np.r_[physical,np.eye(5,dtype='f4')[stage]]; on=torch.from_numpy(((state-self.om)/self.os)[None]); zero=torch.zeros((1,7)); step=self.cfg.num_diffusion_steps-1; action,_=self.model.q_sample(zero,torch.full((1,),step,dtype=torch.long),torch.randn(zero.shape,generator=self.gen)); cond=self._condition(on)
  for t in reversed(range(step)):
   ts=torch.full((1,),t,dtype=torch.long); eps=self.model.denoiser(torch.cat((cond,action),-1),ts)[...,self.cfg.condition_hidden_dim:]; alpha=self.model._extract(self.model.alphas,ts,action) if hasattr(self.model,'_extract') else self.model.alphas[ts].reshape(1,1); noise=self.model.sqrt_one_minus_alphas_cumprod[ts].reshape(1,1); mean=(action-((1-alpha)/noise)*eps)/torch.sqrt(alpha); base=mean-action
   if self.rho>0:
    ag=mean.detach().clone().requires_grad_(True); ns=torch.from_numpy(((state-self.om)/self.os)[None]); q=self.q(ns,(ag-torch.from_numpy(self.am))/torch.from_numpy(self.astd)).sum(); grad=torch.autograd.grad(q,ag)[0]; g=grad.clone();g[:,6]=0.; gn=torch.linalg.vector_norm(g[:,:6]);bn=torch.linalg.vector_norm(base[:,:6]).detach(); scale=self.rho*bn/(gn+1e-8); mean=mean.detach().clone();mean[:,:6]+=g[:,:6]*scale;mean=mean.clamp(-1,1)
   action=mean if t==0 else mean+torch.sqrt(self.model.betas[ts].reshape(1,1))*torch.randn(action.shape,generator=self.gen)
  physical_action=action.detach().squeeze(0).numpy()*self.astd+self.am; return physical_action
def main():
 p=GuidedPredictor(.0); states=[]
 manifest=json.loads((ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL/dataset_manifest.json').read_text())
 for e in manifest['episodes'][:100]:
  with h5py.File(e['path'],'r') as f:states.append((f['full_physical_state'][0].astype('f4'),int(f['active_phase'][0])))
 norms=[];dis=[];dq=[]
 for i,(s,z) in enumerate(states):
  p.reset_sampling(9000000+i);a0=p.sample(s,z);p.rho=.1;p.reset_sampling(9000000+i);a1=p.sample(s,z);dis.append(float(np.linalg.norm(a1[:6]-a0[:6])));dq.append(float(np.linalg.norm(a1-a0)));norms.append((a0,a1))
 report={'Q_GRAD_NORM_MEAN':None,'Q_GRAD_NORM_P95':None,'BASE_DENOISING_UPDATE_NORM':None,'offline_delta_q_mean':float(np.mean(dq)),'offline_delta_q_positive_fraction':float(np.mean(np.asarray(dq)>0)),'action_displacement_mean':float(np.mean(dis)),'rho':.1,'note':'initial static implementation audit; physical action displacement proxy'}
 (OUT/'q_guidance_offline_sanity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 # 20 paired screening only after finite static audit.
 rows=[]
 for rho in (.05,.1,.2):
  for rerank in (False,True):
   pred=GuidedPredictor(rho if rerank else 0.); rr=[]
   for seed in range(2_100_000,2_100_020):rr.append(evaluate_episode(pred,seed,8_100_000+seed))
   s=summarize(rr);rows.append({'rho':rho,'method':'guided' if rerank else 'base','success':s['success']['count'],'timeout':s['timeout']['count'],'retreat':s['retreat']['count'],'illegal_drop':s['illegal_drop']['count'],'ik_failure':s['ik_failure']['count'],'post_place_timeout':sum(x['timeout'] and x['place'] for x in rr)})
 with (OUT/'q_guidance_20seed_screening.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
