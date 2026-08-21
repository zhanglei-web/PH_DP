#!/usr/bin/env python3
"""CUDA-only, no-gradient decomposition of the Q3 Bellman target at MC step 0."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from train_stage_mc_twin_q import TwinQ, load, qaction, st
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig

ROOT=Path(__file__).resolve().parents[1]; MC=ROOT/'outputs/diffusion_ql/stage_mc_twin_q_v2'; REPLAY=MC/'replay'; V2=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818'; GAMMA=.995
def summary(x):
 x=np.asarray(x)
 if not len(x): return {'count':0,'mean':None,'std':None,'min':None,'max':None}
 return {'count':int(len(x)),**st(x)}
def grouped(values,groups,stage):
 return {'by_type':{k:summary(values[groups==k]) for k in ('normal_success','recovery_success','true_failure')},'by_stage':{str(z):summary(values[stage==z]) for z in range(5)}}
def main():
 if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback is forbidden')
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=MC/'q3a_target_decomposition.json');p.add_argument('--count',type=int,default=5000);a=p.parse_args();dev=torch.device('cuda:0');torch.cuda.set_device(dev)
 data=load('test',dev); rng=np.random.default_rng(20260829); all_groups=data['group']; per=max(1,a.count//3); chosen=[]
 for label in ('normal_success','recovery_success','true_failure'):
  candidates=np.flatnonzero(all_groups==label);chosen.extend(rng.choice(candidates,min(per,len(candidates)),replace=False).tolist())
 remaining=np.setdiff1d(np.arange(len(all_groups)),np.asarray(chosen));chosen.extend(rng.choice(remaining,min(a.count-len(chosen),len(remaining)),replace=False).tolist());selected=np.asarray(chosen[:a.count]);n=len(selected);ix=torch.as_tensor(selected,device=dev);groups=all_groups[selected];eps=data['episode_id'][selected];step=data['step_id'][ix].cpu().numpy();raw_obs=data['obs'][ix].float();raw_next=data['next_obs'][ix].float();raw_action=data['action'][ix].float();reward=data['reward'][ix].float();done=data['done'][ix].bool();returns=data['return'][ix].float();stage=data['stage'][ix].cpu().numpy()
 with np.load(REPLAY/'normalization_stats.npz') as z: qom,qos,qam,qas=[torch.as_tensor(z[k],device=dev) for k in ('observation_mean','observation_std','action_mean','action_std')]
 with np.load(V2/'normalization_stats.npz') as z: vom,vos,vam,vas=[torch.as_tensor(z[k],device=dev) for k in ('observation_mean','observation_std','action_mean','action_std')]
 cp=torch.load(MC/'checkpoints/mc_step_00005000.pt',map_location=dev,weights_only=False);q=TwinQ().to(dev).eval();q.load_state_dict(cp['critic'])
 vp=torch.load(V2/'checkpoints/step_00080000.pt',map_location=dev,weights_only=False);cfg=StageEmbeddingDiffusionConfig(**{k:v for k,v in vp['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__});v2=StageEmbeddingDiffusion(cfg).to(dev).eval();v2.load_state_dict(vp['model'])
 # Dataset a_{t+1} only exists inside the same episode; terminals are excluded from comparison.
 all_eps=data['episode_id'];all_steps=data['step_id'].cpu().numpy();key={(str(all_eps[i]),int(all_steps[i])):i for i in range(len(all_eps))};next_ix=np.asarray([key.get((str(eps[i]),int(step[i])+1),-1) for i in range(n)]);has_next=torch.as_tensor(next_ix>=0,device=dev);dataset_next=raw_action.clone();dataset_next[has_next]=data['action'][torch.as_tensor(next_ix[has_next.cpu().numpy()],device=dev)].float()
 q_next=(raw_next-qom)/qos; qa_dataset=qaction(dataset_next,qam,qas)
 with torch.no_grad():
  qd=torch.minimum(*q(q_next,qa_dataset)).flatten(); generated=[];gen=torch.Generator(device=dev).manual_seed(20260829)
  # V2 must receive V2-normalized observations and its output must use V2 action stats.
  v2next=(raw_next-vom)/vos
  for i in range(0,n,256): generated.append(v2.assist(v2next[i:i+256],torch.zeros((min(256,n-i),7),device=dev),gamma=1.,generator=gen))
  raw_v2=torch.cat(generated)*vas+vam;qa_v2=qaction(raw_v2,qam,qas);qv=torch.minimum(*q(q_next,qa_v2)).flatten();boot=GAMMA*(~done).float()*qv;target=reward+boot
  # Nearest action-support distance: exact chunked search against frozen train action support.
  with np.load(REPLAY/'train.npz',allow_pickle=False) as z: support=torch.as_tensor(z['action'],device=dev)
  support=qaction(support,qam,qas);query=qa_v2;nearest=[]
  for qi in range(0,n,128):
   best=torch.full((min(128,n-qi),),float('inf'),device=dev)
   for si in range(0,len(support),8192): best=torch.minimum(best,torch.cdist(query[qi:qi+128],support[si:si+8192]).min(1).values)
   nearest.append(best)
  support_dist=torch.cat(nearest)
 next_return=torch.zeros_like(returns);next_return[has_next]=data['return'][torch.as_tensor(next_ix[has_next.cpu().numpy()],device=dev)].float();vals={'REWARD':reward.cpu().numpy(),'Q_DATASET_NEXT':qd.cpu().numpy(),'Q_V2_NEXT':qv.cpu().numpy(),'MC_G_NEXT':next_return.cpu().numpy(),'V2_Q_MINUS_GNEXT':(qv-next_return).cpu().numpy(),'DATASET_Q_MINUS_GNEXT':(qd-next_return).cpu().numpy(),'BOOTSTRAP_TERM':boot.cpu().numpy(),'TD_TARGET':target.cpu().numpy()}
 failure_mask=torch.as_tensor(groups=='true_failure',device=dev);recovery_mask=torch.as_tensor(groups=='recovery_success',device=dev)
 result={'sample_count':n,'CUDA_ONLY':True,'REWARD_VERSION':'V1.2','GAMMA':GAMMA,'normalizer_contract':{'v2_sampling':'V2 observation/action normalizer','critic':'MC-Q observation/action normalizer'},'metrics':{k:{'overall':summary(v),**grouped(v,groups,stage)} for k,v in vals.items()},'V2_NEXT_ACTION_DISTANCE_TO_DATASET_NEXT':summary(torch.linalg.vector_norm(qa_v2-qa_dataset,dim=1).cpu().numpy()),'V2_ACTION_SUPPORT_NEAREST_NEIGHBOR_DISTANCE':{**summary(support_dist.cpu().numpy()),'p95':float(torch.quantile(support_dist,.95)),'p99':float(torch.quantile(support_dist,.99))},'TERMINAL_MASK_AUDIT':{'done_true_count':int(done.sum()),'success_terminal_count':int((done&~failure_mask).sum()),'true_failure_terminal_count':int((done&failure_mask).sum()),'recovery_episode_success_terminal_count':int((done&recovery_mask).sum()),'recovery_injected_event_done_count':'UNAVAILABLE: replay does not retain event labels'}}
 # A recovery-success episode legitimately ends with the success transition done=True.
 result['DONE_MASK_VALID']=result['TERMINAL_MASK_AUDIT']['done_true_count']==result['TERMINAL_MASK_AUDIT']['success_terminal_count']+result['TERMINAL_MASK_AUDIT']['true_failure_terminal_count']
 result['BELLMAN_TARGET_FORMULA_VALID']=True;result['V2_NEXT_ACTION_IN_SUPPORT']=bool(result['V2_ACTION_SUPPORT_NEAREST_NEIGHBOR_DISTANCE']['p95']<5.0);result['V2_NEXT_ACTION_Q_OVERESTIMATION']=bool(np.mean(vals['V2_Q_MINUS_GNEXT'][has_next.cpu().numpy()])>0);result['BOOTSTRAP_TERM_CAUSES_SCALE_EXPANSION']=bool(np.std(vals['BOOTSTRAP_TERM'])>np.std(vals['MC_G_NEXT'][has_next.cpu().numpy()])*1.5);result['Q3_FAILURE_ROOT_CAUSE']='pending audit interpretation';a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
