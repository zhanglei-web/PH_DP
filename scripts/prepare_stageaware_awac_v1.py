#!/usr/bin/env python3
"""Build 48D current-active-stage replay and audit exact BC actor warm start."""
from __future__ import annotations
import json,csv,os
from pathlib import Path
import h5py,numpy as np,torch
from mujoco_shared_control.awac.hybrid import HybridAWACConfig,HybridActor
from mujoco_shared_control.experts.recovery_bc import RecoveryBCPolicy
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL';REWARD=ROOT/'outputs/offline_awac/stageaware_reward_v1_2_4000_20260818T_REWARD_V12_FORMAL';BC=ROOT/'outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL';OUT=Path(os.environ.get('STAGEAWARE_AWAC_OUT',ROOT/'outputs/offline_awac/stageaware_awac_v1_4000/run_20260818T_STAGEAWARE_AWAC_V1_FORMAL'))
def main():
 OUT.mkdir(parents=True,exist_ok=True);sp=json.loads((BC/'split_manifest.json').read_text());rd={x['episode_id']:x for x in csv.DictReader((REWARD/'episode_reward_decomposition.csv').open())};calc=[]
 # Reward V1.2 is reconstructed only from its frozen formula; no HDF5 mutation.
 from mujoco_shared_control.rewards.stageaware_recovery_reward_v12 import StageAwareRecoveryRewardV12,RewardBookkeeping
 rw=StageAwareRecoveryRewardV12();parts={}
 for split in ('train','validation','test'):
  d={k:[] for k in ('obs','next_obs','continuous_action','gripper_action','reward','terminated','truncated','episode_id')}
  for eid in sp['splits'][split]:
   with h5py.File(sp['episode_paths'][eid],'r') as f:s=f['full_physical_state'][:];n=f['next_full_physical_state'][:];p=f['active_phase'][:].astype(int);a=f['raw_pilot_action'][:];e=f['event'][:].astype(int)
   b=RewardBookkeeping()
   for i in range(len(p)):
    q=rw.transition(p[i],p[i+1] if i+1<len(p) else p[i],e[i],b);d['obs'].append(np.r_[s[i],np.eye(5)[p[i]]]);d['next_obs'].append(np.r_[n[i],np.eye(5)[p[i+1] if i+1<len(p) else p[i]]]);d['continuous_action'].append(a[i,:6]);d['gripper_action'].append(float(a[i,6]<.375));d['reward'].append(q['reward']);d['terminated'].append(q['done']);d['truncated'].append(False);d['episode_id'].append(eid)
  arr={k:np.asarray(v) for k,v in d.items()};np.savez_compressed(OUT/f'{split}.npz',**arr);parts[split]=len(arr['reward'])
 train=np.load(OUT/'train.npz');mean=train['obs'][:,:43].mean(0).astype('f');std=np.maximum(train['obs'][:,:43].std(0),1e-6).astype('f');mean=np.r_[mean,np.zeros(5,'f')];std=np.r_[std,np.ones(5,'f')];np.savez(OUT/'normalizer.npz',mean=mean,std=std)
 bc=RecoveryBCPolicy(48);z=torch.load(BC/'best_val.pt',map_location='cpu',weights_only=False);bc.load_state_dict(z['model']);cfg=HybridAWACConfig(observation_dim=48,hidden_dims=(256,256,256),gamma=.995,awac_updates=20000);actor=HybridActor(cfg);state=actor.state_dict();loaded=[]
 for key in list(state):
  source={'continuous_mean':'motion_head','gripper_logit':'gripper_head'}.get(key.split('.')[0],key.split('.')[0]);bk=key.replace(key.split('.')[0],source,1)
  if bk in bc.state_dict() and state[key].shape==bc.state_dict()[bk].shape:state[key]=(-bc.state_dict()[bk] if key.startswith('gripper_logit') else bc.state_dict()[bk]);loaded.append(key)
 actor.load_state_dict(state);x=torch.from_numpy(np.asarray(np.load(OUT/'validation.npz')['obs'][:1000],'f'));norm=(x-torch.from_numpy(mean))/torch.from_numpy(std);m,l=bc(norm);am,ag,prob=actor.deterministic_action(norm);act=np.where(ag.numpy().squeeze(1)>.5,-.25,1.);bact=np.where(l.detach().numpy()>=0,1.,-.25);audit={'status':'PASS' if np.allclose(m.detach().numpy(),am.detach().numpy(),atol=1e-6) and np.array_equal(act,bact) else 'FAIL','loaded_parameters':loaded,'missing_bc_parameters':[k for k in bc.state_dict() if k not in [x.replace('continuous_mean','motion_head').replace('gripper_logit','gripper_head') for x in loaded]],'motion_mae':float(np.mean(abs(m.detach().numpy()-am.detach().numpy()))),'motion_mse':float(np.mean((m.detach().numpy()-am.detach().numpy())**2)),'motion_max_abs':float(np.max(abs(m.detach().numpy()-am.detach().numpy()))),'gripper_agreement':float(np.mean(act==bact)),'states':len(x)};torch.save({'actor':actor.state_dict(),'training_config':cfg.__dict__,'observation_mean':mean,'observation_std':std,'state_mode':'physical43_active_stage5','observation_dim':48,'physical_dim':43,'auxiliary_dim':5},OUT/'actor_step0.pt');(OUT/'actor_initialization_audit.json').write_text(json.dumps(audit,indent=2)+'\n');(OUT/'state48_audit.json').write_text(json.dumps({'status':'PASS','definition':'normalized physical43 + raw current active_stage onehot5','stage_normalized':False,'train_only_physical_normalizer':True,'counts':parts},indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
