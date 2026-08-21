#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import h5py, numpy as np
from mujoco_shared_control.rewards.stageaware_recovery_reward_v12 import StageAwareRecoveryRewardV12, RewardBookkeeping

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL'
OUT=ROOT/'outputs/diffusion_ql/stage_diffusion_ql_v1_20260819/replay'

def main():
 manifest=json.loads((ROOT/'outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL/split_manifest.json').read_text()); paths={r['episode_id']:Path(r['path']) for r in json.loads((DATA/'dataset_manifest.json').read_text())['episodes']}
 OUT.mkdir(parents=True,exist_ok=True); reward=StageAwareRecoveryRewardV12()
 for split,ids in manifest['splits'].items():
  obs=[]; nxt=[]; act=[]; rew=[]; done=[]; eids=[]
  for eid in ids:
   with h5py.File(paths[eid],'r') as f:
    s=f['full_physical_state'][:].astype('f4'); ns=f['next_full_physical_state'][:].astype('f4'); phase=f['active_phase'][:].astype('i8'); raw=f['raw_pilot_action'][:].astype('f4'); events=f['event'][:].astype('i8')
   book=RewardBookkeeping()
   for i in range(len(phase)):
    j=min(i+1,len(phase)-1); q=reward.transition(int(phase[i]),int(phase[j]),int(events[i]),book)
    obs.append(np.r_[s[i],np.eye(5,dtype='f4')[phase[i]]]); nxt.append(np.r_[ns[i],np.eye(5,dtype='f4')[phase[j]]]); act.append(raw[i]); rew.append(q['reward']); done.append(q['done']); eids.append(eid)
  np.savez_compressed(OUT/f'{split}.npz',obs=np.asarray(obs,'f4'),next_obs=np.asarray(nxt,'f4'),action=np.asarray(act,'f4'),reward=np.asarray(rew,'f4'),done=np.asarray(done,bool),episode_id=np.asarray(eids))
 train=np.load(OUT/'train.npz'); om=train['obs'].mean(0).astype('f4'); os=np.maximum(train['obs'].std(0),1e-6).astype('f4'); am=train['action'].mean(0).astype('f4'); astd=np.maximum(train['action'].std(0),1e-6).astype('f4'); om[43:]=0.; os[43:]=1.
 np.savez(OUT/'normalization_stats.npz',observation_mean=om,observation_std=os,action_mean=am,action_std=astd)
 (OUT/'audit.json').write_text(json.dumps({'status':'PASS','state_mode':'physical43_active_stage5','action_mode':'raw_pilot_action_7','reward_version':'V1.2','gamma':.995,'splits':{k:int(len(np.load(OUT/f'{k}.npz')['reward'])) for k in manifest['splits']}},indent=2)+'\n')
if __name__=='__main__': main()
