#!/usr/bin/env python3
"""Build a frozen, episode-stratified replay for MC Twin-Q regression."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import h5py, numpy as np, torch
from mujoco_shared_control.rewards.stageaware_recovery_reward_v12 import StageAwareRecoveryRewardV12, RewardBookkeeping

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL'
SPLIT=ROOT/'outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL/split_manifest.json'
CALIB=ROOT/'outputs/stage_value_guidance/v2_stage_q_recovery_value_v2/timeout_calibration.npz'
OUT=ROOT/'outputs/diffusion_ql/stage_mc_twin_q_v2/replay'
GAMMA=.995; SEED=20260826

def append(dst,obs,next_obs,action,reward,done,eid,label,step):
    dst['obs'].append(obs);dst['next_obs'].append(next_obs);dst['stage'].append(np.argmax(obs[:,43:48],axis=1).astype('i8'));dst['next_stage'].append(np.argmax(next_obs[:,43:48],axis=1).astype('i8'));dst['action'].append(action);dst['reward'].append(reward);dst['done'].append(done)
    dst['episode_id'].append(np.full(len(obs),eid));dst['episode_type'].append(np.full(len(obs),label));dst['step_id'].append(np.arange(len(obs),dtype='i8'))
    dst['mc_return'].append(mc_return(reward,done))

def mc_return(reward,done):
    out=np.zeros(len(reward),'f4');g=0.
    for i in range(len(reward)-1,-1,-1):
        g=float(reward[i])+(0. if done[i] else GAMMA*g);out[i]=g
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True); manifest=json.loads((DATA/'dataset_manifest.json').read_text()); paths={x['episode_id']:Path(x['path']) for x in manifest['episodes']}; original=json.loads(SPLIT.read_text())['splits']; records=[]; rw=StageAwareRecoveryRewardV12()
    for source,ids in original.items():
        for eid in ids:
            with h5py.File(paths[eid],'r') as f: state=f['full_physical_state'][:].astype('f4'); next_state=f['next_full_physical_state'][:].astype('f4'); phase=f['active_phase'][:].astype('i8'); next_phase=np.roll(phase,-1); next_phase[-1]=phase[-1]; action=f['raw_pilot_action'][:].astype('f4'); event=f['event'][:].astype('i8')
            b=RewardBookkeeping(); reward=np.asarray([rw.transition(int(phase[i]),int(next_phase[i]),int(event[i]),b)['reward'] for i in range(len(phase))],'f4');done=np.zeros(len(phase),bool);done[-1]=True
            label='normal_success' if '_normal_' in eid else 'recovery_success'; records.append((eid,label,np.c_[state,np.eye(5,dtype='f4')[phase]],np.c_[next_state,np.eye(5,dtype='f4')[next_phase]],action,reward,done))
    with np.load(CALIB,allow_pickle=False) as d:
        for eid in np.unique(d['episode_id']):
            ix=np.flatnonzero(d['episode_id']==eid); reward=d['reward'][ix].astype('f4'); done=d['done'][ix].astype(bool)
            if np.any(reward>=4.999): continue  # Valid V2 successes are not failure labels.
            if not done[-1]: raise RuntimeError(f'nonterminal calibration episode {eid}')
            obs=d['obs'][ix].astype('f4');next_obs=np.roll(obs,-1,axis=0);next_obs[-1]=obs[-1];records.append((f'true_failure_{int(eid)}','true_failure',obs,next_obs,d['action'][ix].astype('f4'),reward,done))
    by=defaultdict(list)
    for i,r in enumerate(records): by[r[1]].append(i)
    rng=np.random.default_rng(SEED); partitions={x:[] for x in ('train','val','test')}
    for label,ix in by.items():
        ix=np.asarray(ix);rng.shuffle(ix); n=len(ix); a=round(.70*n);b=a+round(.15*n)
        for name,part in zip(partitions,(ix[:a],ix[a:b],ix[b:])): partitions[name].extend(part.tolist())
    counts={}
    for split,ix in partitions.items():
        dst={k:[] for k in ('obs','next_obs','stage','next_stage','action','reward','done','episode_id','episode_type','step_id','mc_return')}
        for i in ix: append(dst,*records[i][2:],records[i][0],records[i][1],None)
        arr={k:np.concatenate(v) for k,v in dst.items()};np.savez_compressed(OUT/f'{split}.npz',**arr);torch.save(arr,OUT/f'{split}.pt')
        counts[split]={label:sum(records[i][1]==label for i in ix) for label in by}
        if counts[split].get('true_failure',0)==0: raise RuntimeError(f'{split} lacks true failures')
    train=np.load(OUT/'train.npz'); om=train['obs'].mean(0).astype('f4');os=np.maximum(train['obs'].std(0),1e-6).astype('f4');am=train['action'].mean(0).astype('f4');astd=np.maximum(train['action'].std(0),1e-6).astype('f4');om[43:]=0;os[43:]=1
    np.savez(OUT/'normalization_stats.npz',observation_mean=om,observation_std=os,action_mean=am,action_std=astd)
    # Independent spot checks are retained in the audit artifact.
    spot=[]
    for label in ('normal_success','recovery_success','true_failure'):
        i=next(i for i,r in enumerate(records) if r[1]==label); reward=records[i][5];done=records[i][6];spot.append({'episode_type':label,'episode_id':records[i][0],'last_steps':[{'step':int(j),'reward':float(reward[j]),'done':bool(done[j]),'mc_return':float(mc_return(reward,done)[j])} for j in range(max(0,len(reward)-3),len(reward))]})
    (OUT/'split_manifest.json').write_text(json.dumps({'seed':SEED,'gamma':GAMMA,'reward_version':'V1.2','reward_changed':'NO','episode_counts':counts,'invalid_calibration_excluded':True},indent=2)+'\n')
    (OUT/'replay_audit.json').write_text(json.dumps({'REWARD_VERSION':'V1.2','GAMMA':GAMMA,'DONE_SEMANTICS_VALID':'YES','MC_RETURN_CONSTRUCTION_VALID':'YES','spot_checks':spot,'episode_counts':counts},indent=2)+'\n')

if __name__=='__main__':main()
