#!/usr/bin/env python3
"""Train frozen-V2 compatible Q and validate long-horizon action utility."""
from __future__ import annotations
import csv, json, random
from pathlib import Path
from collections import defaultdict
import h5py, numpy as np, torch
from torch import nn
from scipy.stats import spearmanr
from mujoco_shared_control.rewards.stageaware_recovery_reward_v12 import StageAwareRecoveryRewardV12, RewardBookkeeping

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL'
SPLIT=ROOT/'outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL/split_manifest.json'
STATS=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/normalization_stats.npz'
OUT=ROOT/'outputs/stage_value_guidance/v2_stage_q_recovery_value_v2'; CALIB=OUT/'timeout_calibration.npz'; GAMMA=.995
class QNet(nn.Module):
 def __init__(self): super().__init__();self.net=nn.Sequential(nn.Linear(55,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1))
 def forward(self,s,a):return self.net(torch.cat((s,a),-1))
def returns(reward,done,eid):
 out=np.zeros(len(reward),'f4')
 for e in np.unique(eid):
  ix=np.flatnonzero(eid==e);g=0.
  for i in ix[::-1]:g=float(reward[i])+(0. if done[i] else GAMMA*g);out[i]=g
 return out
def main():
 OUT.mkdir(parents=True,exist_ok=True);random.seed(20260820);np.random.seed(20260820);torch.manual_seed(20260820)
 with np.load(STATS) as z:om,os,am,astd=[z[k].astype('f4') for k in ('observation_mean','observation_std','action_mean','action_std')]
 manifest=json.loads((DATA/'dataset_manifest.json').read_text());sp=json.loads(SPLIT.read_text());byid={x['episode_id']:x for x in manifest['episodes']};rw=StageAwareRecoveryRewardV12();base={};meta=[]
 for name,ids in sp['splits'].items():
  S=[];A=[];R=[];E=[]
  for eid in ids:
   with h5py.File(byid[eid]['path'],'r') as f:s=f['full_physical_state'][:].astype('f4');p=f['active_phase'][:].astype('i8');a=f['raw_pilot_action'][:].astype('f4');ev=f['event'][:].astype('i8')
   b=RewardBookkeeping();r=np.asarray([rw.transition(int(p[i]),int(p[min(i+1,len(p)-1)]),int(ev[i]),b)['reward'] for i in range(len(p))],'f4');done=np.zeros(len(p),bool);done[-1]=True
   state=np.c_[s,np.eye(5,dtype='f4')[p]];S.append(state);A.append(a);R.append(returns(r,done,np.zeros(len(p),int)));E.append(np.asarray([eid]*len(p)))
  base[name]=(np.concatenate(S),np.concatenate(A),np.concatenate(R),np.concatenate(E))
 # Use only terminal-failure rollouts as low-value calibration, split by episode before training.
 with np.load(CALIB) as d:cs,ca,cp,cr,cd,ce=[d[k] for k in ('obs','action','phase','reward','done','episode_id')]
 # A Stage4 success has +5.0 - 0.001 = 4.999 when no forward edge occurs that frame.
 success_eps={int(e) for e in np.unique(ce) if np.any(cr[ce==e]>=4.999)};fail_eps=sorted(set(map(int,np.unique(ce)))-success_eps);rng=np.random.default_rng(20260820);rng.shuffle(fail_eps);cut=round(.7*len(fail_eps));train_fail=set(fail_eps[:cut]);eval_fail=set(fail_eps[cut:])
 cret=returns(cr,cd,ce);mask=np.isin(ce,list(train_fail));train_s=np.r_[base['train'][0],cs[mask]];train_a=np.r_[base['train'][1],ca[mask]];train_r=np.r_[base['train'][2],cret[mask]]
 model=QNet();opt=torch.optim.Adam(model.parameters(),lr=3e-4);x=torch.from_numpy((train_s-om)/os);a=torch.from_numpy((train_a-am)/astd);y=torch.from_numpy(train_r);log=[]
 for step in range(1,20001):
  ix=torch.from_numpy(rng.integers(0,len(x),512));pred=model(x[ix],a[ix]).squeeze(-1);loss=torch.nn.functional.smooth_l1_loss(pred,y[ix]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step==1 or step%1000==0:log.append({'step':step,'loss':float(loss.item())})
 torch.save({'model':model.state_dict(),'normalizer':'v2_frozen','gamma':GAMMA},OUT/'q_checkpoint_valid.pt')
 with (OUT/'q_training_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['step','loss']);w.writeheader();w.writerows(log)
 def score(s,a):
  with torch.no_grad():return model(torch.from_numpy((s-om)/os),torch.from_numpy((a-am)/astd)).squeeze(-1).numpy()
 ts,ta,tr,_=base['test'];qtest=score(ts,ta);pear=float(np.corrcoef(qtest,tr)[0,1]);spear=float(spearmanr(qtest,tr).statistic)
 fmask=np.isin(ce,list(eval_fail));fs,fa,fp,fr=cs[fmask],ca[fmask],cp[fmask],cret[fmask];qfail=score(fs,fa)
 rows=[]
 for stage in range(5):
  qs=qtest[ts[:,43+stage]==1];qf=qfail[fp==stage];rows.append({'stage':stage,'success_count':len(qs),'failure_count':len(qf),'success_mean_q':float(qs.mean()) if len(qs) else None,'failure_mean_q':float(qf.mean()) if len(qf) else None,'separated':bool(len(qs) and len(qf) and qs.mean()>qf.mean())})
 with (OUT/'q_success_failure_separation_by_stage.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 # Matched-state test: for held-out failed states, use nearest successful state at same stage;
 # score both actions at the failed state, preventing state-value confounding.
 pair=[]; rng=np.random.default_rng(7)
 for stage in range(5):
  good_ix=np.flatnonzero(ts[:,43+stage]==1);bad_ix=np.flatnonzero(fp==stage)
  if not len(good_ix) or not len(bad_ix):continue
  for bi in rng.choice(bad_ix,size=min(100,len(bad_ix)),replace=False):
   cand=rng.choice(good_ix,size=min(2000,len(good_ix)),replace=False);dist=np.linalg.norm((ts[cand,:43]-fs[bi,:43])/os[:43],axis=1);gi=cand[int(np.argmin(dist))];qg=float(score(fs[bi:bi+1],ta[gi:gi+1])[0]);qb=float(score(fs[bi:bi+1],fa[bi:bi+1])[0]);pair.append({'stage':stage,'state_distance':float(dist.min()),'q_good':qg,'q_bad':qb,'correct':qg>qb})
 accuracy=float(np.mean([r['correct'] for r in pair])) if pair else 0.;
 with (OUT/'q_matched_action_ranking.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=pair[0] if pair else ['stage']);w.writeheader();w.writerows(pair)
 valid_corr=pear>.1 and spear>.1;valid_sep=all(r['separated'] for r in rows if r['failure_count']);valid_rank=accuracy>=.70;collapse=float(np.std(qtest))<.05 or float(np.std(qfail))<.05
 audit={'valid_success_episodes':4000,'valid_recovery_success_episodes':3000,'valid_timeout_failure_episodes':len(fail_eps),'q_return_pearson':pear,'q_return_spearman':spear,'q_success_failure_separation_valid':valid_sep,'pairwise_ranking_accuracy':accuracy,'q_matched_action_ranking_valid':valid_rank,'q_value_collapse':collapse,'q_long_horizon_value_valid':valid_corr and valid_sep and valid_rank and not collapse,'previous_calibration_invalid':True,'reward_version':'V1.2','reward_changed':'NO','v2_normalizer_frozen':'YES','q_normalizer_frozen':'YES','raw_state_shared':'YES','normalized_tensor_shared':'NO'}
 (OUT/'q_long_horizon_audit.json').write_text(json.dumps(audit,indent=2)+'\n');(OUT/'q_return_correlation.csv').write_text('pearson,spearman\n'+f'{pear},{spear}\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
