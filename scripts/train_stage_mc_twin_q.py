#!/usr/bin/env python3
"""CUDA-only Stage-aware Twin-Q Monte-Carlo return regression."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import pearsonr,spearmanr

ROOT=Path(__file__).resolve().parents[1]; REPLAY=ROOT/'outputs/diffusion_ql/stage_mc_twin_q_v2/replay'; OUT=ROOT/'outputs/diffusion_ql/stage_mc_twin_q_v2'; GAMMA=.995
class TwinQ(nn.Module):
 def __init__(self):
  super().__init__();
  def n():return nn.Sequential(nn.Linear(55,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1))
  self.q1,self.q2=n(),n()
 def forward(self,s,a):x=torch.cat((s,a),-1);return self.q1(x),self.q2(x)
def mc(rew,done,eid):
 out=np.zeros(len(rew),'f4')
 for e in np.unique(eid):
  ix=np.flatnonzero(eid==e);g=0.
  for i in ix[::-1]:g=float(rew[i])+(0 if done[i] else GAMMA*g);out[i]=g
 return out
def load(name,dev):
 with np.load(REPLAY/f'{name}.npz',allow_pickle=False) as d:
  raw={k:d[k] for k in d.files}; raw['group']=raw['episode_type']; raw['return']=raw['mc_return']
 # Episode identifiers and types are audit metadata, not CUDA tensors.
 return {k:torch.as_tensor(v,device=dev) if k not in ('episode_id','episode_type','group') else v for k,v in raw.items()}
def qaction(raw,am,astd):
 x=raw.clamp(-1,1).clone();x[:,6]=torch.where(x[:,6]<.375,-1.,1.);return (x-am)/astd
def st(x):return {'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max())}
def matched_ranking(data,s,a,q,os):
 """Rank only pairs with same stage/gripper and a local state match."""
 stage=data['obs'][:,43:].argmax(1).cpu().numpy(); raw=data['action'].cpu().numpy(); groups=data['group']; state=data['obs'][:,:43].cpu().numpy(); scale=os[:43].cpu().numpy();rng=np.random.default_rng(20260826);correct=[];bad_count=empty_good=distance_reject=0
 for z in range(5):
  good_all=np.flatnonzero(((groups=='normal_success')|(groups=='recovery_success'))&(stage==z));bad_all=np.flatnonzero((groups=='true_failure')&(stage==z))
  for b in rng.choice(bad_all,min(200,len(bad_all)),False) if len(bad_all) else []:
   bad_count+=1; grip=-1. if raw[b,6]<.375 else 1.; good=good_all[np.where(np.where(raw[good_all,6]<.375,-1.,1.)==grip)[0]]
   if not len(good): empty_good+=1;continue
   dist=np.linalg.norm((state[good]-state[b])/scale,axis=1);nearest=int(np.argmin(dist))
   if dist[nearest]>5.0: distance_reject+=1;continue
   g=int(good[nearest]);sb=s[b:b+1];qb=torch.minimum(*q(sb,a[b:b+1]));qg=torch.minimum(*q(sb,a[g:g+1]));correct.append(qg.item()>qb.item())
 valid=len(correct);coverage=valid/bad_count if bad_count else 0.
 return (float(np.mean(correct)) if valid else float('nan'),{'MATCHED_PAIR_BAD_COUNT':bad_count,'MATCHED_PAIR_EMPTY_GOOD_COUNT':empty_good,'MATCHED_PAIR_DISTANCE_REJECT_COUNT':distance_reject,'MATCHED_PAIR_VALID_COUNT':valid,'MATCHED_PAIR_COVERAGE':coverage})
@torch.no_grad()
def audit(path,data,normal,dev,out):
 p=torch.load(path,map_location=dev,weights_only=False);q=TwinQ().to(dev).eval();q.load_state_dict(p['critic']);om,os,am,astd=normal;s=(data['obs'].float()-om)/os;a=qaction(data['action'].float(),am,astd);q1,q2=q(s,a);v=torch.minimum(q1,q2)[:,0].cpu().numpy();ret=data['return'].cpu().numpy();stage=data['obs'][:,43:].argmax(1).cpu().numpy();gap=(q1-q2).abs()[:,0].cpu().numpy();groups=data['group']
 ranks,pair=matched_ranking(data,s,a,q,os);separation=float(v[groups=='normal_success'].mean())>float(v[groups=='true_failure'].mean()) and float(v[groups=='recovery_success'].mean())>float(v[groups=='true_failure'].mean())
 rank_valid='YES' if pair['MATCHED_PAIR_VALID_COUNT']>=100 and pair['MATCHED_PAIR_COVERAGE']>=.5 and ranks>=.70 else 'INCOMPLETE' if pair['MATCHED_PAIR_VALID_COUNT']<100 or pair['MATCHED_PAIR_COVERAGE']<.5 else 'NO'
 q1a=q1[:,0].cpu().numpy();q2a=q2[:,0].cpu().numpy();r={'step':p['step'],'checkpoint':str(path.resolve()),'Q_RETURN_PEARSON':float(pearsonr(v,ret).statistic),'Q_RETURN_SPEARMAN':float(spearmanr(v,ret).statistic),'Q1_RETURN_PEARSON':float(pearsonr(q1a,ret).statistic),'Q2_RETURN_PEARSON':float(pearsonr(q2a,ret).statistic),'PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY':ranks,'Q_SUCCESS_FAILURE_SEPARATION_VALID':separation,'Q_MATCHED_ACTION_RANKING_VALID':rank_valid,'Q_VALUE_COLLAPSE':bool(v.std()<.05),'Q_SCALE_EXPLOSION':bool(v.std()>10.),'Q1_Q2_DISAGREEMENT_MEAN':float(gap.mean()),'Q1_Q2_DISAGREEMENT_STD':float(gap.std()),'Q1':st(q1a),'Q2':st(q2a),'minQ':st(v),'by_stage':{str(z):st(v[stage==z]) for z in range(5)},'group_means':{x:float(v[groups==x].mean()) for x in ('normal_success','recovery_success','true_failure')},'group_counts':{x:int((groups==x).sum()) for x in ('normal_success','recovery_success','true_failure')},'CUDA_DEVICE':str(dev),**pair}; (out/'audits').mkdir(exist_ok=True);(out/f'audits/mc_audit_step_{p["step"]}.json').write_text(json.dumps(r,indent=2)+'\n');return r
def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is forbidden')
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=5000);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--resume',type=Path);p.add_argument('--audit-only',action='store_true');p.add_argument('--checkpoint',type=Path);a=p.parse_args();dev=torch.device('cuda:0');torch.cuda.set_device(dev);random.seed(20260826);np.random.seed(20260826);torch.manual_seed(20260826);a.output.mkdir(parents=True,exist_ok=True)
 split=json.loads((REPLAY/'split_manifest.json').read_text());
 if any(split['episode_counts'][x]['true_failure']==0 for x in ('val','test')):raise RuntimeError('frozen val/test split lacks true failures')
 train,test=load('train',dev),load('test',dev)
 with np.load(REPLAY/'normalization_stats.npz') as z:normal=tuple(torch.as_tensor(z[k],device=dev) for k in ('observation_mean','observation_std','action_mean','action_std'))
 if a.audit_only:
  if a.checkpoint is None: p.error('--audit-only requires --checkpoint')
  audit(a.checkpoint,test,normal,dev,a.output);return
 om,os,am,astd=normal;s=(train['obs'].float()-om)/os;act=qaction(train['action'].float(),am,astd);target=train['return'].float();q=TwinQ().to(dev);opt=torch.optim.Adam(q.parameters(),lr=3e-4);rng=torch.Generator(device=dev).manual_seed(20260827);audits=[]
 start=0
 if a.resume:
  previous=torch.load(a.resume,map_location=dev,weights_only=False)
  if 'rng_state' not in previous: raise RuntimeError('checkpoint has no rng_state; exact resume is forbidden')
  q.load_state_dict(previous['critic']);opt.load_state_dict(previous['optimizer_critic']);rng.set_state(previous['rng_state']);start=int(previous['step'])
 log=[]
 for step in range(start+1,a.steps+1):
  ix=torch.randint(len(s),(512,),device=dev,generator=rng);q1,q2=q(s[ix],act[ix]);loss=F.mse_loss(q1[:,0],target[ix])+F.mse_loss(q2[:,0],target[ix]);opt.zero_grad();loss.backward();opt.step()
  if step % 100 == 0 or step == a.steps:
   q1_grad=torch.sqrt(sum((x.grad.detach()**2).sum() for x in q.q1.parameters() if x.grad is not None)+1e-12);q2_grad=torch.sqrt(sum((x.grad.detach()**2).sum() for x in q.q2.parameters() if x.grad is not None)+1e-12)
   log.append({'step':step,'train_loss':float(loss),'q1_loss':float(F.mse_loss(q1[:,0],target[ix])),'q2_loss':float(F.mse_loss(q2[:,0],target[ix])),'q1_mean':float(q1.mean()),'q2_mean':float(q2.mean()),'target_return_mean':float(target[ix].mean()),'target_return_std':float(target[ix].std()),'q1_q2_gap':float((q1-q2).abs().mean()),'grad_norm_q1':float(q1_grad),'grad_norm_q2':float(q2_grad),'NaN':bool(not torch.isfinite(loss)),'Inf':bool(torch.isinf(q1).any() or torch.isinf(q2).any()),'GPU_MEMORY_ALLOCATED_MB':float(torch.cuda.memory_allocated(dev)/1024**2)})
  if step in {1000,2500,5000}:
   ck={'step':step,'critic':q.state_dict(),'optimizer_critic':opt.state_dict(),'rng_state':rng.get_state(),'config':{'gamma':GAMMA,'target':'MC_return','reward_version':'V1.2'},'normalization':{k:v.cpu().numpy() for k,v in zip(('observation_mean','observation_std','action_mean','action_std'),normal)}};d=a.output/'checkpoints';d.mkdir(exist_ok=True);path=d/f'mc_step_{step:08d}.pt';torch.save(ck,path);audits.append(audit(path,test,normal,dev,a.output))
 if audits:
  best=max(audits,key=lambda x:(x['Q_RETURN_PEARSON']+x['Q_RETURN_SPEARMAN'],x['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY'],-x['Q1_Q2_DISAGREEMENT_MEAN']));valid=all(x['Q_RETURN_PEARSON']>=.5 and x['Q_RETURN_SPEARMAN']>=.5 and x['Q_MATCHED_ACTION_RANKING_VALID']=='YES' and x['Q_SUCCESS_FAILURE_SEPARATION_VALID'] and not x['Q_VALUE_COLLAPSE'] for x in audits);(a.output/'mc_comparison.json').write_text(json.dumps({'audits':audits,'BEST_MC_CRITIC_STEP':best['step'],'MC_TWIN_Q_VALID':'YES' if valid else 'NO','READY_FOR_TD_BOOTSTRAP_ABLATION':'YES' if valid else 'NO','READY_FOR_PHASE2_DPQL':'NO'},indent=2)+'\n')
 (a.output/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in log)+'\n')
 if a.steps <= 10:
  smoke={'CUDA_TRAINING_VALID':'YES','MC_RETURN_CONSTRUCTION_VALID':json.loads((REPLAY/'replay_audit.json').read_text())['MC_RETURN_CONSTRUCTION_VALID'],'steps':a.steps,'final':log[-1] if log else None}
  (a.output/'smoke_summary.json').write_text(json.dumps(smoke,indent=2)+'\n')
if __name__=='__main__':main()
