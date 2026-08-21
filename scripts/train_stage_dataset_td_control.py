#!/usr/bin/env python3
"""Q3-B1: Bellman Twin-Q with recorded dataset next actions only."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from train_stage_mc_twin_q import TwinQ,load,qaction,audit

ROOT=Path(__file__).resolve().parents[1];MC=ROOT/'outputs/diffusion_ql/stage_mc_twin_q_v2';REPLAY=MC/'replay';OUT=ROOT/'outputs/diffusion_ql/stage_dataset_td_control_v1';GAMMA=.995;TAU=.005;BATCH=512;SEED=20260830;POINTS=(250,500,1000)
def next_alignment(data):
 e=data['episode_id'];s=data['step_id'].cpu().numpy();idx={(str(e[i]),int(s[i])):i for i in range(len(s))};nxt=np.asarray([idx.get((str(e[i]),int(s[i])+1),-1) for i in range(len(s))]);non=~data['done'].cpu().numpy().astype(bool);return nxt, bool(np.all((nxt[non]>=0)&(e[nxt[non]]==e[non]))), bool(np.all(s[nxt[non]]==s[non]+1))
def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is forbidden')
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();dev=torch.device('cuda:0');torch.cuda.set_device(dev);a.output.mkdir(parents=True,exist_ok=True);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 data,test=load('train',dev),load('test',dev);nxt,ep_ok,step_ok=next_alignment(data);nxt_t=torch.as_tensor(nxt,device=dev);(a.output/'alignment_audit.json').write_text(json.dumps({'NEXT_ACTION_EPISODE_ALIGNMENT_VALID':ep_ok,'NEXT_ACTION_STEP_ALIGNMENT_VALID':step_ok,'nonterminal_count':int((~data['done'].bool()).sum())},indent=2)+'\n')
 if not ep_ok or not step_ok:raise RuntimeError('dataset next-action alignment failed')
 with np.load(REPLAY/'normalization_stats.npz') as z:normal=tuple(torch.as_tensor(z[k],device=dev) for k in ('observation_mean','observation_std','action_mean','action_std'))
 om,os,am,astd=normal;s=(data['obs'].float()-om)/os;act=qaction(data['action'].float(),am,astd);next_act=qaction(data['action'][nxt_t.clamp_min(0)].float(),am,astd);rew=data['reward'].float().reshape(-1,1);done=data['done'].float().reshape(-1,1)
 cp=torch.load(MC/'checkpoints/mc_step_00005000.pt',map_location=dev,weights_only=False);q=TwinQ().to(dev);q.load_state_dict(cp['critic']);target=TwinQ().to(dev);target.load_state_dict(cp['critic']);target.requires_grad_(False);opt=torch.optim.Adam(q.parameters(),lr=3e-4);rng=torch.Generator(device=dev).manual_seed(SEED+1);logs=[];audits=[]
 for step in range(1,a.steps+1):
  ix=torch.randint(len(s),(BATCH,),device=dev,generator=rng);sb=s[ix];ns=(data['next_obs'][ix].float()-om)/os;ab=act[ix];nab=next_act[ix];rb=rew[ix];db=done[ix]
  with torch.no_grad():y=rb+GAMMA*(1-db)*torch.minimum(*target(ns,nab))
  q1,q2=q(sb,ab);l1=F.mse_loss(q1,y);l2=F.mse_loss(q2,y);loss=l1+l2;opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():
   for src,dst in zip(q.parameters(),target.parameters(),strict=True):dst.lerp_(src,TAU)
  if step%100==0 or step==a.steps:logs.append({'step':step,'critic_loss':float(loss.detach()),'td_target_mean':float(y.mean()),'td_target_std':float(y.std()),'q_gap':float((q1-q2).abs().mean()),'nan':bool(not torch.isfinite(loss)),'inf':bool(torch.isinf(q1).any() or torch.isinf(q2).any()),'gpu_memory_mb':float(torch.cuda.memory_allocated(dev)/1024**2)})
  if step in POINTS:
   payload={'step':step,'critic':q.state_dict(),'critic_target':target.state_dict(),'optimizer_critic':opt.state_dict(),'rng_state':rng.get_state(),'config':{'gamma':GAMMA,'tau':TAU,'target_action':'dataset_next_action','reward_version':'V1.2'},'normalization':{k:v.cpu().numpy() for k,v in zip(('observation_mean','observation_std','action_mean','action_std'),normal)}};d=a.output/'checkpoints';d.mkdir(exist_ok=True);path=d/f'td_dataset_step_{step:08d}.pt';torch.save(payload,path);r=audit(path,test,normal,dev,a.output);r.update({'TD_TARGET_MEAN':float(y.mean()),'TD_TARGET_STD':float(y.std()),'TD_TARGET_MIN':float(y.min()),'TD_TARGET_MAX':float(y.max()),'DELTA_PEARSON_FROM_MC':r['Q_RETURN_PEARSON']-.9562298059,'DELTA_SPEARMAN_FROM_MC':r['Q_RETURN_SPEARMAN']-.6658250887,'DELTA_RANKING_FROM_MC':r['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY']-.9977578475,'TD_TARGET_STD_RATIO_TO_MC':float(y.std()/1.333941361),'Q_STD_RATIO_TO_MC':float(r['minQ']['std']/1.236063361),'Q1Q2_GAP_RATIO_TO_MC':float(r['Q1_Q2_DISAGREEMENT_MEAN']/.0397039875)});(a.output/'audits').mkdir(exist_ok=True);(a.output/f'audits/audit_step_{step}.json').write_text(json.dumps(r,indent=2)+'\n');audits.append(r)
   if not(r['Q_RETURN_PEARSON']>=.5 and r['Q_RETURN_SPEARMAN']>=.5 and r['Q_MATCHED_ACTION_RANKING_VALID']=='YES' and r['MATCHED_PAIR_COVERAGE']>=.5 and not r['Q_VALUE_COLLAPSE'] and not r['Q_SCALE_EXPLOSION']):break
 (a.output/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n');valid=bool(audits) and all(x['Q_RETURN_PEARSON']>=.5 and x['Q_RETURN_SPEARMAN']>=.5 and x['Q_MATCHED_ACTION_RANKING_VALID']=='YES' and x['MATCHED_PAIR_COVERAGE']>=.5 and not x['Q_VALUE_COLLAPSE'] and not x['Q_SCALE_EXPLOSION'] for x in audits);best=max(audits,key=lambda x:(x['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY'],x['Q_RETURN_SPEARMAN'],x['Q_RETURN_PEARSON']));(a.output/'q3b1_comparison.json').write_text(json.dumps({'audits':audits,'CUDA_TRAINING_VALID':'YES','NEXT_ACTION_EPISODE_ALIGNMENT_VALID':ep_ok,'NEXT_ACTION_STEP_ALIGNMENT_VALID':step_ok,'DATASET_TD_STABLE':'YES' if valid else 'NO','V2_TD_STABLE':'NO','TARGET_ACTION_SHIFT_CONFIRMED_AS_ROOT_CAUSE':'YES' if valid else 'NO','Q3B1_DATASET_TD_VALID':'YES' if valid else 'NO','BEST_DATASET_TD_STEP':best['step'] if valid else 'NOT_SELECTED','READY_FOR_Q3B2_POLICY_SHIFT_ABLATION':'YES' if valid else 'NO','DPQL_ACTOR':'NOT_RUN','CLOSED_LOOP':'NOT_RUN'},indent=2)+'\n')
if __name__=='__main__':main()
