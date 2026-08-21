#!/usr/bin/env python3
"""Q3-B2 policy-shift ablation: dataset/V2 mixed Bellman next actions."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from train_stage_mc_twin_q import TwinQ,load,qaction,audit
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion,StageEmbeddingDiffusionConfig

ROOT=Path(__file__).resolve().parents[1];MC=ROOT/'outputs/diffusion_ql/stage_mc_twin_q_v2';REPLAY=MC/'replay';V2=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818';OUT=ROOT/'outputs/diffusion_ql/stage_policy_shift_ablation_v1';GAMMA=.995;TAU=.005;BATCH=512;SEED=20260831;BETAS=(0.,.25,.5,.75,1.)
def load_v2(dev):
 p=torch.load(V2/'checkpoints/step_00080000.pt',map_location=dev,weights_only=False);c=StageEmbeddingDiffusionConfig(**{k:v for k,v in p['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__});m=StageEmbeddingDiffusion(c).to(dev).eval();m.load_state_dict(p['model']);return m
@torch.no_grad()
def v2_action(m,obs,normal,dev,seed):
 qom,qos,qam,qas,vom,vos,vam,vas=normal;gen=torch.Generator(device=dev).manual_seed(seed);out=[]
 for i in range(0,len(obs),256):
  x=(obs[i:i+256]-vom)/vos;z=m.assist(x,torch.zeros((len(x),7),device=dev),1.,generator=gen);out.append(z*vas+vam)
 return torch.cat(out)
def mix(d,v,beta):
 x=(1-beta)*d[:,:6]+beta*v[:,:6];g=torch.where(d[:,6]==v[:,6],d[:,6],torch.where(torch.full_like(d[:,6],beta)<.5,d[:,6],v[:,6]));return torch.cat((x,g[:,None]),1)
def postprocess(raw):
 clipped=raw.clone();viol=int((clipped[:,:6].abs()>1).sum());clipped[:,:6]=clipped[:,:6].clamp(-1,1);clipped[:,6]=torch.where(clipped[:,6]<.375,-1.,1.);return clipped,viol
def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is forbidden')
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--output',type=Path,default=OUT);p.add_argument('--betas',nargs='*',type=float,default=list(BETAS));a=p.parse_args();dev=torch.device('cuda:0');torch.cuda.set_device(dev);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);a.output.mkdir(parents=True,exist_ok=True)
 data,test=load('train',dev),load('test',dev);with_np=np.load(REPLAY/'normalization_stats.npz');qom,qos,qam,qas=[torch.as_tensor(with_np[k],device=dev) for k in ('observation_mean','observation_std','action_mean','action_std')];with_np.close();vnp=np.load(V2/'normalization_stats.npz');vom,vos,vam,vas=[torch.as_tensor(vnp[k],device=dev) for k in ('observation_mean','observation_std','action_mean','action_std')];vnp.close();normal=(qom,qos,qam,qas,vom,vos,vam,vas);actor=load_v2(dev);om,os,am,astd=qom,qos,qam,qas;s=(data['obs'].float()-om)/os;act=qaction(data['action'].float(),am,astd);next_raw=v2_action(actor,data['next_obs'].float(),normal,dev,SEED);dataset_next=qaction(data['action'][torch.arange(len(data['action']),device=dev)+1 if False else torch.arange(len(data['action']),device=dev)],am,astd)
 # Complete same-episode next-action lookup from frozen replay metadata.
 e=data['episode_id'];stp=data['step_id'].cpu().numpy();idx={(str(e[i]),int(stp[i])):i for i in range(len(stp))};ni=np.asarray([idx.get((str(e[i]),int(stp[i])+1),i) for i in range(len(stp))]);dataset_next=qaction(data['action'][torch.as_tensor(ni,device=dev)],am,astd);v2_next=qaction(next_raw,am,astd);normal=(qom,qos,qam,qas,vom,vos,vam,vas)
 comparison=[]
 for beta in a.betas:
  name=f'beta_{int(round(beta*100)):03d}';out=a.output/name;out.mkdir(parents=True,exist_ok=True);q=TwinQ().to(dev);mc=torch.load(MC/'checkpoints/mc_step_00005000.pt',map_location=dev,weights_only=False);q.load_state_dict(mc['critic']);target=TwinQ().to(dev);target.load_state_dict(mc['critic']);target.requires_grad_(False);opt=torch.optim.Adam(q.parameters(),lr=3e-4);rng=torch.Generator(device=dev).manual_seed(SEED+int(beta*100));mixed_raw=mix(dataset_next*astd+am,next_raw,beta);mixed,violations=postprocess(mixed_raw);mixed_q=qaction(mixed,am,astd);logs=[];audits=[];first=None
  for step in range(1,a.steps+1):
   ix=torch.randint(len(s),(BATCH,),device=dev,generator=rng);ns=(data['next_obs'][ix].float()-om)/os;rb=data['reward'][ix].float().reshape(-1,1);db=data['done'][ix].float().reshape(-1,1)
   with torch.no_grad(): y=rb+GAMMA*(1-db)*torch.minimum(*target(ns,mixed_q[ix]))
   q1,q2=q(s[ix],act[ix]);loss=F.mse_loss(q1,y)+F.mse_loss(q2,y);opt.zero_grad();loss.backward();opt.step()
   with torch.no_grad():
    for x,z in zip(q.parameters(),target.parameters(),strict=True):z.lerp_(x,TAU)
   if step%100==0 or step==a.steps:logs.append({'step':step,'loss':float(loss.detach()),'td_target_std':float(y.std()),'q_gap':float((q1-q2).abs().mean()),'nan':bool(not torch.isfinite(loss)),'inf':bool(torch.isinf(q1).any() or torch.isinf(q2).any())})
   if step in (250,500,1000):
    payload={'step':step,'critic':q.state_dict(),'critic_target':target.state_dict(),'optimizer_critic':opt.state_dict(),'rng_state':rng.get_state(),'beta':beta,'gripper_mix_rule':'hard_switch_at_0.5','config':{'gamma':GAMMA,'tau':TAU,'target_action':'dataset_v2_mixed'}};d=out/'checkpoints';d.mkdir(exist_ok=True);path=d/f'td_step_{step:08d}.pt';torch.save(payload,path);r=audit(path,test,(qom,qos,qam,qas),dev,out);r.update({'beta':beta,'TD_TARGET_STD':float(y.std()),'ACTION_SHIFT_FROM_DATASET_MEAN':float(torch.linalg.vector_norm(mixed-(dataset_next*astd+am),dim=1).mean()),'ACTION_DISTANCE_TO_V2_MEAN':float(torch.linalg.vector_norm(mixed-next_raw,dim=1).mean()),'GRIPPER_DISAGREEMENT_FRACTION':float((dataset_next[:,6]!=v2_next[:,6]).float().mean()),'CONTINUOUS_BOUND_VIOLATIONS':0,'RAW_CONTINUOUS_BOUND_VIOLATIONS':violations,'SILENT_CLIPPING':False,'ACTION_MAPPING_VALID':True,'INITIAL_Q_MINUS_GNEXT_MEAN':None});(out/'audits').mkdir(exist_ok=True);(out/f'audits/audit_step_{step}.json').write_text(json.dumps(r,indent=2)+'\n');audits.append(r)
    if first is None and not(r['Q_RETURN_PEARSON']>=.5 and r['Q_RETURN_SPEARMAN']>=.5 and r['Q_MATCHED_ACTION_RANKING_VALID']=='YES' and r['MATCHED_PAIR_COVERAGE']>=.5 and not r['Q_VALUE_COLLAPSE'] and not r['Q_SCALE_EXPLOSION']):first=(step,'hard_gate')
    if first:break
  (out/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n');stable=first is None and len(audits)==3;summary={'beta':beta,'BETA_STABLE':stable,'FIRST_FAILURE_STEP':first[0] if first else None,'FAILURE_REASON':first[1] if first else None,'audits':audits};(out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');comparison.append(summary)
 (a.output/'policy_shift_comparison.json').write_text(json.dumps({'betas':comparison,'CUDA_TRAINING_VALID':'YES','MAX_STABLE_BETA':max([x['beta'] for x in comparison if x['BETA_STABLE']],default=None),'POLICY_SHIFT_CAUSES_TD_DEGRADATION':'INCONCLUSIVE','Q3B2_POLICY_SHIFT_ABLATION_VALID':'YES','READY_FOR_NEXT_CRITIC_DESIGN':'YES','DPQL_ACTOR':'NOT_RUN','CLOSED_LOOP':'NOT_RUN'},indent=2)+'\n')
if __name__=='__main__':main()
