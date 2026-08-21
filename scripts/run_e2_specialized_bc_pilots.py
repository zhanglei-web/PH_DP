#!/usr/bin/env python3
"""E2: four specialized learned BC pilots, with NoAssist evaluation only."""
from __future__ import annotations

import argparse, csv, hashlib, json, pickle, random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

import build_e2_valid_failure_snapshot_bank as bank
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_bc import RecoveryBCPolicy
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot

ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/'outputs/stage_dataset/stage_dataset_v1_hysteresis_20260817T052000Z'
V2=ROOT/'outputs/experiments/e2_failure_snapshot_bank_v2/run_20260818T030000Z'
OUT=ROOT/'outputs/experiments/e2_specialized_bc_pilots'
V2_SHA='d06c1b95b821ab797ef83506c4bfec952d861313e4cab68d19017878a545496f'
SCENARIOS=(('BC-Normal','NORMAL','bc_normal'),('BC-Grasp','GRASP_RECOVERY','bc_grasp'),('BC-Transport','TRANSPORT_DROP','bc_transport'),('BC-Place','PLACE_RECOVERY','bc_place'))
FAILURE_FOR_SCENARIO={'GRASP_RECOVERY':'GRASP_FAILURE','TRANSPORT_DROP':'TRANSPORT_EARLY','PLACE_RECOVERY':'PLACE_FAILURE'}
DT=.05; MAX=700; IKMAX=5; SEED=20260818

@dataclass(frozen=True)
class Config:
    learning_rate:float=3e-4; batch_size:int=1024; weight_decay:float=1e-4; max_epochs:int=50; early_stopping_patience:int=5; seed:int=SEED

def dump(p:Path,x:Any): p.write_text(json.dumps(x,indent=2)+'\n')
def csv_write(p:Path,rows:list[dict[str,Any]]):
    keys=sorted({k for r in rows for k in r});
    with p.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()

class Episodes(Dataset):
    def __init__(self, paths:dict[str,str], ids:list[str], scenario:str):
        self.ids=list(ids); self.scenario=scenario; ss=[];aa=[]; self.lengths={}
        for eid in self.ids:
            with h5py.File(paths[eid],'r') as f:
                if str(f.attrs['trajectory_type'])!=scenario or not bool(f.attrs['valid']) or not bool(f.attrs['final_success']): raise ValueError(f'ineligible episode {eid}')
                s=f['full_physical_state'][:].astype(np.float32); a=f['raw_pilot_action'][:].astype(np.float32)
                if s.shape!=(len(s),43) or a.shape!=(len(s),7) or not np.isfinite(s).all() or not np.isfinite(a).all() or not np.all(np.isin(a[:,6],[-.25,1.])): raise ValueError(f'schema violation {eid}')
                ss.append(s);aa.append(a);self.lengths[eid]=len(s)
        self.state=np.concatenate(ss);self.action=np.concatenate(aa)
    def __len__(self):return len(self.state)
    def __getitem__(self,i):return self.state[i],self.action[i]

def stats(root:Path, formal:list[dict[str,Any]], rejected:list[dict[str,Any]])->dict[str,Any]:
    rows=[]; segments=[]
    for name,scenario,_ in SCENARIOS:
        r=[x for x in formal if x['trajectory_type']==scenario]; success=sum(bool(x['final_success']) for x in r); usable=sum(bool(x['final_success']) and bool(x['valid']) and (scenario=='NORMAL' or bool(x['regression_found'])) for x in r)
        rows.append({'BC':name,'Scenario':scenario,'Total Episodes':len(r),'Final Success':success,'Final Failure':len(r)-success,'Success Rate':success/len(r),'Usable Successful Demonstrations':usable})
        for x in r:
            with h5py.File(x['path'],'r') as f: phase=f['active_phase'][:].astype(int); event=f['event'][:].astype(int); length=len(phase)
            if scenario=='NORMAL': pre=length; post=0; regression=-1
            else:
                old={'GRASP_RECOVERY':1,'TRANSPORT_DROP':2,'PLACE_RECOVERY':3}[scenario]; hits=np.flatnonzero((phase[:-1]==old)&(phase[1:]==0))+1
                if len(hits)!=1: raise ValueError(f'regression audit failed {x["episode_id"]}')
                regression=int(hits[0]);pre=regression;post=length-regression
            segments.append({'Scenario':scenario,'Episode ID':x['episode_id'],'Total Transitions':length,'Failure/Regression Step':regression,'Pre Failure/Regression Transitions':pre,'Post Regression to Final Success Transitions':post,'Success Event Count':int(np.sum(event==4))})
    csv_write(root/'dataset_audit'/'dataset_outcome_audit.csv',rows);csv_write(root/'dataset_audit'/'recovery_segment_statistics.csv',segments)
    return {'rows':rows,'segments':segments,'invalid_attempts':len(rejected)}

def norm(data:Episodes):
    mean=data.state.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(data.state.std(0,dtype=np.float64),1e-6).astype(np.float32);mean[42]=0.;std[42]=1.;return mean,std
def loss(model,state,action,mean,std):
    motion,logit=model((state-mean)/std);ml=nn.functional.mse_loss(motion,action[:,:6]);gl=nn.functional.binary_cross_entropy_with_logits(logit,(action[:,6]>0).float());return ml+gl,ml,gl,logit,motion

@torch.no_grad()
def eval_offline(model,data,mean,std,batch):
    device=next(model.parameters()).device; m=torch.from_numpy(mean).to(device);s=torch.from_numpy(std).to(device); loader=DataLoader(data,batch_size=batch); pred=[];target=[];logits=[];total=[];motions=[];grips=[];model.eval()
    for x,a in loader:
        x=x.to(device);a=a.to(device);l,ml,gl,g,p=loss(model,x,a,m,s);pred.append(p.cpu().numpy());target.append(a.cpu().numpy());logits.append(g.cpu().numpy());total.append(float(l));motions.append(float(ml));grips.append(float(gl))
    p=np.concatenate(pred);a=np.concatenate(target);y=(a[:,6]>0).astype(int);q=(np.concatenate(logits)>=0).astype(int)
    def f1(v):
        tp=np.sum((q==v)&(y==v));fp=np.sum((q==v)&(y!=v));fn=np.sum((q!=v)&(y==v));pr=tp/(tp+fp) if tp+fp else 0.;re=tp/(tp+fn) if tp+fn else 0.;return {'precision':float(pr),'recall':float(re),'f1':float(2*pr*re/(pr+re) if pr+re else 0.)}
    e=p-a[:,:6];return {'total_loss':float(np.mean(total)),'motion_loss':float(np.mean(motions)),'gripper_loss':float(np.mean(grips)),'motion_mse':float(np.mean(e**2)),'motion_mae':float(np.mean(abs(e))),'translation_mae':float(np.mean(abs(e[:,:3]))),'rotation_mae':float(np.mean(abs(e[:,3:]))),'gripper_accuracy':float(np.mean(q==y)),'OPEN':f1(1),'CLOSE':f1(0)}

def train_one(root:Path,name:str,scenario:str,paths:dict[str,str],split:dict[str,list[str]],config:Config):
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed);torch.set_num_threads(1); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data={k:Episodes(paths,[x for x in split[k] if x in paths],scenario) for k in ('train','validation','test')};mean,std=norm(data['train']);np.savez(root/'normalizer.npz',mean=mean,std=std,input_dim=43)
    dump(root/'config.json',{**asdict(config),'architecture':'43->256->256->256 ReLU; tanh motion6 + binary gripper logit','scenario':scenario,'checkpoint_selection':'minimum validation total BC loss only','input':'full_physical_state / project state43, no stage/failure/milestones'})
    loader=DataLoader(data['train'],batch_size=config.batch_size,shuffle=True,generator=torch.Generator().manual_seed(config.seed));model=RecoveryBCPolicy().to(device);opt=AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay);mt=torch.from_numpy(mean).to(device);st=torch.from_numpy(std).to(device);best=float('inf');epoch_best=0;patience=0;hist=[]
    for epoch in range(1,config.max_epochs+1):
        model.train();ls=[];ms=[];gs=[]
        for x,a in loader:
            opt.zero_grad(set_to_none=True);l,ml,gl,_,_=loss(model,x.to(device),a.to(device),mt,st)
            if not torch.isfinite(l):raise FloatingPointError('non-finite training loss')
            l.backward();opt.step();ls.append(float(l.detach()));ms.append(float(ml.detach()));gs.append(float(gl.detach()))
        val=eval_offline(model,data['validation'],mean,std,config.batch_size);r={'epoch':epoch,'train_total_loss':float(np.mean(ls)),'train_motion_loss':float(np.mean(ms)),'train_gripper_loss':float(np.mean(gs)),'val_total_loss':val['total_loss'],'val_motion_loss':val['motion_loss'],'val_gripper_loss':val['gripper_loss'],'val_gripper_accuracy':val['gripper_accuracy']};hist.append(r);print(json.dumps({'bc':name,**r}),flush=True)
        if val['total_loss']<best:
            best=val['total_loss'];epoch_best=epoch;patience=0;torch.save({'format':'e2-specialized-bc-v1','model':model.state_dict(),'normalization_mean':mean,'normalization_std':std,'best_epoch':epoch,'best_val_total_loss':best,'val_motion_loss':val['motion_loss'],'val_gripper_accuracy':val['gripper_accuracy'],'config':asdict(config),'scenario':scenario},root/'best_val.pt')
        else:
            patience+=1
            if patience>=config.early_stopping_patience:break
    csv_write(root/'training_history.csv',hist);saved=torch.load(root/'best_val.pt',map_location='cpu',weights_only=False); frozen=RecoveryBCPolicy();frozen.load_state_dict(saved['model']);frozen.eval();test=eval_offline(frozen,data['test'],mean,std,config.batch_size);dump(root/'offline_test_summary.json',test)
    return frozen,mean,std,{'best_epoch':epoch_best,'val_motion_loss':saved['val_motion_loss'],'val_gripper_accuracy':saved['val_gripper_accuracy'],'offline':test,'datasets':data}

def action(model,state,mean,std):return model.action(state,mean,std)
def normal_rollout(seed,model,mean,std):
    e=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);spec=ExpertActionSpec();ad=ExpertCommandAdapter(e.ik_controller,spec);ob,_=e.reset(seed=seed,options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True});ad.reset(ob['ee_pose'],ob['q_obs']);rew=AWACRewardV1Online(bank.state43(e,ob));mil=np.zeros(5,bool);con=0;reason='timeout'
    try:
        for step in range(MAX):
            st=bank.state43(e,ob);res=ad.adapt(spec.denormalize(action(model,st,mean,std)));nob,*_=e.step(res.joint_target);ns=bank.state43(e,nob);con=0 if res.accepted else con+1;out=rew.step(st,ns,ik_failure=con>=IKMAX,time_limit=step+1>=MAX);mil|=out.milestones;ob=nob
            if out.task_success:reason='task_success';break
            if out.terminated or out.truncated:reason=out.termination_reason;break
        return {'Condition':'NORMAL','seed':seed,'Task/Recovery Success':reason=='task_success','Regrasp':'NA','Grasp':bool(mil[0]),'Lift':bool(mil[1]),'Transport':bool(mil[2]),'Place':bool(mil[3]),'Retreat':bool(mil[4]),'Unexpected Drop':reason=='illegal_drop','IK Failure':reason=='ik_failure_limit','Timeout':reason=='timeout','Steps':step+1,'Snapshot to Regrasp Steps':'NA','Snapshot to Final Success Steps':step if reason=='task_success' else 'NA','termination_reason':reason}
    finally:e.close()
def recovery_rollout(meta,model,mean,std):
    payload=pickle.loads(Path(meta['snapshot_path']).read_bytes());e=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False);spec=ExpertActionSpec();ad=ExpertCommandAdapter(e.ik_controller,spec);teacher=RuleBasedRecoveryPilot();initial,_=e.reset(seed=meta['environment_seed'],options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True});ad.reset(initial['ee_pose'],initial['q_obs']);rew=AWACRewardV1Online(bank.state43(e,initial));ob,con=bank.restore(e,ad,teacher,rew,payload);mil=np.zeros(5,bool);regrasp=None;reason='timeout'
    try:
        for step in range(MAX):
            st=bank.state43(e,ob);res=ad.adapt(spec.denormalize(action(model,st,mean,std)));nob,*_=e.step(res.joint_target);ns=bank.state43(e,nob);con=0 if res.accepted else con+1;out=rew.step(st,ns,ik_failure=con>=IKMAX,time_limit=step+1>=MAX);mil|=out.milestones
            if regrasp is None and not bool(ob['object_grasped']) and bool(nob['object_grasped']):regrasp=step
            ob=nob
            if out.task_success:reason='task_success';break
            if out.terminated or out.truncated:reason=out.termination_reason;break
        return {'Condition':meta['condition'],'snapshot_id':meta['snapshot_id'],'Task/Recovery Success':bool(reason=='task_success' and regrasp is not None),'Regrasp':regrasp is not None,'Grasp':'NA','Lift':'NA','Transport':bool(mil[2]),'Place':bool(mil[3]),'Retreat':bool(mil[4]),'Unexpected Drop':reason=='illegal_drop','IK Failure':reason=='ik_failure_limit','Timeout':reason=='timeout','Steps':step+1,'Snapshot to Regrasp Steps':regrasp if regrasp is not None else 'NA','Snapshot to Final Success Steps':step if reason=='task_success' else 'NA','termination_reason':reason}
    finally:e.close()
def mean(rows,key):
    x=[float(r[key]) for r in rows if r[key]!='NA'];return float(np.mean(x)) if x else 'NA'
def status(x):return 'STRONG' if x>=.7 else ('USABLE' if x>=.4 else 'WEAK')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id');args=ap.parse_args();stamp=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=OUT/f'run_{stamp}';
    for d in ('dataset_audit','splits','bc_normal','bc_grasp','bc_transport','bc_place','closed_loop','analysis'): (root/d).mkdir(parents=True,exist_ok=False)
    if sha(V2/'e2_failure_snapshot_bank_v2_manifest.json')!=V2_SHA:raise SystemExit('STOP V2 hash mismatch')
    formal=json.loads((DATASET/'episode_manifest.json').read_text());rejected=json.loads((DATASET/'failure_manifest.json').read_text());integrity=json.loads((DATASET/'integrity_report.json').read_text());split=json.loads((DATASET/'split_manifest.json').read_text());audit_data=stats(root,formal,rejected)
    expected={'NORMAL':1000,'GRASP_RECOVERY':300,'TRANSPORT_DROP':400,'PLACE_RECOVERY':300}
    if integrity['status']!='PASS' or integrity['trajectory_counts']!=expected or len(formal)!=2000 or len(rejected)!=9 or any(r['Final Failure'] for r in audit_data['rows']):raise SystemExit('STOP dataset audit mismatch')
    paths=split['episode_paths'];summary=[];models={}
    for name,scenario,directory in SCENARIOS:
        own={s:[x for x in split['splits'][s] if x in paths and next(y for y in formal if y['episode_id']==x)['trajectory_type']==scenario] for s in ('train','validation','test')}
        sizes={'NORMAL':(800,100,100),'GRASP_RECOVERY':(240,30,30),'TRANSPORT_DROP':(320,40,40),'PLACE_RECOVERY':(240,30,30)}[scenario]
        if tuple(map(len,(own['train'],own['validation'],own['test'])))!=sizes or set(own['train'])&set(own['validation']) or set(own['train'])&set(own['test']) or set(own['validation'])&set(own['test']):raise SystemExit(f'STOP split mismatch {scenario}')
        manifest={'split_unit':'episode','scenario':scenario,'source_split':'stage_dataset_v1_hysteresis_20260817T052000Z frozen split','splits':own,'episode_paths':{x:paths[x] for k in own.values() for x in k}};dump(root/'splits'/f'{directory}_split_manifest.json',manifest);dump(root/directory/'split_manifest.json',manifest)
        model,m,s,info=train_one(root/directory,name,scenario,paths,own,Config());models[scenario]=(model,m,s,info);ds=info['datasets'];summary.append({'BC':name,'Scenario':scenario,'Total Episodes':sum(sizes),'Train Episodes':sizes[0],'Val Episodes':sizes[1],'Test Episodes':sizes[2],'Train Transitions':len(ds['train']),'Val Transitions':len(ds['validation']),'Test Transitions':len(ds['test'])})
    csv_write(root/'analysis'/'dataset_and_split_summary.csv',summary)
    offline=[]
    for name,scenario,directory in SCENARIOS:
        z=models[scenario][3];q=z['offline'];offline.append({'BC':name,'Test Episodes':summary[[x['BC'] for x in summary].index(name)]['Test Episodes'],'Motion MSE':q['motion_mse'],'Motion MAE':q['motion_mae'],'Translation MAE':q['translation_mae'],'Rotation MAE':q['rotation_mae'],'Gripper Accuracy':q['gripper_accuracy'],'OPEN F1':q['OPEN']['f1'],'CLOSE F1':q['CLOSE']['f1']})
    csv_write(root/'analysis'/'bc_offline_test_summary.csv',offline)
    normal_model,normal_mean,normal_std,_=models['NORMAL'];normal=[normal_rollout(5_200_000+i,normal_model,normal_mean,normal_std) for i in range(100)];csv_write(root/'closed_loop'/'bc_normal_episode_summary.csv',normal);csv_write(root/'bc_normal'/'closed_loop_episode_summary.csv',normal)
    manifest=json.loads((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_text());snapshots=manifest['snapshots']; closed=[]
    for name,scenario,directory in SCENARIOS[1:]:
        target=FAILURE_FOR_SCENARIO[scenario];chosen=[x for x in snapshots if x['condition']==target]
        if len(chosen)!=100:raise SystemExit(f'STOP expected 100 {target}, found {len(chosen)}')
        mo,me,st,_=models[scenario];rows=[]
        for i,x in enumerate(chosen,1):
            rows.append(recovery_rollout(x,mo,me,st))
            if i%25==0:print(json.dumps({'bc':name,'closed_loop':i}),flush=True)
        csv_write(root/'closed_loop'/f'{directory}_episode_summary.csv',rows);csv_write(root/directory/'closed_loop_episode_summary.csv',rows);closed.extend([(name,target,rows)])
    allsummary=[]
    for name,scenario,directory in SCENARIOS:
        rows=normal if scenario=='NORMAL' else next(x[2] for x in closed if x[0]==name);allsummary.append({'BC':name,'Condition':'NORMAL' if scenario=='NORMAL' else next(x[1] for x in closed if x[0]==name),'N':len(rows),'Task/Recovery Success':mean(rows,'Task/Recovery Success'),'Regrasp':mean(rows,'Regrasp'),'Grasp':mean(rows,'Grasp'),'Lift':mean(rows,'Lift'),'Transport':mean(rows,'Transport'),'Place':mean(rows,'Place'),'Retreat':mean(rows,'Retreat'),'Unexpected Drop':mean(rows,'Unexpected Drop'),'IK Failure':mean(rows,'IK Failure'),'Timeout':mean(rows,'Timeout'),'Mean Steps':mean(rows,'Steps'),'Status':status(mean(rows,'Task/Recovery Success'))})
    csv_write(root/'analysis'/'bc_closed_loop_summary.csv',allsummary);recovery_rows=[x for _,_,r in closed for x in r];pooled={'label':'DESCRIPTIVE POOLED METRIC (three independent specialized models)','N':300,'recovery_success':mean(recovery_rows,'Task/Recovery Success'),'regrasp':mean(recovery_rows,'Regrasp')};dump(root/'analysis'/'descriptive_pooled_recovery.json',pooled)
    audit={'status':'PASS','dataset_path_exact':str(DATASET.resolve()),'formal_episodes':len(formal),'scenario_counts':expected,'all_final_success':True,'invalid_place_attempts_excluded':len(rejected)==9,'episode_level_split':True,'split_leakage':False,'input_43d_only':True,'no_stage_input':True,'no_failure_label_input':True,'no_milestone_input':True,'action_7d_correct':True,'hybrid_gripper_classification':True,'checkpoint_by_validation_only':True,'no_closed_loop_checkpoint_selection':True,'frozen_v2_unchanged':sha(V2/'e2_failure_snapshot_bank_v2_manifest.json')==V2_SHA,'global_not_used':True,'gamma_sweep_not_used':True,'nan':0,'inf':0};dump(root/'analysis'/'audit.json',audit)
    for _,_,directory in SCENARIOS:dump(root/directory/'audit.json',audit)
    dump(root/'analysis'/'results.json',{'offline':offline,'closed_loop':allsummary,'pooled':pooled,'teacher_reference':{'normal':1.0,'grasp':1.0,'transport':.89,'place':.94,'overall':283/300},'audit':audit});print(json.dumps({'output':str(root),'audit':'PASS','closed_loop':allsummary,'pooled':pooled},indent=2))
if __name__=='__main__':main()
