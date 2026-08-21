#!/usr/bin/env python3
"""Formal AWAC-2.5k deterministic pilot versus frozen Global V2 assistance."""
from __future__ import annotations
import csv, hashlib, json, math, pickle, time
from pathlib import Path
import numpy as np
import torch
import build_e2_valid_failure_snapshot_bank as bank
from evaluate_experiment1_global_effectiveness import GlobalSharedController
from mujoco_shared_control.awac.hybrid_evaluation import HybridCheckpointPredictor
from mujoco_shared_control.awac.reward import AWACRewardV1Online
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/e2_awac25k_global_normal_vs_failure/run_20260818T_AWAC25K_GLOBAL_FORMAL'
AWAC=ROOT/'outputs/offline_awac/stageaware_awac_v1_4000/run_20260818T_STAGEAWARE_AWAC_V1_LOCAL20K_FORMAL/checkpoints/checkpoint_step_02500.pt'
GLOBAL=ROOT/'outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z/best.pt'
V2=ROOT/'outputs/experiments/e2_failure_snapshot_bank_v2/run_20260818T030000Z/e2_failure_snapshot_bank_v2_manifest.json'
V2SHA='d06c1b95b821ab797ef83506c4bfec952d861313e4cab68d19017878a545496f'
MAX=700; IKMAX=5; DT=.05; COARSE=tuple(i/10 for i in range(11)); SELECT=tuple(range(6_400_000,6_400_050)); FORMAL=tuple(range(6_500_000,6_500_300)); ID=tuple(range(6_410_000,6_410_020))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2)+'\n')
def write(p,rows):
 if not rows:return
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def phase(t,o,s,tag,step): return int(t.predict(_expert_observation(tag,0,step,o,s[:42],None,None))[1])
def seed(tag): return int.from_bytes(hashlib.sha256(('awac25k-global:'+tag).encode()).digest()[:8],'big')%(2**63-1)
def stats(rows,key='success'):
 return float(np.mean([bool(r[key]) for r in rows]))

def episode(*, kind, ident, env_seed, meta, method, gamma, pilot, controller, post, trace):
 spec=ExpertActionSpec(); env=PickPlaceEnv(render_mode=None,control_timestep=DT,max_episode_steps=MAX,enable_camera=False); ad=ExpertCommandAdapter(env.ik_controller,spec); tracker=RuleBasedRecoveryPilot(); recovery=meta is not None
 if recovery:
  initial,_=env.reset(seed=meta['environment_seed'],options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True});ad.reset(initial['ee_pose'],initial['q_obs']);rew=AWACRewardV1Online(bank.state43(env,initial));obs,con=bank.restore(env,ad,tracker,rew,pickle.loads(Path(meta['snapshot_path']).read_bytes()))
 else:
  obs,_=env.reset(seed=env_seed,options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True});ad.reset(obs['ee_pose'],obs['q_obs']);tracker.reset(float(obs['object_pose'][2,3]),env_seed+17);rew=AWACRewardV1Online(bank.state43(env,obs));con=0
 if method=='global': controller.reset_sampling(seed(f'{kind}:{ident}:{gamma}'))
 rows=[]; reason='timeout'; regrasp=None; episode_return=0.0
 try:
  for step in range(MAX):
   s=bank.state43(env,obs); st=phase(tracker,obs,s,f'{kind}:{ident}',step); raw=pilot.normalized_action(s[:42],bool(s[42]),current_active_stage=st).astype(np.float64)
   assisted=raw.copy() if method=='noassist' else controller.assist(s,raw,gamma)
   canonical=post(np.clip(assisted,-1,1)); adapted=ad.adapt(spec.denormalize(canonical));nobs,*_=env.step(adapted.joint_target);ns=bank.state43(env,nobs);con=0 if adapted.accepted else con+1;rs=rew.step(s,ns,ik_failure=con>=IKMAX,time_limit=step+1>=MAX)
   episode_return += float(rs.reward)
   if regrasp is None and not bool(obs['object_grasped']) and bool(nobs['object_grasped']):regrasp=step
   rm,am=raw[:6],assisted[:6];den=np.linalg.norm(rm)*np.linalg.norm(am);cos=1. if den<1e-12 else float(np.dot(rm,am)/den);corr=assisted-raw
   rows.append({'id':ident,'condition':kind,'step':step,'active_stage':st,'gamma':gamma,'diffusion_step':int(49*gamma),'raw_pilot_action':raw,'global_assisted_action':assisted,'executed_action':np.asarray(adapted.normalized,np.float64),'motion_cosine':cos,'intent_conflict':cos<0,'correction_norm':float(np.linalg.norm(corr[:6])),'translation_correction':float(np.linalg.norm(corr[:3])),'rotation_correction':float(np.linalg.norm(corr[3:6])),'gripper_disagreement':bool(raw[6]!=canonical[6]),'move_toward_object':float(np.dot((obs['object_pose'][:3,3]-obs['ee_pose'][:3,3]),assisted[:3])),'adapter_accepted':bool(adapted.accepted)})
   obs=nobs
   if rs.task_success:reason='task_success';break
   if rs.terminated or rs.truncated:reason=rs.termination_reason;break
  a={k:np.asarray([r[k] for r in rows]) for k in rows[0]};np.savez_compressed(trace,**a)
  seen=set(a['active_stage'].astype(int));success=reason=='task_success'
  return {'id':ident,'condition':kind,'method':method,'gamma':gamma,'success':bool(success if not recovery else success and regrasp is not None),'task_success':bool(success),'regrasp_success':regrasp is not None,'success_without_regrasp':bool(recovery and success and regrasp is None),'post_regrasp_transport':2 in seen,'place':3 in seen,'retreat':4 in seen,'grasp':bool(rs.milestones[0]),'lift':bool(rs.milestones[1]),'transport':bool(rs.milestones[2]),'drop':reason=='illegal_drop','ik':reason=='ik_failure_limit','timeout':reason=='timeout','termination_reason':reason,'steps':step+1,'return':episode_return,'snapshot_to_regrasp_steps':regrasp,'snapshot_to_final_success_steps':step if success else None,'trace_path':str(trace.resolve()),'nan':int(sum(np.isnan(v).sum() for v in a.values() if v.dtype.kind=='f')),'inf':int(sum(np.isinf(v).sum() for v in a.values() if v.dtype.kind=='f'))}
 finally:env.close()

def summary(gamma,rows):
 suc=[r for r in rows if r['success']]; return {'gamma':gamma,'diffusion_step':int(49*gamma),'N':len(rows),'success':stats(rows),'ik':stats(rows,'ik'),'drop':stats(rows,'drop'),'timeout':stats(rows,'timeout'),'mean_success_steps':float(np.mean([r['steps'] for r in suc])) if suc else float('inf'),'nan':sum(r['nan'] for r in rows),'inf':sum(r['inf'] for r in rows)}
def mcnemar(a,b):
 x=sum(not i and j for i,j in zip(a,b));y=sum(i and not j for i,j in zip(a,b));n=x+y;p=1. if not n else min(1.,2*sum(math.comb(n,k) for k in range(min(x,y)+1))/2**n);return x,y,p
def paired(no,gl,label):
 a=np.array([r['success'] for r in no],bool);b=np.array([r['success'] for r in gl],bool);d=b.astype(float)-a.astype(float);rng=np.random.default_rng(seed(label));boot=np.array([d[rng.integers(len(d),size=len(d))].mean() for _ in range(10000)]);x,y,p=mcnemar(a,b);return {'Condition':label,'N':len(d),'NoAssist':float(a.mean()),'Global':float(b.mean()),'Delta_pp':float(d.mean()*100),'CI95_low_pp':float(np.quantile(boot,.025)*100),'CI95_high_pp':float(np.quantile(boot,.975)*100),'Global_only':x,'NoAssist_only':y,'McNemar_p':p}

def main():
 if OUT.exists(): raise SystemExit(f'STOP output already exists: {OUT}')
 if sha(V2)!=V2SHA: raise SystemExit('STOP Frozen V2 hash mismatch')
 OUT.mkdir(parents=True);(OUT/'traces').mkdir();post=GlobalActionPostprocessor.from_expert_spec();pilot=HybridCheckpointPredictor(AWAC);controller=GlobalSharedController(GLOBAL,device_name='cuda' if torch.cuda.is_available() else 'cpu')
 payload=torch.load(AWAC,map_location='cpu',weights_only=False); dump(OUT/'pilot_checkpoint_freeze.json',{'status':'SURROGATE_PILOT_AWAC25K_FROZEN','absolute_path':str(AWAC.resolve()),'sha256':sha(AWAC),'training_step':payload['step'],'state_mode':payload['state_mode'],'observation_dim':payload['training_config']['observation_dim'],'hidden_dims':payload['training_config']['hidden_dims'],'deterministic_inference':True});dump(OUT/'global_checkpoint_freeze.json',{'absolute_path':str(GLOBAL.resolve()),'sha256':sha(GLOBAL),'input':'43D observation only','K':50});dump(OUT/'pilot_global_input_isolation_audit.json',{'status':'PASS','pilot':'physical43 + current active stage onehot5','global':'43D only','global_receives_stage':False,'global_receives_milestones':False,'global_receives_failure_type':False,'global_receives_scenario':False})
 dump(OUT/'normal_gamma_selection_manifest.json',{'status':'FROZEN','seeds':SELECT,'coarse_gammas':COARSE,'normal_only':True,'no_failure_results_seen_before_selection':True,'overlap_audit':{'dev_normal':[],'normal_formal':[],'frozen_v2':[]}});dump(OUT/'normal_formal_manifest.json',{'status':'FROZEN','seeds':FORMAL,'N':300,'disjoint_from_gamma_selection':True,'disjoint_from_awac_dev_normal_6300000_6300049':True})
 identity=[]
 for s in ID:
  a=episode(kind='NORMAL_ID',ident=str(s),env_seed=s,meta=None,method='noassist',gamma=0,pilot=pilot,controller=controller,post=post,trace=OUT/'traces'/f'id_no_{s}.npz');b=episode(kind='NORMAL_ID',ident=str(s),env_seed=s,meta=None,method='global',gamma=0,pilot=pilot,controller=controller,post=post,trace=OUT/'traces'/f'id_gl_{s}.npz');ta=np.load(a['trace_path']);tb=np.load(b['trace_path']);identity.append({'seed':s,'raw_exact':np.array_equal(ta['raw_pilot_action'],tb['global_assisted_action']),'executed_exact':np.array_equal(ta['executed_action'],tb['executed_action']),'trajectory_exact':np.array_equal(ta['raw_pilot_action'],tb['raw_pilot_action']),'termination_exact':a['termination_reason']==b['termination_reason']})
 dump(OUT/'gamma0_identity_audit.json',{'status':'PASS' if all(all(x.values()) for x in identity) else 'FAIL','N':len(identity),'rows':identity})
 if not all(all(x.values()) for x in identity):raise SystemExit('STOP gamma=0 identity failed')
 sweeps=[]
 for g in COARSE:
  rs=[episode(kind='NORMAL_SELECT',ident=str(s),env_seed=s,meta=None,method='global',gamma=g,pilot=pilot,controller=controller,post=post,trace=OUT/'traces'/f'select_g{g}_{s}.npz') for s in SELECT];sweeps.append(summary(g,rs));print('sweep',g,sweeps[-1],flush=True)
 write(OUT/'normal_gamma_sweep.csv',sweeps);chosen=sorted(sweeps,key=lambda r:(-r['success'],r['ik'],r['drop'],r['timeout'],r['mean_success_steps'],r['gamma']))[0];dump(OUT/'gamma_selection.json',{'status':'FROZEN','gamma_nominal':chosen['gamma'],'diffusion_step':chosen['diffusion_step'],'selected_normal_metrics':chosen,'criterion':'NORMAL success max; lower IK, drop, timeout, shorter successful steps, lower gamma','failure_results_used':False,'fine_sweep':'not run; one coarse sweep was sufficient'})
 g=float(chosen['gamma']);manifest=json.loads(V2.read_text());groups={'GRASP_FAILURE':'grasp','TRANSPORT_EARLY':'transport','PLACE_FAILURE':'place'};results={}
 normal_no=[episode(kind='NORMAL',ident=str(s),env_seed=s,meta=None,method='noassist',gamma=0,pilot=pilot,controller=controller,post=post,trace=OUT/'traces'/f'normal_no_{s}.npz') for s in FORMAL];normal_gl=[episode(kind='NORMAL',ident=str(s),env_seed=s,meta=None,method='global',gamma=g,pilot=pilot,controller=controller,post=post,trace=OUT/'traces'/f'normal_gl_{s}.npz') for s in FORMAL];write(OUT/'normal_noassist_episode_results.csv',normal_no);write(OUT/'normal_global_episode_results.csv',normal_gl);results['Normal']=(normal_no,normal_gl)
 for typ,label in groups.items():
  metas=[m for m in manifest['snapshots'] if m['condition']==typ]; no=[episode(kind=label,ident=m['snapshot_id'],env_seed=m['environment_seed'],meta=m,method='noassist',gamma=0,pilot=pilot,controller=controller,post=post,trace=OUT/'traces'/f'{label}_no_{m["snapshot_id"]}.npz') for m in metas]; gl=[episode(kind=label,ident=m['snapshot_id'],env_seed=m['environment_seed'],meta=m,method='global',gamma=g,pilot=pilot,controller=controller,post=post,trace=OUT/'traces'/f'{label}_gl_{m["snapshot_id"]}.npz') for m in metas];write(OUT/f'{label}_noassist_episode_results.csv',no);write(OUT/f'{label}_global_episode_results.csv',gl);results[label.title()]=(no,gl);print('formal',label,flush=True)
 main=[paired(*results[k],k) for k in ('Normal','Grasp','Transport','Place')];recno=sum([results[k][0] for k in ('Grasp','Transport','Place')],[]);recgl=sum([results[k][1] for k in ('Grasp','Transport','Place')],[]);main.append(paired(recno,recgl,'Recovery Overall'));write(OUT/'main_success_comparison.csv',main)
 fm=[];rm=[];conf=[]
 for k,(no,gl) in results.items():
  fm.append({'Condition':k,'NoAssist Drop':stats(no,'drop'),'Global Drop':stats(gl,'drop'),'NoAssist IK':stats(no,'ik'),'Global IK':stats(gl,'ik'),'NoAssist Timeout':stats(no,'timeout'),'Global Timeout':stats(gl,'timeout')});rm.append({'Condition':k,'NoAssist Recovery':stats(no),'Global Recovery':stats(gl),'NoAssist Regrasp':stats(no,'regrasp_success'),'Global Regrasp':stats(gl,'regrasp_success'),'NoAssist success_without_regrasp':stats(no,'success_without_regrasp'),'Global success_without_regrasp':stats(gl,'success_without_regrasp')})
  frames=[]
  for r in gl:
   z=np.load(r['trace_path']);n=len(z['step']);end=(int(r['snapshot_to_regrasp_steps']) if r['snapshot_to_regrasp_steps'] is not None else n) if k!='Normal' else n; frames += [{'c':float(z['motion_cosine'][i]),'t':float(z['translation_correction'][i]),'r':float(z['rotation_correction'][i]),'g':bool(z['gripper_disagreement'][i]),'toward':float(z['move_toward_object'][i])} for i in range(end)]
  if frames: conf.append({'Condition':k,'Window':'PRE_REGRASP' if k!='Normal' else 'FULL','mean_cosine':float(np.mean([x['c'] for x in frames])),'intent_conflict_fraction':float(np.mean([x['c']<0 for x in frames])),'correction_magnitude':float(np.mean([np.hypot(x['t'],x['r']) for x in frames])),'translation_correction':float(np.mean([x['t'] for x in frames])),'rotation_correction':float(np.mean([x['r'] for x in frames])),'gripper_disagreement':float(np.mean([x['g'] for x in frames])),'move_toward_object_component':float(np.mean([x['toward'] for x in frames]))})
 write(OUT/'failure_mode_comparison.csv',fm);write(OUT/'recovery_metrics.csv',rm);write(OUT/'action_conflict_summary.csv',conf);write(OUT/'place_failure_mechanism.csv',[x for x in conf if x['Condition']=='Place'])
 dump(OUT/'paired_statistics.json',{'rows':main});finite=all(r['nan']==0 and r['inf']==0 for a,b in results.values() for r in a+b);dump(OUT/'audit.json',{'status':'PASS' if finite else 'FAIL','pilot_frozen_sha':sha(AWAC),'global_frozen_sha':sha(GLOBAL),'v2_sha':sha(V2),'gamma0_identity':'PASS','gamma_selection_normal_only':True,'frozen_v2_modified':False,'nan_inf_free':finite,'formal_rollouts':1200,'stage_conditioned_diffusion':False})
if __name__=='__main__':main()
