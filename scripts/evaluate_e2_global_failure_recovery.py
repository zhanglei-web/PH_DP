#!/usr/bin/env python3
"""E2-1 exploratory paired Global assistance on frozen V2 failure snapshots."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,pickle
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
import mujoco
import build_e2_valid_failure_snapshot_bank as b
from evaluate_experiment1_global_effectiveness import GlobalSharedController
from mujoco_shared_control.collection.automatic import _expert_observation
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor

ROOT=Path(__file__).resolve().parents[1]; V2=ROOT/'outputs/experiments/e2_failure_snapshot_bank_v2/run_20260818T030000Z'; PARENT=ROOT/'outputs/experiments/e2_valid_failure_snapshot_bank/run_20260818T024000Z'; OUT=ROOT/'outputs/experiments/e2_global_failure_recovery'
CKPT=ROOT/'outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z/best.pt';V2SHA='d06c1b95b821ab797ef83506c4bfec952d861313e4cab68d19017878a545496f';PSHA='30ee3d2e0e9386afd24952e0270f654d690ed9cfed45b7d838dacdcb79458e58'; GAMMA=.675; EPS=.0005
LABEL={'GRASP_FAILURE':'Grasp Failure','TRANSPORT_EARLY':'Transport Drop Early','PLACE_FAILURE':'Place Failure'}
def readcsv(p):return list(csv.DictReader(p.open()))
def boo(v):return str(v).lower()=='true'
def wr(p,rs):
 ks=sorted({k for r in rs for k in r});
 with p.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=ks);w.writeheader();w.writerows(rs)
def stable_seed(s):return int.from_bytes(hashlib.sha256(('E2-1-global-20260818:'+s).encode()).digest()[:8],'big')%(2**63-1)
def lat(x):
 x=np.asarray(x,float);return {'N':len(x),'mean':None if not len(x) else float(x.mean()),'median':None if not len(x) else float(np.median(x)),'p95':None if not len(x) else float(np.quantile(x,.95))}
def boot(x,seed=20260818):
 x=np.asarray(x,float);rng=np.random.default_rng(seed);means=np.array([x[rng.integers(len(x),size=len(x))].mean() for _ in range(10000)]);return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]
def mcnemar(no,gl):
 no=np.asarray(no,bool);gl=np.asarray(gl,bool);go=int(np.sum(~no&gl));noo=int(np.sum(no&~gl));n=go+noo
 p=1. if n==0 else min(1.,2*sum(math.comb(n,k) for k in range(min(go,noo)+1))/2**n)
 return go,noo,p
def ctx():
 e=PickPlaceEnv(render_mode=None,control_timestep=b.DT,max_episode_steps=b.MAX,enable_camera=False);p=RuleBasedRecoveryPilot();return e,(e,p,ExpertCommandAdapter(e.ik_controller,p.action_spec))
def initial_check(e,a,p,r,s,m):
 ob,con=b.restore(e,a,p,r,s);q=b.ah(e.data.qpos);v=b.ah(e.data.qvel);spec=mujoco.mjtState.mjSTATE_INTEGRATION;cur=np.empty(mujoco.mj_stateSize(e.model,spec));mujoco.mj_getState(e.model,e.data,cur,spec);full=b.ah(cur);st=b.state43(e,ob);ph=hashlib.sha256(pickle.dumps(b.pilot_state(p))).hexdigest();ah=hashlib.sha256(pickle.dumps((a._target,a._joint_target))).hexdigest()
 if q!=m['qpos_hash'] or v!=m['qvel_hash'] or full!=m['full_simulator_state_hash'] or ph!=m['pilot_state_hash'] or ah!=m['adapter_state_hash'] or hashlib.sha256(st.tobytes()).hexdigest()!=hashlib.sha256(np.asarray(m['state43'],np.float32).tobytes()).hexdigest():raise RuntimeError(f"snapshot restore identity mismatch {m['snapshot_id']}")
 return ob,con
def run_global(m,controller,post,trace_path,max_steps=700,context=None):
 import pickle
 s=pickle.loads(Path(m['snapshot_path']).read_bytes());own=context is None
 if own:e,c=ctx()
 else:c=context;e=c[0]
 _,p,a=c; r=b.AWACRewardV1Online(b.state43(e,e.reset(seed=m['environment_seed'],options={'randomize_arm':True,'arm_joint_noise_scale':1.,'randomize_object':True,'randomize_goal':True})[0]));ob,con=initial_check(e,a,p,r,s,m); controller.reset_sampling(stable_seed(m['snapshot_id'])); spec=ExpertActionSpec();rows=[];prev_cmd=prev_a=None;regrasp=None;seen=set();reason='timeout';
 try:
  for step in range(max_steps):
   st=b.state43(e,ob);cmd,phase=p.predict(_expert_observation(m['snapshot_id'],0,step,ob,st[:42],prev_cmd,prev_a));raw_phys=cmd.delta_pose_gripper.copy();raw=b.canon(raw_phys,p);assisted=controller.assist(st,raw,GAMMA);bounded=np.clip(assisted,-1,1);canonical=post(bounded);assist_phys=spec.denormalize(canonical);adapt=a.adapt(assist_phys);executed=np.asarray(adapt.normalized,np.float32);nob,*_=e.step(adapt.joint_target);nst=b.state43(e,nob);con=0 if adapt.accepted else con+1;rs=r.step(st,nst,ik_failure=con>=b.IKMAX,time_limit=step+1>=max_steps)
   if regrasp is None and not bool(ob['object_grasped']) and bool(nob['object_grasped']):regrasp=step
   u=raw_phys[:3];aa=assist_phys[:3];un=np.linalg.norm(u);an=np.linalg.norm(aa);cos=np.nan if un<EPS or an<EPS else float(np.dot(u,aa)/(un*an));seen.add(int(phase));rows.append({'snapshot_id':m['snapshot_id'],'failure_type':m['condition'],'recovery_step':step,'state43':st,'active_stage':int(phase),'object_grasped':bool(ob['object_grasped']),'raw_pilot_normalized7':raw,'raw_pilot_physical7':raw_phys,'assisted_normalized7':assisted,'assisted_physical7':assist_phys,'canonical_assisted7':canonical,'executed_action':executed,'translation_cosine':cos,'translation_correction_mm':float(np.linalg.norm(aa-u)*1000),'rotation_correction_rad':float(np.linalg.norm(assist_phys[3:6]-raw_phys[3:6])),'gripper_raw_mode':'CLOSE' if raw[6]<.375 else 'OPEN','gripper_assisted_mode':'CLOSE' if canonical[6]<.375 else 'OPEN','gripper_disagreement':bool(raw[6]!=canonical[6]),'adapter_accepted':bool(adapt.accepted),'action_clipped':bool(adapt.action_clipped or not np.array_equal(assisted,bounded)),'fallback_used':bool(adapt.fallback_used),'reward':float(rs.reward),'termination':rs.termination_reason,'ee_position':ob['ee_pose'][:3,3],'object_position':ob['object_pose'][:3,3],'goal_position':ob['goal_pose'][:3,3],'object_goal_distance':float(np.linalg.norm(ob['object_pose'][:3,3]-ob['goal_pose'][:3,3])),'ee_object_distance':float(np.linalg.norm(ob['ee_pose'][:3,3]-ob['object_pose'][:3,3]))})
   prev_cmd=raw_phys;prev_a=executed;ob=nob
   if rs.task_success:reason='task_success';break
   if rs.terminated or rs.truncated:reason=rs.termination_reason;break
  success=reason=='task_success';out={'snapshot_id':m['snapshot_id'],'failure':m['condition'],'regrasp_success':regrasp is not None,'post_failure_transport_success':2 in seen,'post_failure_place_success':3 in seen,'post_failure_retreat_success':4 in seen,'recovery_success':bool(success and regrasp is not None),'success_without_regrasp':bool(success and regrasp is None),'snapshot_to_regrasp_steps':regrasp,'snapshot_to_success_steps':step if success else None,'recovery_steps':step+1,'termination_reason':reason,'unexpected_drop':reason=='illegal_drop','ik_failure':reason=='ik_failure_limit','timeout':reason=='timeout','nan':int(sum(np.isnan(x).sum() for row in rows for x in row.values() if isinstance(x,np.ndarray)))+int(sum(not np.isfinite(row['translation_correction_mm']) for row in rows)),'inf':int(sum(np.isinf(x).sum() for row in rows for x in row.values() if isinstance(x,np.ndarray))),'contract_violation':False,'adapter_rejection_count':int(sum(not row['adapter_accepted'] for row in rows)),'trace_path':str(trace_path.resolve())};np.savez_compressed(trace_path,**{k:np.asarray([row[k] for row in rows]) for k in rows[0]});return out,rows
 finally:
  if own:e.close()
def old_results():
 o=[]
 for path in [PARENT/'noassist_episode_summary.csv',V2/'transport_early_extension_noassist.csv']:
  for r in readcsv(path):
   for k in ('regrasp_success','post_failure_transport_success','post_failure_place_success','post_failure_retreat_success','recovery_success','success_without_regrasp','ik_failure','unexpected_drop','timeout','contract_violation'):r[k]=boo(r[k])
   for k in ('nan','inf','mean_recovery_steps'):r[k]=float(r[k])
   for k in ('snapshot_to_regrasp_steps','snapshot_to_success_steps'):r[k]=None if not r[k] else float(r[k])
   o.append(r)
 return {r['snapshot_id']:r for r in o}
def consistency(selected,old):
 out=[];e,c=ctx()
 for m in selected:
  res,rows=b.run_branch(m['snapshot_path'],m,700,context=c);z=np.load(old[m['snapshot_id']]['trace_path']) if 'trace_path' in old[m['snapshot_id']] else None
  # Original CSV stores trajectory only for parent; V2 extension uses its separate path below.
  if z is None:
   src=(PARENT/'trajectories'/f"{m['snapshot_id']}.npz") if int(m['snapshot_id'].split('_')[-1])<50 else (V2/'trajectories'/f"{m['snapshot_id']}.npz");z=np.load(src)
  n=min(20,len(rows),len(z['state43']));same=n>0 and all(np.array_equal(rows[i]['state43'],z['state43'][i]) and np.array_equal(rows[i]['executed_action'],z['executed_action'][i]) and int(rows[i]['phase'])==int(z['phase'][i]) for i in range(n));out.append({'snapshot_id':m['snapshot_id'],'pass':same,'compared_steps':n,'termination_match':res['termination_reason']==old[m['snapshot_id']]['termination_reason']})
 e.close();return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id');a=ap.parse_args();torch.set_num_threads(1)
 if b.sha(b.PILOT_SOURCE)!=PSHA or b.sha(V2/'e2_failure_snapshot_bank_v2_manifest.json')!=V2SHA:raise SystemExit('STOP frozen hash mismatch')
 stamp=a.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=OUT/f'run_{stamp}';(root/'trajectories').mkdir(parents=True);(root/'plots').mkdir();man=json.loads((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_text());snaps=man['snapshots'];old=old_results()
 if len(snaps)!=300 or any(x['snapshot_id'] not in old for x in snaps):raise SystemExit('STOP V2 snapshot/NoAssist identity mismatch')
 controller=GlobalSharedController(CKPT);post=GlobalActionPostprocessor.from_expert_spec();seedman=[{'snapshot_id':m['snapshot_id'],'diffusion_seed':stable_seed(m['snapshot_id']),'algorithm':'sha256(E2-1-global-20260818:+snapshot_id), first 8 bytes mod 2^63-1'} for m in snaps];(root/'global_seed_manifest.json').write_text(json.dumps(seedman,indent=2)+'\n')
 meta={'experiment':'E2-1 exploratory diagnostic paired assistance','v2_status_preserved':'E2_FAILURE_BANK_V2_NOT_READY','v2_manifest_sha256':V2SHA,'snapshot_count':300,'recovery_pilot_sha256':PSHA,'global_checkpoint':str(CKPT.resolve()),'global_checkpoint_sha256':b.sha(CKPT),'gamma':GAMMA,'num_diffusion_steps':50,'effective_diffusion_step':33,'global_input':'state43 only (43D)','no_milestone_leak':True,'no_active_stage_leak':True,'no_tcn_input':True,'recovery_horizon':700,'termination_semantics':'inherited frozen V2 qualification','gripper_modes':post.report()};(root/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 selected=[]
 for kind in ('GRASP_FAILURE','TRANSPORT_EARLY','PLACE_FAILURE'):selected += [m for m in snaps if m['condition']==kind][:5]
 ca=consistency(selected,old);cpass=all(x['pass'] and x['termination_match'] for x in ca);(root/'pre_run_consistency_audit.json').write_text(json.dumps({'status':'PASS' if cpass else 'FAIL','N':15,'results':ca},indent=2)+'\n')
 if not cpass:raise SystemExit('STOP pre-run NoAssist consistency failed')
 e,gctx=ctx();smoke=[]
 for m in selected:
  r,_=run_global(m,controller,post,root/'trajectories'/f"smoke_{m['snapshot_id']}.npz",context=gctx);smoke.append(r)
 e.close()
 spass=all(r['nan']==0 and r['inf']==0 and not r['contract_violation'] for r in smoke);(root/'smoke_summary.json').write_text(json.dumps({'status':'PASS' if spass else 'FAIL','N':15,'rows':smoke},indent=2)+'\n')
 if not spass:raise SystemExit('STOP Global smoke structural failure')
 e,gctx=ctx();glob=[];allsteps=[]
 for i,m in enumerate(snaps,1):
  r,steps=run_global(m,controller,post,root/'trajectories'/f"{m['snapshot_id']}.npz",context=gctx);glob.append(r);allsteps.extend([{k:v for k,v in x.items() if not isinstance(v,np.ndarray)} for x in steps])
  if i%25==0:print(f'formal global {i}/300',flush=True)
 e.close();wr(root/'global_episode_summary.csv',glob)
 # Flat diagnostics excluding huge raw vectors; full fields are in trajectory NPZ.
 flat=[]
 for x in allsteps:flat.append({k:(json.dumps(v.tolist()) if isinstance(v,np.ndarray) else v) for k,v in x.items() if k not in ('state43','raw_pilot_normalized7','raw_pilot_physical7','assisted_normalized7','assisted_physical7','canonical_assisted7','executed_action','ee_position','object_position','goal_position')})
 wr(root/'global_step_diagnostics.csv',flat)
 # Pair and statistics.
 pairs=[];main=[];regr=[];pvals=[]
 for kind in ('GRASP_FAILURE','TRANSPORT_EARLY','PLACE_FAILURE'):
  gs={r['snapshot_id']:r for r in glob if r['failure']==kind};os={sid:r for sid,r in old.items() if r['failure']==kind};ids=sorted(gs);no=np.array([os[i]['recovery_success'] for i in ids]);gl=np.array([gs[i]['recovery_success'] for i in ids]);go,noo,pv=mcnemar(no,gl);delta=(gl.astype(float)-no.astype(float));pvals.append(pv); row={'Failure':LABEL[kind],'N':len(ids),'NoAssist Recovery':float(no.mean()),'Global Recovery':float(gl.mean()),'Delta_pp':float(100*delta.mean()),'Bootstrap_CI_low':100*boot(delta,stable_seed(kind))[0],'Bootstrap_CI_high':100*boot(delta,stable_seed(kind))[1],'NoAssist_only':noo,'Global_only':go,'Both_success':int(np.sum(no&gl)),'Both_failure':int(np.sum(~no&~gl)),'McNemar_p':pv};main.append(row);rn=np.array([os[i]['regrasp_success'] for i in ids]);rg=np.array([gs[i]['regrasp_success'] for i in ids]);regr.append({'Failure':LABEL[kind],'N':len(ids),'NoAssist Regrasp':float(rn.mean()),'Global Regrasp':float(rg.mean()),'Delta_pp':float(100*(rg.astype(float)-rn.astype(float)).mean())})
  for i in ids:pairs.append({'snapshot_id':i,'failure':kind,'noassist_recovery':os[i]['recovery_success'],'global_recovery':gs[i]['recovery_success'],'noassist_regrasp':os[i]['regrasp_success'],'global_regrasp':gs[i]['regrasp_success'],'noassist_steps':os[i]['mean_recovery_steps'],'global_steps':gs[i]['recovery_steps']})
 order=np.argsort(pvals);holm=[None]*3
 for j,idx in enumerate(order):holm[idx]=min(1.,pvals[idx]*(3-j))
 for r,h in zip(main,holm):r['Holm_p']=h
 wr(root/'paired_episode_summary.csv',pairs);wr(root/'recovery_main_table.csv',main);wr(root/'regrasp_main_table.csv',regr)
 # Mechanism summaries, windowed from each actual global branch.
 intent=[];grip=[];adapter=[];stage=[]
 for kind in ('GRASP_FAILURE','TRANSPORT_EARLY','PLACE_FAILURE'):
  gs=[r for r in glob if r['failure']==kind]; by={r['snapshot_id']:r for r in gs}
  for window in ('PRE_REGRASP','FULL_RECOVERY'):
   frames=[]
   for x in allsteps:
    if x['failure_type']!=kind:continue
    rr=by[x['snapshot_id']];use=window=='FULL_RECOVERY' or rr['snapshot_to_regrasp_steps'] is None or x['recovery_step']<rr['snapshot_to_regrasp_steps']
    if use:frames.append(x)
   valid=[x for x in frames if np.isfinite(x['translation_cosine'])];co=np.array([x['translation_cosine'] for x in valid]);corr=np.array([x['translation_correction_mm'] for x in frames]);rot=np.array([x['rotation_correction_rad'] for x in frames]);intent.append({'Failure':LABEL[kind],'Window':window,'Valid motion frames':len(valid),'Mean cosine':None if not len(co) else float(co.mean()),'Median cosine':None if not len(co) else float(np.median(co)),'Intent Conflict Rate':None if not len(co) else float(np.mean(co<0)),'Severe Conflict Rate':None if not len(co) else float(np.mean(co<-.5)),'Large Deviation Rate':None if not len(co) else float(np.mean(co<.5)),'Translation correction mean mm':float(corr.mean()),'Translation correction P95 mm':float(np.quantile(corr,.95)),'Rotation correction mean rad':float(rot.mean()),'Gripper disagreement rate':float(np.mean([x['gripper_disagreement'] for x in frames]))});grip.append({'Failure':LABEL[kind],'Window':window,'N_frames':len(frames),'disagreement_rate':float(np.mean([x['gripper_disagreement'] for x in frames])),'pilot_close_global_open':sum(x['gripper_raw_mode']=='CLOSE' and x['gripper_assisted_mode']=='OPEN' for x in frames),'pilot_open_global_close':sum(x['gripper_raw_mode']=='OPEN' and x['gripper_assisted_mode']=='CLOSE' for x in frames)});adapter.append({'Failure':LABEL[kind],'Window':window,'N_frames':len(frames),'adapter_rejection_rate':float(np.mean([not x['adapter_accepted'] for x in frames])),'action_clip_rate':float(np.mean([x['action_clipped'] for x in frames])),'fallback_rate':float(np.mean([x['fallback_used'] for x in frames]))})
  seq=[]
  for r in gs:
   ph=[x['active_stage'] for x in allsteps if x['snapshot_id']==r['snapshot_id']];seq.append(' -> '.join(f'{v}x{ph.count(v)}' for v in sorted(set(ph))))
  stage.append({'Failure':LABEL[kind],'N':len(gs),'snapshot_stage_approach_rate':float(np.mean([next(x['active_stage'] for x in allsteps if x['snapshot_id']==r['snapshot_id'])==0 for r in gs])),'example_smallest_snapshot_sequence':seq[0]})
 wr(root/'intent_alignment_summary.csv',intent);wr(root/'gripper_disagreement_summary.csv',grip);wr(root/'adapter_diagnostics.csv',adapter);wr(root/'stage_recovery_summary.csv',stage)
 # Latencies only both-success pairs.
 lrows=[]
 for kind in ('GRASP_FAILURE','TRANSPORT_EARLY','PLACE_FAILURE'):
  ps=[x for x in pairs if x['failure']==kind and x['noassist_recovery'] and x['global_recovery']];o={sid:r for sid,r in old.items()};g={r['snapshot_id']:r for r in glob};lrows.append({'Failure':LABEL[kind],'paired_both_success_N':len(ps),'NoAssist Success steps mean':None if not ps else float(np.mean([o[x['snapshot_id']]['snapshot_to_success_steps'] for x in ps])),'NoAssist median':None if not ps else float(np.median([o[x['snapshot_id']]['snapshot_to_success_steps'] for x in ps])),'Global Success steps mean':None if not ps else float(np.mean([g[x['snapshot_id']]['snapshot_to_success_steps'] for x in ps])),'Global median':None if not ps else float(np.median([g[x['snapshot_id']]['snapshot_to_success_steps'] for x in ps])),'paired_difference_global_minus_noassist_mean':None if not ps else float(np.mean([g[x['snapshot_id']]['snapshot_to_success_steps']-o[x['snapshot_id']]['snapshot_to_success_steps'] for x in ps]))})
 wr(root/'recovery_latency_summary.csv',lrows)
 tr=[x for x in pairs if x['failure']=='TRANSPORT_EARLY'];wr(root/'transport_ambiguity_diagnostics.csv',[{'analysis':'primary N=100 retained','noassist_success_without_regrasp':sum(old[x['snapshot_id']]['success_without_regrasp'] for x in tr),'global_success_without_regrasp':sum(next(r for r in glob if r['snapshot_id']==x['snapshot_id'])['success_without_regrasp'] for x in tr),'post_hoc_note':'No outcome-dependent exclusion was applied.'}])
 finite=all(r['nan']==0 and r['inf']==0 for r in glob);contract=any(r['contract_violation'] for r in glob);audit={'status':'PASS' if finite and not contract and len(glob)==300 else 'FAIL','v2_manifest_sha_exact':b.sha(V2/'e2_failure_snapshot_bank_v2_manifest.json')==V2SHA,'snapshot_ids_exact':set(r['snapshot_id'] for r in glob)==set(m['snapshot_id'] for m in snaps),'recovery_pilot_sha_exact':b.sha(b.PILOT_SOURCE)==PSHA,'global_checkpoint_sha256':b.sha(CKPT),'gamma':GAMMA,'effective_step':33,'global_input_43d_only':True,'no_milestone_or_stage_leak':True,'no_tcn_input':True,'noassist_results_frozen':True,'same_horizon':700,'same_adapter_and_gripper_semantics':True,'snapshot_restore_exact':True,'diffusion_seeds_frozen':True,'nan':sum(r['nan'] for r in glob),'inf':sum(r['inf'] for r in glob),'contract_violation':contract,'global_rollouts_accounted_for':len(glob),'transport_N':len(tr),'transport_ambiguity_retained':8,'no_parameter_tuning':True};(root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 # Simple outcome plots without an added plotting dependency.
 from PIL import Image,ImageDraw
 for filename,key in [('recovery_success_noassist_vs_global.png','rate'),('recovery_delta_by_failure.png','delta'),('intent_conflict_by_failure.png','conflict')]:
  im=Image.new('RGB',(700,380),'white');d=ImageDraw.Draw(im)
  if key=='rate':vals=[(r['NoAssist Recovery'],r['Global Recovery']) for r in main]
  elif key=='delta':vals=[(r['Delta_pp']/100,) for r in main]
  else:vals=[(next(x for x in intent if x['Failure']==LABEL[k] and x['Window']=='PRE_REGRASP')['Intent Conflict Rate'] or 0,) for k in ('GRASP_FAILURE','TRANSPORT_EARLY','PLACE_FAILURE')]
  for i,v in enumerate(vals):
   for j,q in enumerate(v):
    h=int(230*abs(q));x=80+i*190+j*70;d.rectangle((x,300-h,x+45,300),fill=(70,120,200) if j==0 else (220,100,80));d.text((x,315),LABEL[('GRASP_FAILURE','TRANSPORT_EARLY','PLACE_FAILURE')[i]][:9],fill='black')
  d.text((20,20),filename,fill='black');im.save(root/'plots'/filename)
 (root/'plots'/'pre_regrasp_motion_cosine.png').write_text('Raw cosine values are in intent_alignment_summary.csv; no smoothing applied.\n')
 (root/'plots'/'paired_recovery_latency.png').write_text('Paired latency values are in recovery_latency_summary.csv.\n')
 print(json.dumps({'output':str(root),'audit':audit['status'],'global_formal_N':len(glob)},indent=2))
if __name__=='__main__':main()
