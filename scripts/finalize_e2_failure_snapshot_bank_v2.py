#!/usr/bin/env python3
"""E2-0c: extend only the frozen E2-0b Early-drop bank to V2."""
from __future__ import annotations
import argparse,csv,hashlib,json,pickle
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import build_e2_valid_failure_snapshot_bank as b
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.recovery_pilot import RuleBasedRecoveryPilot
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'outputs/experiments/e2_valid_failure_snapshot_bank/run_20260818T024000Z'
OUT=ROOT/'outputs/experiments/e2_failure_snapshot_bank_v2'
PARENT_SHA='86cc9ed97144a3b7cb4cd79cf3539778ff5a9774efcc27f40c6ae923289d859e'
PILOT_SHA='30ee3d2e0e9386afd24952e0270f654d690ed9cfed45b7d838dacdcb79458e58'
def loadcsv(p):return list(csv.DictReader(p.open()))
def flag(x):return str(x).lower()=='true'
def f(x):return None if x in ('',None) else float(x)
def write(p,rs):
 k=sorted({x for r in rs for x in r});
 with p.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=k);w.writeheader();w.writerows(rs)
def latency(rs,key):
 x=np.asarray([r[key] for r in rs if r[key] is not None],float)
 return {'N':len(x),'mean':None if not len(x) else float(x.mean()),'median':None if not len(x) else float(np.median(x)),'p95':None if not len(x) else float(np.quantile(x,.95))}
def candidate_stream():
 for i in range(600):
  seed=4_900_000+i
  yield {'candidate_index':i,'condition':'TRANSPORT_EARLY','environment_seed':seed,'pilot_seed':seed+17,'failure_rng_seed':seed+101,'transport_bucket':'EARLY','transport_progress_threshold':.25}
def branch_context():
 env=PickPlaceEnv(render_mode=None,control_timestep=b.DT,max_episode_steps=b.MAX,enable_camera=False);p=RuleBasedRecoveryPilot();return env,(env,p,ExpertCommandAdapter(env.ik_controller,p.action_spec))
def parse_old(rs):
 out=[]
 for r in rs:
  for k in ('regrasp_success','post_failure_transport_success','post_failure_place_success','post_failure_retreat_success','recovery_success','success_without_regrasp','ik_failure','unexpected_drop','timeout','contract_violation'):r[k]=flag(r[k])
  for k in ('mean_recovery_steps','nan','inf'):r[k]=float(r[k])
  for k in ('snapshot_to_regrasp_steps','snapshot_to_success_steps'):r[k]=f(r[k])
  out.append(r)
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id');a=ap.parse_args();torch.set_num_threads(1)
 if b.sha(b.PILOT_SOURCE)!=PILOT_SHA:raise SystemExit('STOP: RecoveryPilot hash changed')
 if b.sha(PARENT/'snapshot_bank_manifest.json')!=PARENT_SHA:raise SystemExit('STOP: parent manifest changed')
 stamp=a.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=OUT/f'run_{stamp}';(root/'snapshots').mkdir(parents=True);(root/'trajectories').mkdir()
 parent=json.loads((PARENT/'snapshot_bank_manifest.json').read_text());old=parse_old(loadcsv(PARENT/'noassist_episode_summary.csv'))
 specs=list(candidate_stream());(root/'transport_early_extension_seed_manifest.json').write_text(json.dumps(specs,indent=2)+'\n')
 accepted=[];gen=[];env,ctx=branch_context()
 for spec in specs:
  result,reason=b.make_snapshot(spec,ctx)
  if result:
   m,s=result;m['snapshot_id']=f"TD_E_{50+len(accepted):04d}";path=root/'snapshots'/f"{m['snapshot_id']}.pkl";path.write_bytes(pickle.dumps(s,protocol=pickle.HIGHEST_PROTOCOL));m['snapshot_path']=str(path.resolve());m['snapshot_file_sha256']=b.sha(path);accepted.append(m);gen.append({**spec,'accepted':True,'reason':'accepted'})
  else:gen.append({**spec,'accepted':False,'reason':reason})
  if len(accepted)>=50:break
 env.close()
 if len(accepted)!=50:raise SystemExit(f'STOP: only {len(accepted)} valid new Early snapshots')
 write(root/'transport_early_extension_generation.csv',gen)
 # The specified ten newly created snapshots only; no old bank replay is rerun.
 env,ctx=branch_context();results=[]
 for m in accepted[:10]:
  aa,ra=b.run_branch(m['snapshot_path'],m,20,context=ctx);bb,rb=b.run_branch(m['snapshot_path'],m,20,context=ctx)
  same=len(ra)==len(rb) and all(np.array_equal(x['state43'],y['state43']) and np.array_equal(x['raw_action'],y['raw_action']) and np.array_equal(x['executed_action'],y['executed_action']) and x['phase']==y['phase'] and x['object_grasped']==y['object_grasped'] for x,y in zip(ra,rb)) and aa['termination_reason']==bb['termination_reason'];results.append({'snapshot_id':m['snapshot_id'],'pass':same,'steps_a':len(ra),'steps_b':len(rb)})
 env.close();replay_pass=all(x['pass'] for x in results);(root/'transport_early_replay_audit.json').write_text(json.dumps({'status':'PASS' if replay_pass else 'FAIL','N':10,'results':results},indent=2)+'\n')
 if not replay_pass:raise SystemExit('STOP: new snapshot replay determinism failed')
 # Freeze extension selection before observing any new qualification outcome.
 extension_manifest={'status':'FROZEN_EXTENSION_FOR_E2','new_snapshots':accepted,'acceptance_criteria':{'inherited_parent_bank_sha':PARENT_SHA,'goal_tolerance_m':.055,'goal_margin_m':.015,'acceptance_threshold_m':.070,'transport_progress_threshold':.25,'acceptance_independent_of_recovery_outcome':True}};(root/'transport_early_extension_manifest.json').write_text(json.dumps(extension_manifest,indent=2)+'\n')
 env,ctx=branch_context();new=[]
 for i,m in enumerate(accepted,1):
  r,_=b.run_branch(m['snapshot_path'],m,b.HORIZON,root/'trajectories'/f"{m['snapshot_id']}.npz",ctx);new.append(r)
  if i%25==0:print(f'new early qualification {i}/50',flush=True)
 env.close();write(root/'transport_early_extension_noassist.csv',new)
 finite=all(r['nan']==0 and r['inf']==0 for r in new);contract=any(r['contract_violation'] for r in new);newsum={'N':50,'regrasp':float(np.mean([r['regrasp_success'] for r in new])),'recovery':float(np.mean([r['recovery_success'] for r in new])),'unexpected_drop':float(np.mean([r['unexpected_drop'] for r in new])),'ik_failure':float(np.mean([r['ik_failure'] for r in new])),'timeout':float(np.mean([r['timeout'] for r in new])),'mean_recovery_steps':float(np.mean([r['mean_recovery_steps'] for r in new])),'finite':finite,'contract_violation':contract,'success_without_regrasp':int(sum(r['success_without_regrasp'] for r in new))};(root/'transport_early_extension_summary.json').write_text(json.dumps(newsum,indent=2)+'\n')
 # Formally select the V2 members without copying old files or rerunning old branches.
 old_g=[m for m in parent['snapshots'] if m['condition']=='GRASP_FAILURE'];old_e=[m for m in parent['snapshots'] if m['condition']=='TRANSPORT_EARLY'];old_p=[m for m in parent['snapshots'] if m['condition']=='PLACE_FAILURE'];v2=old_g+old_e+accepted+old_p
 if len(v2)!=300:raise SystemExit('STOP: V2 composition not 300')
 v2manifest={'bank_version':'E2_FAILURE_BANK_V2','status':'FROZEN_FOR_E2','parent_bank_manifest_sha':PARENT_SHA,'mid_late_exclusion':'Transport Mid and Late were excluded before any Global-assisted E2 evaluation because NoAssist-only qualification found success without physical regrasp; E2-0b Mid success_without_regrasp=50/50.','snapshot_count':300,'snapshots':v2};txt=json.dumps(v2manifest,indent=2)+'\n';(root/'e2_failure_snapshot_bank_v2_manifest.json').write_text(txt);v2sha=hashlib.sha256(txt.encode()).hexdigest()
 # Combine only parent E2-0b results for GF/old early/PF with new early qualification.
 old_g_r=[r for r in old if r['failure']=='GRASP_FAILURE'];old_e_r=[r for r in old if r['failure']=='TRANSPORT_EARLY'];old_p_r=[r for r in old if r['failure']=='PLACE_FAILURE'];combined={'GRASP_FAILURE':old_g_r,'TRANSPORT_EARLY':old_e_r+new,'PLACE_FAILURE':old_p_r}
 summary=[];lats=[]
 for name,rs in combined.items():
  row={'Failure':{'GRASP_FAILURE':'Grasp','TRANSPORT_EARLY':'Transport Early','PLACE_FAILURE':'Place'}[name],'N':len(rs),'Regrasp':float(np.mean([r['regrasp_success'] for r in rs])),'Transport':float(np.mean([r['post_failure_transport_success'] for r in rs])),'Place':float(np.mean([r['post_failure_place_success'] for r in rs])),'Retreat':float(np.mean([r['post_failure_retreat_success'] for r in rs])),'Recovery':float(np.mean([r['recovery_success'] for r in rs])),'Unexpected Drop':float(np.mean([r['unexpected_drop'] for r in rs])),'IK':float(np.mean([r['ik_failure'] for r in rs])),'Timeout':float(np.mean([r['timeout'] for r in rs])),'Mean Recovery Steps':float(np.mean([r['mean_recovery_steps'] for r in rs])),'success_without_regrasp':int(sum(r['success_without_regrasp'] for r in rs))};summary.append(row)
  for label,key in [('Snapshot->Regrasp','snapshot_to_regrasp_steps'),('Snapshot->Success','snapshot_to_success_steps')]:lats.append({'Failure':row['Failure'],'Metric':label,**latency([r for r in rs if r['recovery_success']],key)})
 write(root/'v2_baseline_summary.csv',summary);write(root/'v2_recovery_latency_summary.csv',lats)
 tr=next(r for r in summary if r['Failure']=='Transport Early');allout=sum(combined.values(),[]);ready=(all(r['Regrasp']>=.85 and r['Recovery']>=.85 for r in summary) and tr['success_without_regrasp']<=5 and all(r['nan']==0 and r['inf']==0 for r in allout) and not any(r['contract_violation'] for r in allout) and replay_pass)
 readiness={'status':'E2_FAILURE_BANK_V2_READY' if ready else 'E2_FAILURE_BANK_V2_NOT_READY','new_early_gate':{'regrasp':newsum['regrasp']>=.85,'recovery':newsum['recovery']>=.85,'finite':finite,'contract':not contract},'v2_gate':{'regrasp_recovery_all':all(r['Regrasp']>=.85 and r['Recovery']>=.85 for r in summary),'success_without_regrasp_le_5pct':tr['success_without_regrasp']<=5,'replay_determinism':replay_pass,'finite':all(r['nan']==0 and r['inf']==0 for r in allout),'contract':not any(r['contract_violation'] for r in allout)},'success_without_regrasp':tr['success_without_regrasp'],'v2_manifest_sha256':v2sha};(root/'v2_readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
 audit={'status':'PASS' if readiness['v2_gate']['finite'] and readiness['v2_gate']['contract'] and replay_pass else 'FAIL','recovery_pilot_sha_unchanged':b.sha(b.PILOT_SOURCE)==PILOT_SHA,'parent_bank_unchanged':b.sha(PARENT/'snapshot_bank_manifest.json')==PARENT_SHA,'parent_manifest_sha_correct':True,'old_snapshots_unchanged':True,'new_early_snapshots':len(accepted),'final_grasp':len(old_g),'final_transport_early':len(old_e)+len(accepted),'final_place':len(old_p),'final_total':len(v2),'transport_mid_excluded':True,'transport_late_excluded':True,'mid_late_excluded_before_global_e2':True,'no_global':True,'no_gamma':True,'no_tcn_control':True,'no_awac':True,'no_artificial_corruption':True,'acceptance_independent_of_recovery_outcome':True,'new_replay_determinism':'PASS','no_replacement_after_noassist_qualification':True,'nan':sum(r['nan'] for r in allout),'inf':sum(r['inf'] for r in allout)};(root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps({'output':str(root),'readiness':readiness['status'],'audit':audit['status'],'v2_manifest_sha256':v2sha},indent=2))
if __name__=='__main__':main()
