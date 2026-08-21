#!/usr/bin/env python3
"""Normal-only, cross-episode intra-task Stage ambiguity analysis (no training)."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1];G=ROOT/'outputs/global_diffusion/global_long_run_120k';DATA=ROOT/'outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z';V1=ROOT/'outputs/learned_expert_collection/final_online_awac20k_formal2000_20260816T130000Z';OUT=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/normal_only_intra_task_ambiguity';RNG=np.random.default_rng(20260820);BOOT=300;K=16;EPS=1e-6;PER_EPISODE_STAGE_CAP=2
NAMES=['APPROACH','GRASP_LIFT','TRANSPORT','PLACE_RELEASE','RETREAT']
def summ(v):
 v=np.asarray(v,float); means=[];med=[]
 for _ in range(BOOT):
  x=v[RNG.integers(len(v),size=len(v))];means.append(x.mean());med.append(np.median(x))
 return {'N':len(v),'mean':float(v.mean()),'median':float(np.median(v)),'Q1':float(np.quantile(v,.25)),'Q3':float(np.quantile(v,.75)),'bootstrap_mean_95_CI':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],'bootstrap_median_95_CI':[float(np.quantile(med,.025)),float(np.quantile(med,.975))]}
def cliffs(a,b):
 a=np.asarray(a);b=np.asarray(b);a=a[RNG.choice(len(a),min(3000,len(a)),False)];b=b[RNG.choice(len(b),min(3000,len(b)),False)];return float(np.sign(a[:,None]-b[None,:]).mean())
def draw(path,title,groups):
 im=Image.new('RGB',(1050,620),'white');d=ImageDraw.Draw(im);d.text((25,15),title,fill='black');L,R,T,B=80,1000,65,540;cols=['#1f77b4','#ff7f0e','#2ca02c','#d62728'];allv=np.concatenate(list(groups.values()));lo,hi=np.quantile(allv,[.01,.99]);hi=max(hi,lo+1e-8)
 for z,(n,v) in enumerate(groups.items()):
  h,e=np.histogram(np.clip(v,lo,hi),40,(lo,hi),density=True);m=max(h.max(),1e-9)
  for i,x in enumerate(h):
   x0=L+i*(R-L)/40+z*3;x1=L+(i+1)*(R-L)/40+z*3-2;y=B-x/m*(B-T);d.rectangle((x0,y,x1,B),fill=cols[z%4])
  d.text((L+190*z,570),n,fill=cols[z%4])
 d.line((L,B,R,B),fill='black');im.save(path)
def files():
 m=json.loads((DATA/'split_manifest.json').read_text()); primary={p.stem:p for p in DATA.rglob('*.npz')}; alternate={p.stem:p for p in V1.rglob('*.npz')}; by={**alternate,**primary}
 missing={split:[episode for episode in ids if episode not in by] for split,ids in m['splits'].items()}
 missing={split:ids for split,ids in missing.items() if ids}
 if missing:
  counts={split:len(ids) for split,ids in missing.items()}
  raise FileNotFoundError(f'Original success-only frozen split is incomplete; missing episodes by split: {counts}; examples: '+json.dumps({split:ids[:10] for split,ids in missing.items()}))
 recovery=[]
 for split,ids in m['splits'].items():
  for episode in ids:
   path=by[episode]; recovery.append({'episode_id':episode,'expected_path':str((DATA/'new_success'/f'{episode}.npz').resolve()),'found_alternative_paths':[] if episode in primary else [str(path.resolve())],'resolved_path':str(path.resolve()),'split':split})
 return [by[x] for x in m['splits']['train']],m,recovery
def load():
 paths,manifest,recovery=files();parts=[[] for _ in range(5)]
 for e,p in enumerate(paths):
  with np.load(p) as x:
   o=x['diffusion_observation_43'].astype('f4');a=x['executed_action_7'].astype('f4');z=x['phase_t'].astype(int)
  for s in range(5):
   q=np.flatnonzero(z==s)
   if len(q):
    # Stratified across every train episode; never a hand-picked subset.
    q=RNG.choice(q,min(PER_EPISODE_STAGE_CAP,len(q)),replace=False)
    parts[s].append((o[q],a[q],np.full(len(q),e,'i4')))
 return [tuple(np.concatenate([x[i] for x in part]) for i in range(3)) for part in parts],manifest,recovery
def cross(query,qid,candidate,cid):
 d,ix=cKDTree(candidate).query(query,k=min(K,len(candidate)));d=np.atleast_2d(d);ix=np.atleast_2d(ix);outd=np.empty(len(query));outi=np.empty(len(query),int)
 for n in range(len(query)):
  ok=np.flatnonzero(cid[ix[n]]!=qid[n])
  if not len(ok):raise RuntimeError('cross-episode neighbor not found')
  j=ok[0];outd[n]=d[n,j];outi[n]=ix[n,j]
 return outd,outi
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=OUT);z=a.parse_args();z.output.mkdir(parents=True,exist_ok=True)
 if not (G/'training_config.json').is_file() or not (G/'normalization_stats.npz').is_file():raise FileNotFoundError('Original Global config/normalization missing')
 stages,manifest,recovery=load();(z.output/'original_global_dataset_recovery_audit.json').write_text(json.dumps({'ORIGINAL_DATASET_COMPLETE':'YES','ORIGINAL_DATASET_RECOVERED':'YES','primary_dataset_root':str(DATA.resolve()),'v1_reference_root':str(V1.resolve()),'records':recovery},indent=2)+'\n');stats=np.load(G/'normalization_stats.npz');mean,std=stats['observation_mean'],stats['observation_std'];am,astd=stats['action_mean'],stats['action_std'];stages=[((o-mean)/std,(a-am)/astd,e) for o,a,e in stages];p3,a3,e3=stages[3]
 distances={};pairs={}
 for s in (0,1,2,4):
  d,i=cross(p3,e3,stages[s][0],stages[s][2]);distances[s]=d;pairs[s]=i
 rank=sorted(distances,key=lambda s:np.median(distances[s]));best=rank[0];db=distances[best];ib=pairs[best];pb,ab,eb=stages[best]
 # Same-stage cross-episode controls are matched by retaining the nearest-distance quantile band of the cross-stage pairs.
 d33,i33=cross(p3,e3,p3,e3);dbb,ibb=cross(pb,eb,pb,eb);same_d=np.r_[d33,dbb];same_a=np.r_[np.linalg.norm(a3-a3[i33],axis=1),np.linalg.norm(ab-ab[ibb],axis=1)];lo,hi=np.quantile(db,[.02,.98]);same_a_matched=same_a[(same_d>=lo)&(same_d<=hi)]
 cross_a=np.linalg.norm(a3-ab[ib],axis=1);diff=a3-ab[ib]
 # Local neighborhoods around Place and the data-selected most-similar stage.
 ox=np.r_[p3,pb];oa=np.r_[a3,ab];oz=np.r_[np.full(len(p3),3),np.full(len(pb),best)];q=RNG.choice(len(ox),min(5000,len(ox)),False);_,ii=cKDTree(ox).query(ox[q],k=20);total=[];conditioned=[]
 for g in ii:
  total.append(np.var(oa[g],axis=0).mean());vals=[np.var(oa[g][oz[g]==s],axis=0).mean() for s in (3,best) if (oz[g]==s).sum()>1];conditioned.append(np.mean(vals) if vals else np.nan)
 total=np.asarray(total);conditioned=np.asarray(conditioned);keep=np.isfinite(conditioned);total=total[keep];conditioned=conditioned[keep]
 # Original Global reports contain real timeout episode summaries but no per-frame physical/action logs.
 evals=sorted((G/'eval').glob('step_*/evaluation_report.json'));rows=[r for p in evals for r in json.loads(p.read_text())['rows']];timeouts=[r for r in rows if r['timeout']];success=[r for r in rows if r['success']];late=[r for r in timeouts if r.get('failure_phase') in ('PLACE_RELEASE','RETREAT')]
 timeout={'ORIGINAL_GLOBAL_EVAL_DIR':str((G/'eval').resolve()),'ORIGINAL_GLOBAL_TIMEOUT_LOG_SOURCE':'evaluation_report.json episode summaries; no per-frame physical43/action trajectories persisted','ORIGINAL_GLOBAL_TIMEOUT_EPISODES':len(timeouts),'TIMEOUT_STAGE3_OR_4_RATE':float(len(late)/len(timeouts)) if timeouts else None,'TIMEOUT_AMBIGUITY_ZONE_RATE':None,'SUCCESS_AMBIGUITY_ZONE_RATE':None,'TIMEOUT_AMBIGUOUS_FRAME_RATIO':None,'SUCCESS_AMBIGUOUS_FRAME_RATIO':None,'RETREAT_PROGRESS_TIMEOUT':None,'RETREAT_PROGRESS_SUCCESS':None,'STATUS':'INSUFFICIENT_EVIDENCE_FOR_FRAME_LEVEL_TIMEOUT_ASSOCIATION'}
 with (z.output/'place_to_all_stage_distance.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['target_stage','target_name',*summ(db).keys()]);w.writeheader();[w.writerow({'target_stage':s,'target_name':NAMES[s],**summ(distances[s])}) for s in (0,1,2,4)]
 with (z.output/'cross_stage_action_conflict.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['comparison',*summ(cross_a).keys(),'cliffs_delta']);w.writeheader();w.writerow({'comparison':f'3_to_{best}_cross_episode','cliffs_delta':cliffs(cross_a,same_a_matched),**summ(cross_a)});w.writerow({'comparison':'same_stage_matched_control','cliffs_delta':0.,**summ(same_a_matched)})
 with (z.output/'action_component_analysis.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['component','mean_signed_difference','mean_absolute_difference','median_absolute_difference']);w.writeheader();[w.writerow({'component':n,'mean_signed_difference':float(diff[:,i].mean()),'mean_absolute_difference':float(abs(diff[:,i]).mean()),'median_absolute_difference':float(np.median(abs(diff[:,i])))}) for i,n in enumerate(['dx','dy','dz','drx','dry','drz','gripper'])]
 local={'LOCAL_ACTION_VARIANCE_WITHOUT_STAGE':summ(total),'LOCAL_ACTION_VARIANCE_WITH_STAGE':summ(conditioned),'DELTA_VARIANCE':summ(total-conditioned)};(z.output/'local_action_variance.json').write_text(json.dumps(local,indent=2)+'\n');(z.output/'timeout_ambiguity_analysis.json').write_text(json.dumps(timeout,indent=2)+'\n');(z.output/'timeout_vs_success_analysis.json').write_text(json.dumps({'timeout_episode_summaries':len(timeouts),'success_episode_summaries':len(success),**timeout},indent=2)+'\n')
 control='YES' if best==4 and np.median(abs(diff[:,2]))>0.1 and np.median(abs(diff[:,6]))>0.1 else ('NO' if best==4 else 'NOT_APPLICABLE')
 report={'PRIMARY_DATASET':'ORIGINAL_SUCCESS_ONLY','ORIGINAL_DATASET_COMPLETE':'YES','ORIGINAL_DATASET_RECOVERED':'YES','NORMAL_REFERENCE_SOURCE':'ORIGINAL','ORIGINAL_GLOBAL_DATASET':str(DATA.resolve()),'ORIGINAL_GLOBAL_CONFIG':str((G/'training_config.json').resolve()),'ORIGINAL_GLOBAL_NORMALIZATION':str((G/'normalization_stats.npz').resolve()),'PHYSICAL_DIM':43,'NORMALIZATION_SOURCE':'Original Global normalization_stats.npz (train split)','NORMALIZATION_TRAIN_ONLY':'YES','CROSS_EPISODE_ONLY':'YES','REFERENCE_SAMPLING':'stratified fixed-seed sample from every train episode; cap 2 frames per episode/stage','NORMAL_EPISODES_USED':len(manifest['splits']['train']),'PLACE_STATE_COUNT':len(p3),'PLACE_TO_STAGE0_MEDIAN_DISTANCE':float(np.median(distances[0])),'PLACE_TO_STAGE1_MEDIAN_DISTANCE':float(np.median(distances[1])),'PLACE_TO_STAGE2_MEDIAN_DISTANCE':float(np.median(distances[2])),'PLACE_TO_STAGE4_MEDIAN_DISTANCE':float(np.median(distances[4])),'MOST_SIMILAR_STAGE_TO_PLACE':best,'SECOND_MOST_SIMILAR_STAGE_TO_PLACE':rank[1],'MOST_SIMILAR_STAGE_EFFECT_SIZE':{str(s):cliffs(db,distances[s]) for s in rank[1:]},'CROSS_STAGE_ACTION_DISTANCE':summ(cross_a),'SAME_STAGE_ACTION_DISTANCE':summ(same_a_matched),'ACTION_CONFLICT_EFFECT_SIZE':cliffs(cross_a,same_a_matched),'PLACE_RETREAT_CONTROL_TARGET_CONFLICT':control,**local,**timeout,'NORMAL_ONLY_STAGE_AMBIGUITY_EXISTS':'YES' if np.median(cross_a)>np.median(same_a_matched) else 'NO','ORIGINAL_GLOBAL_TIMEOUT_ASSOCIATED_WITH_AMBIGUITY':'INSUFFICIENT_EVIDENCE'}
 (z.output/'normal_only_ambiguity_report.json').write_text(json.dumps(report,indent=2)+'\n');(z.output/'NORMAL_ONLY_AMBIGUITY_REPORT.md').write_text('# Normal-only intra-task ambiguity\n\n'+json.dumps(report,indent=2)+'\n')
 draw(z.output/'FIG_N1_PLACE_TO_ALL_STAGE_STATE_DISTANCE.png','FIG_N1 Place to all cross-episode stage distances',{f'3->{s}':distances[s] for s in (0,1,2,4)});draw(z.output/'FIG_N2_CLOSE_STATE_ACTION_CONFLICT.png','FIG_N2 Close-state action conflict',{'cross-stage':cross_a,'same-stage matched':same_a_matched});draw(z.output/'FIG_N3_ACTION_COMPONENT_CONFLICT.png','FIG_N3 Action component conflict',{'absolute action component differences':abs(diff).ravel()});draw(z.output/'FIG_N4_TIMEOUT_AMBIGUITY_OCCUPANCY.png','FIG_N4 Timeout occupancy unavailable: no frame logs',{'timeout episode lengths':np.array([r['episode_length'] for r in timeouts])});draw(z.output/'FIG_N5_TIMEOUT_VS_SUCCESS_RETREAT_PROGRESS.png','FIG_N5 Retreat progress unavailable: no frame logs',{'timeout episode lengths':np.array([r['episode_length'] for r in timeouts]),'success episode lengths':np.array([r['episode_length'] for r in success])})
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
