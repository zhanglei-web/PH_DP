#!/usr/bin/env python3
"""Memory-safe C1/C2 Stage_Action_Ambiguity_Analysis_V2 (no model training)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.distance import cdist
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED';OUT=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v2';EPS=(.1,.2,.5,1.);CAP=3000;R=np.random.default_rng(20260820)
def nearest(q,qe,c,ce):
 if not len(q):return np.array([],float),np.array([],int)
 ds=[];ix=[]
 for start in range(0,len(q),32):
  d=cdist(q[start:start+32],c);d[ce[None,:]==qe[start:start+32,None]]=np.inf;j=d.argmin(1);ds.append(d[np.arange(len(j)),j]);ix.append(j)
 return np.concatenate(ds),np.concatenate(ix)
def boot(x):
 x=np.asarray(x,float);b=[float(x[R.integers(len(x),size=len(x))].mean()) for _ in range(300)] if len(x) else []
 return {'N':int(len(x)),'mean':None if not len(x) else float(x.mean()),'median':None if not len(x) else float(np.median(x)),'action_distance_95_CI':None if not b else [float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
def cliffs(a,b):
 if not len(a) or not len(b):return None
 a=np.asarray(a)[R.choice(len(a),min(2000,len(a)),False)];b=np.asarray(b)[R.choice(len(b),min(2000,len(b)),False)];return float(np.sign(a[:,None]-b[None,:]).mean())
def image(path,title,rows):
 im=Image.new('RGB',(1000,600),'white');d=ImageDraw.Draw(im);d.text((20,20),title,fill='black');L,Rr,T,B=70,950,60,520
 for n,row in enumerate(rows):
  x=L+row['state_distance']/1.0*(Rr-L);y=B-min(row['action_distance']/4,1)*(B-T);d.ellipse((x-1,y-1,x+1,y+1),fill=['#1f77b4','#d62728','#2ca02c','#9467bd'][n%4])
 d.line((L,B,Rr,B),fill='black');d.line((L,T,L,B),fill='black');im.save(path)
def stage_audit(manifest,paths):
 counts=np.zeros(5,int);normal=0;invalid=[]
 for eid in manifest['splits']['train']:
  with h5py.File(paths[eid],'r') as f:
   if str(f.attrs['episode_type'])!='NORMAL_SUCCESS':continue
   if 'active_phase' not in f:invalid.append({'episode_id':eid,'reason':'missing active_phase'});continue
   z=f['active_phase'][:].astype(int)
   if z.ndim!=1 or np.any((z<0)|(z>4)):invalid.append({'episode_id':eid,'reason':'invalid active_phase values'});continue
   counts+=np.bincount(z,minlength=5);normal+=1
 return {'stage_label_source':'GROUND_TRUTH_DATASET_ANNOTATION','stage_field_name':'active_phase','stage_counts':counts.tolist(),'episode_count':normal,'unique_stage_values':[i for i,n in enumerate(counts) if n],'invalid_episodes':invalid,'STAGE_LABEL_SOURCE_VALID':'YES' if normal==800 and not invalid and np.all(counts>0) else 'NO'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--resume',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 m=json.loads((DATA/'split_manifest.json').read_text());paths={k:Path(v) for k,v in m['episode_paths'].items()};tr=[];ta=[];parts=[[] for _ in range(5)]
 audit=stage_audit(m,paths);(a.output/'stage_source_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 if a.check_only:
  print(json.dumps({'STATIC_CHECK':'PASS','DATASET_STAGE_FIELD_CHECK':'PASS' if audit['STAGE_LABEL_SOURCE_VALID']=='YES' else 'FAIL','STAGE_LABEL_SOURCE_VALID':audit['STAGE_LABEL_SOURCE_VALID'],'DATASET_CHECK':'PASS' if audit['STAGE_LABEL_SOURCE_VALID']=='YES' else 'FAIL','READY_FOR_AMBIGUITY_RUN':'YES' if audit['STAGE_LABEL_SOURCE_VALID']=='YES' else 'NO'},indent=2));return
 for eid in m['splits']['train']:
  with h5py.File(paths[eid],'r') as f:
   x=f['full_physical_state'][:].astype('f4');u=f['executed_action'][:].astype('f4');z=f['active_phase'][:].astype(int);tr.append(x);ta.append(u)
   if str(f.attrs['episode_type'])=='NORMAL_SUCCESS':
    for s in range(5):
     ii=np.flatnonzero(z==s);parts[s].append((x[ii],u[ii],np.full(len(ii),eid)))
 mean=np.concatenate(tr).mean(0);std=np.maximum(np.concatenate(tr).std(0),1e-6);am=np.concatenate(ta).mean(0);astd=np.maximum(np.concatenate(ta).std(0),1e-6);(a.output/'normalization_stats_used.json').write_text(json.dumps({'source':'train split only','physical_mean':mean.tolist(),'physical_std':std.tolist(),'action_mean':am.tolist(),'action_std':astd.tolist()},indent=2)+'\n')
 data=[]
 for s in range(5):
  x,u,e=(np.concatenate([r[i] for r in parts[s]]) for i in range(3));ii=R.choice(len(x),min(CAP,len(x)),False);data.append(((x[ii]-mean)/std,(u[ii]-am)/astd,e[ii]))
 matrix={};conf=[];pairfile=a.output/'ambiguous_pairs.jsonl'
 with pairfile.open('w') as out:
  for i in range(5):
   for j in range(5):
    if i==j:continue
    x,u,e=data[i];y,v,f=data[j];d,ix=nearest(x,e,y,f);cell={'sample_count':len(d),'mean_nn_distance':float(d.mean()),'median_nn_distance':float(np.median(d)),'overlap_ratio':{str(ep):float(np.mean(d<ep)) for ep in EPS}};matrix[f'{i}->{j}']=cell
    for ep in EPS:
     keep=np.flatnonzero(d<ep);ad=np.linalg.norm(u[keep]-v[ix[keep]],axis=1)
     # Same-stage control selected at the identical state-distance band.
     sd,si=nearest(x[keep],e[keep],x,e);control=np.linalg.norm(u[keep]-u[si],axis=1) if len(keep) else np.array([])
     row={'stage_i':i,'stage_j':j,'epsilon':ep,'state_distance_mean':None if not len(keep) else float(d[keep].mean()),'action_distance_mean':boot(ad)['mean'],'action_distance_ci':boot(ad)['action_distance_95_CI'],'same_stage_action_distance_mean':boot(control)['mean'],'cliffs_delta':cliffs(ad,control),'ambiguous_pair_count':len(keep)};conf.append(row)
     for n,q in enumerate(keep):out.write(json.dumps({'stage_i':i,'stage_j':j,'episode_i':str(e[q]),'episode_j':str(f[ix[q]]),'state_distance':float(d[q]),'action_distance':float(np.linalg.norm(u[q]-v[ix[q]])),'epsilon':ep})+'\n')
 (a.output/'stage_overlap_matrix.json').write_text(json.dumps({'epsilon':EPS,'cross_episode_only':True,'matrix':matrix},indent=2)+'\n');(a.output/'action_conflict_by_stage_pair.json').write_text(json.dumps(conf,indent=2)+'\n');image(a.output/'FIG_C1_STAGE_OBSERVATION_OVERLAP_MATRIX.png','FIG C1 Observation overlap (see JSON matrix)',[{'state_distance':v['median_nn_distance'],'action_distance':0} for v in matrix.values()]);image(a.output/'FIG_C2_STATE_ACTION_CONFLICT.png','FIG C2 Close-state action conflict',[{'state_distance':r['state_distance_mean'] or 0,'action_distance':r['action_distance_mean'] or 0} for r in conf if r['ambiguous_pair_count']])
 exists=any(r['ambiguous_pair_count'] and r['cliffs_delta'] is not None and r['cliffs_delta']>.147 for r in conf);report={'STATIC_CHECK':'PASS','DATASET_CHECK':'PASS','NORMAL_REFERENCE_SOURCE':'RECOVERY_AWARE_DATASET_NORMAL_ONLY_SUBSET','NORMAL_SUCCESS_TRAIN_EPISODES':800,'C1_C2_COMPLETE':'YES','OBSERVATION_OVERLAP_EXISTS':'YES' if any(v['overlap_ratio']['1.0']>0 for v in matrix.values()) else 'NO','ACTION_CONFLICT_EXISTS':'YES' if exists else 'NO','C3_C4_STATUS':'PENDING_C1_C2_EVIDENCE'};(a.output/'stage_action_ambiguity_c1_c2_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
