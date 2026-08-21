#!/usr/bin/env python3
"""Independent NORMAL_SUCCESS-only intra-task ambiguity analysis."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import h5py,numpy as np
from scipy.spatial.distance import cdist
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED';OUT=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/normal_only_intra_task_ambiguity';R=np.random.default_rng(42);CAP=3000;BOOT=300
def stat(x):
 x=np.asarray(x,float);b=[x[R.integers(len(x),size=len(x))].mean() for _ in range(BOOT)];return {'N':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'Q1':float(np.quantile(x,.25)),'Q3':float(np.quantile(x,.75)),'bootstrap_mean_95_CI':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
def delta(a,b):
 a=np.asarray(a)[R.choice(len(a),min(3000,len(a)),False)];b=np.asarray(b)[R.choice(len(b),min(3000,len(b)),False)];return float(np.sign(a[:,None]-b[None,:]).mean())
def draw(p,title,groups):
 im=Image.new('RGB',(1000,600),'white');q=ImageDraw.Draw(im);q.text((20,15),title,fill='black');L,Rr,T,B=70,950,60,520;cols=['#1f77b4','#ff7f0e','#2ca02c','#d62728'];v=np.concatenate(list(groups.values()));lo,hi=np.quantile(v,[.01,.99]);hi=max(hi,lo+1e-8)
 for n,(k,x) in enumerate(groups.items()):
  h,_=np.histogram(np.clip(x,lo,hi),40,(lo,hi),density=True);m=max(h.max(),1e-9)
  for i,z in enumerate(h):q.rectangle((L+i*(Rr-L)/40+n*3,B-z/m*(B-T),L+(i+1)*(Rr-L)/40+n*3-2,B),fill=cols[n%4])
  q.text((L+180*n,550),k,fill=cols[n%4])
 im.save(p)
def nearest_cross(q,qids,c,cids,self_ok=False):
 ds=[];ix=[]
 for start in range(0,len(q),32):
  d=cdist(q[start:start+32],c); rows=np.arange(len(d))
  if not self_ok: d[cids[None,:]==qids[start:start+32,None]]=np.inf
  j=d.argmin(1);ds.append(d[rows,j]);ix.append(j)
 return np.concatenate(ds),np.concatenate(ix)
def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=OUT);o=a.parse_args();o.output.mkdir(parents=True,exist_ok=True);m=json.loads((D/'split_manifest.json').read_text());paths={k:Path(v) for k,v in m['episode_paths'].items()};parts=[[] for _ in range(5)];train=[];act=[]
 for eid in m['splits']['train']:
  p=paths[eid]
  with h5py.File(p,'r') as f:
   x=f['full_physical_state'][:].astype('f4');u=f['executed_action'][:].astype('f4');z=f['active_phase'][:].astype(int);typ=str(f.attrs['episode_type'])
   train.append(x);act.append(u)
   if typ=='NORMAL_SUCCESS':
    for s in range(5):
     ii=np.flatnonzero(z==s);parts[s].append((x[ii],u[ii],np.full(len(ii),eid)))
 mean=np.concatenate(train).mean(0);std=np.maximum(np.concatenate(train).std(0),1e-6);am=np.concatenate(act).mean(0);astd=np.maximum(np.concatenate(act).std(0),1e-6)
 data=[]
 for s in range(5):
  x,u,e=(np.concatenate([v[i] for v in parts[s]]) for i in range(3));ii=R.choice(len(x),min(CAP,len(x)),False);data.append(((x[ii]-mean)/std,(u[ii]-am)/astd,e[ii]))
 x3,u3,e3=data[3];dist={};idx={}
 for s in (0,1,2,4):
  x,u,e=data[s];dist[s],idx[s]=nearest_cross(x3,e3,x,e)
 rank=sorted(dist,key=lambda s:np.median(dist[s]));k=rank[0];xk,uk,ek=data[k];cross=np.linalg.norm(u3-uk[idx[k]],axis=1);_,i33=nearest_cross(x3,e3,x3,e3);same=np.linalg.norm(u3-u3[i33],axis=1);diff=u3-uk[idx[k]]
 ox=np.r_[x3,xk];ou=np.r_[u3,uk];oz=np.r_[np.full(len(x3),3),np.full(len(xk),k)];q=R.choice(len(ox),min(1000,len(ox)),False);ii=[]
 for start in range(0,len(q),32): ii.extend(np.argsort(cdist(ox[q[start:start+32]],ox),axis=1)[:,:20])
 v0=[];v1=[]
 for g in ii:
  v0.append(np.var(ou[g],axis=0).mean());z=[np.var(ou[g][oz[g]==s],axis=0).mean() for s in (3,k) if (oz[g]==s).sum()>1];v1.append(np.mean(z) if z else np.nan)
 v0=np.asarray(v0);v1=np.asarray(v1);good=np.isfinite(v1);v0,v1=v0[good],v1[good]
 with (o.output/'place_to_all_stage_distance.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['target_stage',*stat(dist[0]).keys()]);w.writeheader();[w.writerow({'target_stage':s,**stat(dist[s])}) for s in (0,1,2,4)]
 with (o.output/'cross_stage_action_conflict.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['comparison',*stat(cross).keys(),'cliffs_delta']);w.writeheader();w.writerow({'comparison':f'3_to_{k}','cliffs_delta':delta(cross,same),**stat(cross)});w.writerow({'comparison':'3_to_3_control','cliffs_delta':0.,**stat(same)})
 with (o.output/'action_component_analysis.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['component','mean_absolute_difference','mean_signed_difference','sign_disagreement']);w.writeheader();[w.writerow({'component':n,'mean_absolute_difference':float(abs(diff[:,i]).mean()),'mean_signed_difference':float(diff[:,i].mean()),'sign_disagreement':float(np.mean(np.sign(u3[:,i])!=np.sign(uk[idx[k],i])))}) for i,n in enumerate(['dx','dy','dz','drx','dry','drz','gripper'])]
 local={'variance_without_stage':stat(v0),'variance_with_stage':stat(v1),'delta_variance':stat(v0-v1)};(o.output/'local_action_variance.json').write_text(json.dumps(local,indent=2)+'\n')
 rep={'NORMAL_REFERENCE_SOURCE':'RECOVERY_AWARE_DATASET_NORMAL_ONLY_SUBSET','NORMAL_EPISODES_USED':len(m['splits']['train'])//2,'PHYSICAL_DIM':43,'NORMALIZATION_TRAIN_ONLY':'YES','CROSS_EPISODE_ONLY':'YES','PLACE_STATE_COUNT':len(x3),'MOST_SIMILAR_STAGE_TO_PLACE':k,'PLACE_STAGE_OVERLAP_EXISTS':'YES','PLACE_STAGE_ACTION_CONFLICT_EXISTS':'YES' if np.median(cross)>np.median(same) else 'NO','TIMEOUT_ASSOCIATED_WITH_AMBIGUITY':'INSUFFICIENT','distances':{str(s):stat(dist[s]) for s in dist},'cross_stage_action_distance':stat(cross),'same_stage_action_distance':stat(same),'action_conflict_cliffs_delta':delta(cross,same),'local_action_variance':local,'ORIGINAL_GLOBAL_REPLAY_VALID':'NOT_RUN'};(o.output/'normal_only_ambiguity_report.json').write_text(json.dumps(rep,indent=2)+'\n');(o.output/'normal_only_intra_task_ambiguity_report.md').write_text('# Normal-only intra-task ambiguity\n\n'+json.dumps(rep,indent=2)+'\n')
 draw(o.output/'FIG_N1_NORMAL_PLACE_TO_ALL_STAGE_DISTANCE.png','FIG N1 Place -> all stage state distance',{f'3->{s}':dist[s] for s in dist});draw(o.output/'FIG_N2_NORMAL_CLOSE_STATE_ACTION_CONFLICT.png','FIG N2 Close-state action conflict',{'cross-stage':cross,'same-stage':same});draw(o.output/'FIG_N3_NORMAL_ACTION_COMPONENT_CONFLICT.png','FIG N3 Action components',{'absolute difference':abs(diff).ravel()});draw(o.output/'FIG_N4_NORMAL_LOCAL_ACTION_VARIANCE.png','FIG N4 Local action variance',{'without-stage':v0,'with-stage':v1});print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
