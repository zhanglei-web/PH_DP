#!/usr/bin/env python3
"""Independent clean-data Stage Ambiguity analysis for final-stage recovery."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED'
OUT=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/ambiguity_analysis'
EPS=1e-6; SAMPLE=10_000; BOOTSTRAPS=500; NEIGHBORS=20; RNG=np.random.default_rng(20260820)

def summary(x):
 x=np.asarray(x,float); b=[]
 for _ in range(BOOTSTRAPS): b.append(float(np.mean(x[RNG.integers(len(x),size=len(x))])))
 return {'n':int(len(x)),'mean':float(x.mean()),'median':float(np.median(x)),'q1':float(np.quantile(x,.25)),'q3':float(np.quantile(x,.75)),'bootstrap_mean_95_ci':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}

def effect(a,b):
 # Cliff's delta on a deterministic bounded subsample, avoiding a huge outer product.
 a,b=np.asarray(a),np.asarray(b); a=a[RNG.choice(len(a),min(3000,len(a)),replace=False)]; b=b[RNG.choice(len(b),min(3000,len(b)),replace=False)]
 return float((np.sign(a[:,None]-b[None,:]).mean()))

def png_hist(path, groups, title):
 im=Image.new('RGB',(1100,650),'white'); d=ImageDraw.Draw(im); d.text((25,18),title,fill='black'); left,right,top,bottom=80,1050,80,570
 values=np.concatenate([np.asarray(v) for v in groups.values()]); lo,hi=np.quantile(values,[.01,.99]); hi=max(hi,lo+1e-9); colors=['#1f77b4','#d62728','#2ca02c']
 for n,(name,v) in enumerate(groups.items()):
  hist,edges=np.histogram(np.clip(v,lo,hi),bins=40,range=(lo,hi),density=True); peak=max(hist.max(),1e-9)
  for i,h in enumerate(hist):
   x0=left+i*(right-left)/40+n*4; x1=left+(i+1)*(right-left)/40+n*4-2; y=bottom-h/peak*(bottom-top)
   d.rectangle((x0,y,x1,bottom),fill=colors[n%len(colors)])
  d.text((left+220*n,600),name,fill=colors[n%len(colors)])
 d.line((left,bottom,right,bottom),fill='black'); d.text((left,575),f'{lo:.3g}',fill='black'); d.text((right-70,575),f'{hi:.3g}',fill='black'); im.save(path)

def png_scatter(path, x1,y1,x2,y2):
 im=Image.new('RGB',(900,650),'white');d=ImageDraw.Draw(im);d.text((25,18),'FIG_C2 State distance vs action distance',fill='black'); L,R,T,B=80,850,60,570
 xx=np.r_[x1,x2];yy=np.r_[y1,y2]; xmax=max(np.quantile(xx,.99),1e-9); ymax=max(np.quantile(yy,.99),1e-9)
 for x,y,c in ((x1,y1,'#1f77b4'),(x2,y2,'#d62728')):
  for a,b in zip(x[::max(1,len(x)//3000)],y[::max(1,len(y)//3000)]):
   d.ellipse((L+a/xmax*(R-L)-1,B-b/ymax*(B-T)-1,L+a/xmax*(R-L)+1,B-b/ymax*(B-T)+1),fill=c)
 d.line((L,B,R,B),fill='black');d.line((L,T,L,B),fill='black');d.text((100,590),'blue=same-stage, red=cross-stage',fill='black');im.save(path)

def load():
 manifest=json.loads((DATA/'split_manifest.json').read_text()); paths={k:Path(v) for k,v in manifest['episode_paths'].items()}; train_x=[];train_a=[]; normal=[]; recovery=[]
 for split in ('train','validation','test'):
  for eid in manifest['splits'][split]:
   path=paths[eid]
   with h5py.File(path,'r') as f:
    x=f['full_physical_state'][:].astype('f4');a=f['executed_action'][:].astype('f4');phase=f['active_phase'][:].astype(int);time=f['timestep_raw'][:].astype(int);typ=str(f.attrs['episode_type'])
    if split=='train': train_x.append(x);train_a.append(a)
    if typ=='NORMAL_SUCCESS':
     m=np.isin(phase,(3,4)); normal.append((x[m],a[m],phase[m]))
    elif typ=='PLACE_RECOVERY_SUCCESS':
     raw=DATA/'raw_rollouts'/path.relative_to(DATA/'episodes');
     with h5py.File(raw,'r') as r:
      p=r['active_phase'][:].astype(int);rt=r['timestep_raw'][:].astype(int); reg=np.flatnonzero((p[:-1]==3)&(p[1:]==0))+1
     if len(reg):
      start=rt[reg[0]];m=time>=start; recovery.append((x[m],a[m],phase[m]))
 mean=np.concatenate(train_x).mean(0);std=np.maximum(np.concatenate(train_x).std(0),1e-6);am=np.concatenate(train_a).mean(0);astd=np.maximum(np.concatenate(train_a).std(0),1e-6)
 def join(rows): return tuple(np.concatenate([r[i] for r in rows]) for i in range(3))
 return join(normal),join(recovery),mean,std,am,astd,manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 (nx,na,np_), (rx,ra,rp),mean,std,am,astd,manifest=load(); nx=(nx-mean)/std;rx=(rx-mean)/std;na=(na-am)/astd;ra=(ra-am)/astd
 # Bound samples for reproducible, tractable nonparametric analyses.
 ni=RNG.choice(len(nx),min(SAMPLE,len(nx)),replace=False);ri=RNG.choice(len(rx),min(SAMPLE,len(rx)),replace=False); nx,na,np_=nx[ni],na[ni],np_[ni];rx,ra,rp=rx[ri],ra[ri],rp[ri]
 cross_tree=cKDTree(nx);cross_d,cross_i=cross_tree.query(rx,k=1);cross_action=np.linalg.norm(ra-na[cross_i],axis=1)
 random_i=RNG.integers(len(nx),size=len(rx));random_d=np.linalg.norm(rx-nx[random_i],axis=1)
 # Same-stage nearest within recovery context, excluding self.
 same_d=np.empty(len(rx));same_action=np.empty(len(rx))
 for stage in range(5):
  ids=np.flatnonzero(rp==stage)
  if len(ids)<2: same_d[ids]=np.nan;same_action[ids]=np.nan;continue
  d,i=cKDTree(rx[ids]).query(rx[ids],k=2);same_d[ids]=d[:,1];same_action[ids]=np.linalg.norm(ra[ids]-ra[ids][i[:,1]],axis=1)
 valid=np.isfinite(same_d);same_d,same_action=same_d[valid],same_action[valid]
 cross_score=cross_action/(cross_d+EPS);same_score=same_action/(same_d+EPS);random_action=np.linalg.norm(ra-na[random_i],axis=1);random_score=random_action/(random_d+EPS)
 # Neighborhood variance: exact k nearest from combined final-stage/recovery contexts.
 ox=np.r_[nx,rx];oa=np.r_[na,ra];oz=np.r_[np_,rp];q=RNG.choice(len(ox),min(3000,len(ox)),replace=False);_,idx=cKDTree(ox).query(ox[q],k=min(NEIGHBORS,len(ox)))
 total=[];conditioned=[]
 for group in idx:
  aa=oa[group];total.append(float(np.mean(np.var(aa,axis=0))))
  pieces=[np.var(aa[oz[group]==z],axis=0).mean() for z in range(5) if (oz[group]==z).sum()>=2]
  conditioned.append(float(np.mean(pieces)) if pieces else np.nan)
 conditioned=np.asarray(conditioned);keep=np.isfinite(conditioned);total=np.asarray(total)[keep];conditioned=conditioned[keep]
 report={'ANALYSIS':'independent clean-data stage ambiguity','dataset':str(DATA.resolve()),'normalization':'train split only, physical43 and action7 separately','epsilon':EPS,'normal_final_states':int(len(nx)),'place_recovery_post_regression_states':int(len(rx)),'cross_stage_state_distance':summary(cross_d),'same_stage_state_distance':summary(same_d),'random_cross_stage_state_distance':summary(random_d),'cross_stage_action_distance':summary(cross_action),'same_stage_action_distance':summary(same_action),'cross_stage_ambiguity_score':summary(cross_score),'same_stage_ambiguity_score':summary(same_score),'random_ambiguity_score':summary(random_score),'local_action_variance_without_stage':summary(total),'local_action_variance_with_stage':summary(conditioned),'statistics':{'cross_vs_random_state_distance_cliffs_delta':effect(cross_d,random_d),'cross_vs_same_action_distance_cliffs_delta':effect(cross_action,same_action),'cross_vs_same_ambiguity_score_cliffs_delta':effect(cross_score,same_score),'local_variance_reduction_mean':float(np.mean(total-conditioned))},'interpretation_constraint':'This analysis is independent of closed-loop model success results.'}
 (a.output/'ambiguity_report.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'ambiguity_config.json').write_text(json.dumps({'sample_cap':SAMPLE,'bootstrap_resamples':BOOTSTRAPS,'neighborhood_k':NEIGHBORS,'epsilon':EPS,'normal_context':'NORMAL_SUCCESS stages 3/4','recovery_context':'PLACE_RECOVERY_SUCCESS clean states at/after raw true 3->0 regression'},indent=2)+'\n')
 png_hist(a.output/'FIG_C1_CROSS_STAGE_STATE_DISTANCE.png',{'cross-stage':cross_d,'same-stage':same_d,'random':random_d},'FIG_C1 Cross-stage state distance');png_scatter(a.output/'FIG_C2_STATE_DISTANCE_VS_ACTION_DISTANCE.png',same_d,same_action,cross_d,cross_action);png_hist(a.output/'FIG_C3_AMBIGUITY_SCORE_DISTRIBUTION.png',{'cross-stage':cross_score,'same-stage':same_score,'random':random_score},'FIG_C3 Ambiguity score');png_hist(a.output/'FIG_C4_LOCAL_ACTION_VARIANCE.png',{'without-stage':total,'with-stage':conditioned},'FIG_C4 Local action variance')
 print(json.dumps({'AMBIGUITY_ANALYSIS_VALID':'YES','output':str(a.output.resolve()),'cross_pairs':len(cross_d),'local_neighborhoods':len(total)},indent=2))
if __name__=='__main__':main()
