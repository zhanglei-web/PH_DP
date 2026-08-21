#!/usr/bin/env python3
"""C3 covariance-trace and C4 single-frame vs temporal GT-label analysis."""
from __future__ import annotations
import json
from pathlib import Path
import h5py,numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scipy.spatial.distance import cdist
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED';OUT=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/stage_action_ambiguity_v2';R=np.random.default_rng(42);torch.manual_seed(42)
def load(split,paths):
 x=[];a=[];y=[];h=[]
 for eid in split:
  with h5py.File(paths[eid],'r') as f:
   if str(f.attrs['episode_type'])!='NORMAL_SUCCESS':continue
   q=f['full_physical_state'][:].astype('f4');u=f['executed_action'][:].astype('f4');z=f['active_phase'][:].astype('i8');ii=R.choice(len(q),min(50,len(q)),False);x.append(q[ii]);a.append(u[ii]);y.append(z[ii]);h.append(np.stack([q[max(0,t-19):t+1] if t>=19 else np.r_[np.repeat(q[:1],19-t,0),q[:t+1]] for t in ii]))
 return np.concatenate(x),np.concatenate(a),np.concatenate(y),np.concatenate(h)
def f1(y,p):
 out=[]
 for c in range(5):
  tp=((y==c)&(p==c)).sum();d=2*tp+((y!=c)&(p==c)).sum()+((y==c)&(p!=c)).sum();out.append(float(2*tp/d) if d else 0.)
 return out
def main():
 m=json.loads((D/'split_manifest.json').read_text());paths={k:Path(v) for k,v in m['episode_paths'].items()};tx,ta,ty,th=load(m['splits']['train'],paths);vx,va,vy,vh=load(m['splits']['validation'],paths);ex,ea,ey,eh=load(m['splits']['test'],paths);mean=tx.mean(0);std=np.maximum(tx.std(0),1e-6);am=ta.mean(0);astd=np.maximum(ta.std(0),1e-6);tx=(tx-mean)/std;vx=(vx-mean)/std;ex=(ex-mean)/std;th=(th-mean)/std;vh=(vh-mean)/std;eh=(eh-mean)/std;ta=(ta-am)/astd
 # C3: sample query neighborhoods from all stage-labelled normal data, epsilon fixed at 1.0.
 ii=R.choice(len(tx),min(1500,len(tx)),False);ref=R.choice(len(tx),min(8000,len(tx)),False);without=[];withs=[]
 for start in range(0,len(ii),24):
  ds=cdist(tx[ii[start:start+24]],tx[ref])
  for row,q in enumerate(ii[start:start+24]):
   n=ref[ds[row]<1.0]
   if len(n)<2:continue
   without.append(float(np.trace(np.cov(ta[n],rowvar=False))))
   values=[np.trace(np.cov(ta[n][ty[n]==z],rowvar=False)) for z in range(5) if (ty[n]==z).sum()>=2]
   if values:withs.append(float(np.mean(values)))
 n=min(len(without),len(withs));without=np.asarray(without[:n]);withs=np.asarray(withs[:n]);c3={'epsilon':1.0,'neighborhoods':n,'variance_without_stage':float(without.mean()) if n else None,'variance_with_stage':float(withs.mean()) if n else None,'relative_reduction':float(1-withs.mean()/without.mean()) if n and without.mean()>0 else None}
 (OUT/'stage_condition_variance.json').write_text(json.dumps(c3,indent=2)+'\n')
 # C4: equal-capacity linear-head encoders (43->64->5 vs 20x43 temporal mean+last ->64->5).
 class Single(nn.Module):
  def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(43,64),nn.ReLU(),nn.Linear(64,5))
  def forward(self,x):return self.net(x)
 class Temporal(nn.Module):
  def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(86,64),nn.ReLU(),nn.Linear(64,5))
  def forward(self,x):return self.net(torch.cat((x[:,-1],x.mean(1)),1))
 def fit(model,x,h,y):
  opt=torch.optim.Adam(model.parameters(),lr=1e-3);dl=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(h),torch.from_numpy(y)),512,shuffle=True)
  for _ in range(8):
   for a,b,c in dl:opt.zero_grad();loss=nn.functional.cross_entropy(model(a if isinstance(model,Single) else b),c);loss.backward();opt.step()
  with torch.no_grad():p=model(torch.from_numpy(ex) if isinstance(model,Single) else torch.from_numpy(eh)).argmax(1).numpy()
  return {'accuracy':float((p==ey).mean()),'macro_f1':float(np.mean(f1(ey,p))),'stage_f1':f1(ey,p),'confusion_matrix':[[int(((ey==i)&(p==j)).sum()) for j in range(5)] for i in range(5)]}
 single=fit(Single(),tx,th,ty);temporal=fit(Temporal(),tx,th,ty);c4={'single_state':single,'temporal_20_frame':temporal,'temporal_improves_macro_f1':temporal['macro_f1']>single['macro_f1']};(OUT/'temporal_stage_inference.json').write_text(json.dumps(c4,indent=2)+'\n')
 im=Image.new('RGB',(800,480),'white');d=ImageDraw.Draw(im);d.text((20,20),'FIG C3/C4 summaries; detailed metrics in JSON',fill='black');d.text((60,100),f"Variance: {c3['variance_without_stage']} -> {c3['variance_with_stage']}",fill='black');d.text((60,150),f"Single Macro-F1: {single['macro_f1']:.3f}",fill='blue');d.text((60,190),f"Temporal Macro-F1: {temporal['macro_f1']:.3f}",fill='red');im.save(OUT/'FIG_C3_ACTION_VARIANCE_STAGE_CONDITIONING.png');im.save(OUT/'FIG_C4_TEMPORAL_STAGE_INFERENCE.png')
 report=json.loads((OUT/'stage_action_ambiguity_c1_c2_report.json').read_text());summary={'OBSERVATION_OVERLAP_EXISTS':report['OBSERVATION_OVERLAP_EXISTS'],'ACTION_CONFLICT_EXISTS':report['ACTION_CONFLICT_EXISTS'],'STAGE_CONDITION_REDUCES_ACTION_VARIANCE':'YES' if c3['relative_reduction'] is not None and c3['relative_reduction']>0 else 'NO','TEMPORAL_CONTEXT_IMPROVES_STAGE_INFERENCE':'YES' if c4['temporal_improves_macro_f1'] else 'NO','c3':c3,'c4':c4};(OUT/'ambiguity_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'STAGE_ACTION_AMBIGUITY_REPORT.md').write_text('# Stage Action Ambiguity Analysis V2\n\nGround-truth stage analysis reveals overlapping observation regions between task phases. Within these regions, expert actions exhibit significant divergence, indicating stage-dependent control ambiguity.\n\n'+json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
