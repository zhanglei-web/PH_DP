#!/usr/bin/env python3
"""Resumable fixed-configuration training for the E2 Unified Recovery BC."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import h5py,numpy as np,torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from mujoco_shared_control.experts.recovery_bc import RecoveryBCPolicy

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL';OUT=ROOT/'outputs/experiments/unified_recovery_bc_v1/run_20260818T_UNIFIED_RECOVERY_BC_FORMAL';SCEN=('NORMAL','GRASP_RECOVERY','TRANSPORT_DROP','PLACE_RECOVERY');SEED=20260818
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def write(p,rows):
 k=sorted({z for x in rows for z in x});
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(rows)
class Data(Dataset):
 def __init__(self,paths,ids):
  ss=[];aa=[];cc=[];self.ids=ids
  for e in ids:
   with h5py.File(paths[e],'r') as f:s=f['full_physical_state'][:].astype('f');a=f['raw_pilot_action'][:].astype('f');c=str(f.attrs['trajectory_type'])
   if s.shape[1:]!=(43,) or a.shape[1:]!=(7,) or not np.isfinite(s).all() or not np.isfinite(a).all():raise ValueError(e)
   ss.append(s);aa.append(a);cc.append(np.full(len(s),c,object))
  self.s=np.concatenate(ss);self.a=np.concatenate(aa);self.c=np.concatenate(cc)
 def __len__(self):return len(self.s)
 def __getitem__(self,i):return self.s[i],self.a[i]
 def weights(self):
  counts={x:int(np.sum(self.c==x)) for x in SCEN};return torch.from_numpy(np.asarray([.25/counts[x] for x in self.c],np.float64))
def los(m,x,a,mean,std):
 motion,logit=m((x-mean)/std);ml=nn.functional.mse_loss(motion,a[:,:6]);gl=nn.functional.binary_cross_entropy_with_logits(logit,(a[:,6]>0).float());return ml+gl,ml,gl,logit,motion
@torch.no_grad()
def metrics(m,d,mean,std):
 dev=next(m.parameters()).device;mean=torch.from_numpy(mean).to(dev);std=torch.from_numpy(std).to(dev); loader=DataLoader(d,batch_size=1024);P=[];A=[];L=[];ML=[];GL=[];G=[];m.eval()
 for x,a in loader:
  l,ml,gl,g,p=los(m,x.to(dev),a.to(dev),mean,std);P.append(p.cpu().numpy());A.append(a.numpy());G.append(g.cpu().numpy());L.append(float(l));ML.append(float(ml));GL.append(float(gl))
 p=np.concatenate(P);a=np.concatenate(A);q=(np.concatenate(G)>=0).astype(int);y=(a[:,6]>0).astype(int);e=p-a[:,:6]
 def one(mask):
  qq=q[mask];yy=y[mask];ee=e[mask]
  def f(v):
   tp=np.sum((qq==v)&(yy==v));fp=np.sum((qq==v)&(yy!=v));fn=np.sum((qq!=v)&(yy==v));pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0;return {'precision':float(pr),'recall':float(re),'f1':float(2*pr*re/(pr+re) if pr+re else 0)}
  return {'motion_mse':float(np.mean(ee**2)),'motion_mae':float(np.mean(abs(ee))),'translation_mae':float(np.mean(abs(ee[:,:3]))),'rotation_mae':float(np.mean(abs(ee[:,3:]))),'gripper_accuracy':float(np.mean(qq==yy)),'OPEN':f(1),'CLOSE':f(0)}
 return {'total_loss':float(np.mean(L)),'motion_loss':float(np.mean(ML)),'gripper_loss':float(np.mean(GL)),'overall':one(np.ones(len(d),bool)),'by_scenario':{x:one(d.c==x) for x in SCEN}}
def init():
 OUT.mkdir(parents=True,exist_ok=True);man=json.loads((DATA/'dataset_manifest.json').read_text());paths={x['episode_id']:x['path'] for x in man['episodes']};split={'split_unit':'episode','seed':SEED,'splits':{k:[] for k in ('train','validation','test')},'episode_paths':paths}
 for c in SCEN:
  ids=[x['episode_id'] for x in man['episodes'] if x['trajectory_type']==c];rng=np.random.default_rng(SEED+SCEN.index(c));rng.shuffle(ids);split['splits']['train']+=ids[:800];split['splits']['validation']+=ids[800:900];split['splits']['test']+=ids[900:]
 dump(OUT/'split_manifest.json',split);tr=Data(paths,split['splits']['train']);mean=tr.s.mean(0,dtype=np.float64).astype('f');std=np.maximum(tr.s.std(0,dtype=np.float64),1e-6).astype('f');mean[42]=0;std[42]=1;np.savez(OUT/'normalizer.npz',mean=mean,std=std);m=RecoveryBCPolicy();o=AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);state={'epoch':0,'best':float('inf'),'best_epoch':0,'patience':0,'model':m.state_dict(),'optimizer':o.state_dict(),'history':[]};torch.save(state,OUT/'training_state.pt');dump(OUT/'training_config.json',{'seed':SEED,'architecture':'43->256->256->256 ReLU, tanh motion6 + binary gripper logit','optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'batch_size':1024,'max_epochs':50,'patience':5,'loss':'motion MSE + 1.0*gripper BCE','sampler':'scenario-balanced 25% each by transition','checkpoint_selection':'validation total loss only','prohibited_inputs':['stage','failure label','milestones']})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--init',action='store_true');ap.add_argument('--epochs',type=int,default=5);a=ap.parse_args()
 if a.init:init()
 if not (OUT/'training_state.pt').exists():raise SystemExit('run --init first')
 split=json.loads((OUT/'split_manifest.json').read_text());paths=split['episode_paths'];tr=Data(paths,split['splits']['train']);va=Data(paths,split['splits']['validation']);te=Data(paths,split['splits']['test']);norm=np.load(OUT/'normalizer.npz');mean,std=norm['mean'],norm['std'];state=torch.load(OUT/'training_state.pt',map_location='cpu',weights_only=False);m=RecoveryBCPolicy();m.load_state_dict(state['model']);o=AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);o.load_state_dict(state['optimizer']);mt=torch.from_numpy(mean);st=torch.from_numpy(std)
 for _ in range(a.epochs):
  if state['epoch']>=50 or state['patience']>=5:break
  epoch=state['epoch']+1;m.train();loader=DataLoader(tr,batch_size=1024,sampler=WeightedRandomSampler(tr.weights(),len(tr),replacement=True,generator=torch.Generator().manual_seed(SEED+epoch)));ls=[];ms=[];gs=[]
  for x,y in loader:
   o.zero_grad(set_to_none=True);l,ml,gl,_,_=los(m,x,y,mt,st);l.backward();o.step();ls+=[float(l.detach())];ms+=[float(ml.detach())];gs+=[float(gl.detach())]
  v=metrics(m,va,mean,std);rec={'epoch':epoch,'train_total_loss':float(np.mean(ls)),'train_motion_loss':float(np.mean(ms)),'train_gripper_loss':float(np.mean(gs)),'val_total_loss':v['total_loss'],'val_motion_loss':v['motion_loss'],'val_gripper_loss':v['gripper_loss'],'val_gripper_accuracy':v['overall']['gripper_accuracy']};state['history'].append(rec);state['epoch']=epoch
  if v['total_loss']<state['best']:
   state['best']=v['total_loss'];state['best_epoch']=epoch;state['patience']=0;torch.save({'model':m.state_dict(),'normalization_mean':mean,'normalization_std':std,'best_epoch':epoch,'best_val_total_loss':v['total_loss'],'val_motion_loss':v['motion_loss'],'val_gripper_accuracy':v['overall']['gripper_accuracy']},OUT/'best_val.pt')
  else:state['patience']+=1
  print(json.dumps(rec),flush=True)
 state['model']=m.state_dict();state['optimizer']=o.state_dict();torch.save(state,OUT/'training_state.pt');write(OUT/'training_history.csv',state['history'])
 if state['patience']>=5 or state['epoch']>=50:
  z=torch.load(OUT/'best_val.pt',map_location='cpu',weights_only=False);m=RecoveryBCPolicy();m.load_state_dict(z['model']);q=metrics(m,te,mean,std);dump(OUT/'offline_test_summary.json',q);rows=[{'Scenario':'OVERALL',**q['overall']}] + [{'Scenario':x,**q['by_scenario'][x]} for x in SCEN];write(OUT/'offline_test_summary.csv',[{'Scenario':r['Scenario'],'Motion MSE':r['motion_mse'],'Motion MAE':r['motion_mae'],'Translation MAE':r['translation_mae'],'Rotation MAE':r['rotation_mae'],'Gripper Accuracy':r['gripper_accuracy'],'OPEN Precision':r['OPEN']['precision'],'OPEN Recall':r['OPEN']['recall'],'OPEN F1':r['OPEN']['f1'],'CLOSE Precision':r['CLOSE']['precision'],'CLOSE Recall':r['CLOSE']['recall'],'CLOSE F1':r['CLOSE']['f1']} for r in rows]);print(json.dumps({'complete':True,'epoch':state['epoch'],'best_epoch':state['best_epoch']}))
if __name__=='__main__':main()
