#!/usr/bin/env python3
"""Fixed-split 48D current-ActiveStage Unified BC training and offline audit."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import h5py,numpy as np,torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from mujoco_shared_control.experts.recovery_bc import RecoveryBCPolicy

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL';MEM=ROOT/'outputs/experiments/unified_recovery_bc_v1/run_20260818T_UNIFIED_RECOVERY_BC_FORMAL';OUT=ROOT/'outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL';SCEN=('NORMAL','GRASP_RECOVERY','TRANSPORT_DROP','PLACE_RECOVERY');SEED=20260818
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def write(p,rows):
 k=sorted({z for x in rows for z in x});
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=k);w.writeheader();w.writerows(rows)
class Data(Dataset):
 def __init__(self,paths,ids):
  ss=[];aa=[];pp=[];cc=[];self.ids=ids
  for e in ids:
   with h5py.File(paths[e],'r') as f:s=f['full_physical_state'][:].astype('f');a=f['raw_pilot_action'][:].astype('f');p=f['active_phase'][:].astype('i');c=str(f.attrs['trajectory_type']);n=f['next_full_physical_state'][:].astype('f')
   if s.shape!=(len(p),43) or a.shape!=(len(p),7) or np.any((p<0)|(p>4)) or not np.allclose(n[:-1],s[1:],atol=1e-7):raise ValueError(e)
   ss.append(s);aa.append(a);pp.append(p);cc.append(np.full(len(s),c,object))
  self.s=np.concatenate(ss);self.a=np.concatenate(aa);self.p=np.concatenate(pp);self.c=np.concatenate(cc)
 def __len__(self):return len(self.s)
 def __getitem__(self,i):return self.s[i],self.p[i],self.a[i]
 def weights(self):
  n={x:int(np.sum(self.c==x)) for x in SCEN};return torch.from_numpy(np.asarray([.25/n[x] for x in self.c],np.float64))
def inp(s,p,mean,std):
 return torch.cat(((s-mean)/std,nn.functional.one_hot(p.long(),5).float()),-1)
def los(m,s,p,a,mean,std):
 motion,logit=m(inp(s,p,mean,std));ml=nn.functional.mse_loss(motion,a[:,:6]);gl=nn.functional.binary_cross_entropy_with_logits(logit,(a[:,6]>0).float());return ml+gl,ml,gl,logit,motion
@torch.no_grad()
def metrics(m,d,mean,std):
 dev=next(m.parameters()).device;mean=torch.from_numpy(mean).to(dev);std=torch.from_numpy(std).to(dev);P=[];A=[];G=[];PH=[];L=[];ML=[];GL=[];m.eval()
 for s,p,a in DataLoader(d,batch_size=1024):
  l,ml,gl,g,z=los(m,s.to(dev),p.to(dev),a.to(dev),mean,std);P.append(z.cpu().numpy());A.append(a.numpy());G.append(g.cpu().numpy());PH.append(p.numpy());L.append(float(l));ML.append(float(ml));GL.append(float(gl))
 z=np.concatenate(P);a=np.concatenate(A);q=(np.concatenate(G)>=0).astype(int);y=(a[:,6]>0).astype(int);ph=np.concatenate(PH);e=z-a[:,:6]
 def one(mask):
  ee=e[mask];qq=q[mask];yy=y[mask]
  def f(v):
   tp=np.sum((qq==v)&(yy==v));fp=np.sum((qq==v)&(yy!=v));fn=np.sum((qq!=v)&(yy==v));pr=tp/(tp+fp) if tp+fp else 0;re=tp/(tp+fn) if tp+fn else 0;return {'precision':float(pr),'recall':float(re),'f1':float(2*pr*re/(pr+re) if pr+re else 0)}
  return {'motion_mse':float(np.mean(ee**2)),'motion_mae':float(np.mean(abs(ee))),'translation_mae':float(np.mean(abs(ee[:,:3]))),'rotation_mae':float(np.mean(abs(ee[:,3:]))),'gripper_accuracy':float(np.mean(qq==yy)),'gripper_error':float(np.mean(qq!=yy)),'OPEN':f(1),'CLOSE':f(0)}
 return {'total_loss':float(np.mean(L)),'motion_loss':float(np.mean(ML)),'gripper_loss':float(np.mean(GL)),'overall':one(np.ones(len(d),bool)),'by_scenario':{x:one(d.c==x) for x in SCEN},'by_stage':{str(i):one(ph==i) for i in range(5)}}
def audit(paths,ids):
 mat={x:np.zeros((5,5),int) for x in SCEN};occ={x:np.zeros(5,int) for x in SCEN};reg={'1->0':0,'2->0':0,'3->0':0};align=0
 for e in ids:
  with h5py.File(paths[e],'r') as f:p=f['active_phase'][:].astype(int);s=f['full_physical_state'][:];n=f['next_full_physical_state'][:];c=str(f.attrs['trajectory_type'])
  occ[c]+=np.bincount(p,minlength=5);align+=int(not np.allclose(n[:-1],s[1:],atol=1e-7));mat[c]+=np.bincount(p[:-1]*5+p[1:],minlength=25).reshape(5,5)
  for old in (1,2,3):reg[f'{old}->0']+=int(np.sum((p[:-1]==old)&(p[1:]==0)))
 return {'status':'PASS' if align==0 and reg['1->0']>=1000 and reg['2->0']>=1000 and reg['3->0']>=1000 else 'FAIL','stage_field':'active_phase','state_field':'full_physical_state','action_field':'raw_pilot_action','alignment':'obs[t], active_phase[t], action[t]; next_state[t]=state[t+1]','transition_matrix':{k:v.tolist() for k,v in mat.items()},'occupancy':{k:v.tolist() for k,v in occ.items()},'regression_counts':reg,'required_regressions_at_least_one_per_recovery_episode':True,'alignment_failures':align}
def init(allowed=SCEN):
 OUT.mkdir(parents=True,exist_ok=True);source=json.loads((MEM/'split_manifest.json').read_text());man=json.loads((DATA/'dataset_manifest.json').read_text());ids={x['episode_id'] for x in man['episodes']}
 if any(i not in ids for v in source['splits'].values() for i in v):raise RuntimeError('STOP split not reusable')
 source['splits']={k:[i for i in v if h5py.File(source['episode_paths'][i],'r').attrs['trajectory_type'] in allowed] for k,v in source['splits'].items()}
 if allowed==('NORMAL','PLACE_RECOVERY') and {k:len(v) for k,v in source['splits'].items()}!={'train':1600,'validation':200,'test':200}:raise RuntimeError('STOP normal/place split mismatch')
 dump(OUT/'split_manifest.json',source);dump(OUT/'split_reuse_audit.json',{'status':'PASS','source':str((MEM/'split_manifest.json').resolve()),'exact_episode_ids_reused':True,'counts':{k:len(v) for k,v in source['splits'].items()}});dump(OUT/'active_stage_audit.json',audit(source['episode_paths'],[i for v in source['splits'].values() for i in v]))
 tr=Data(source['episode_paths'],source['splits']['train']);mean=tr.s.mean(0,dtype=np.float64).astype('f');std=np.maximum(tr.s.std(0,dtype=np.float64),1e-6).astype('f');mean[42]=0;std[42]=1;np.savez(OUT/'normalizer.npz',physical_mean=mean,physical_std=std,stage_onehot_normalized=False)
 m=RecoveryBCPolicy(48);o=AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);torch.save({'epoch':0,'best':float('inf'),'best_epoch':0,'patience':0,'model':m.state_dict(),'optimizer':o.state_dict(),'history':[]},OUT/'training_state.pt');dump(OUT/'training_config.json',{'architecture':'48->256->256->256 ReLU; tanh motion6 + binary gripper logit','input':'normalized physical43 + raw CURRENT active_phase onehot5','not_cumulative_milestones':True,'allowed_scenarios':allowed,'optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'batch_size':1024,'max_epochs':50,'patience':5,'seed':SEED,'loss':'motion MSE + BCE','sampler':'scenario-balanced equal per allowed scenario','checkpoint_selection':'validation total loss only'})
def main():
 global OUT,SCEN
 ap=argparse.ArgumentParser();ap.add_argument('--init',action='store_true');ap.add_argument('--audit-only',action='store_true');ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--normal-place',action='store_true');a=ap.parse_args()
 if a.normal_place:OUT=ROOT/'outputs/experiments/unified_normal_place_stageaware_bc_v1/run_20260818T_UNIFIED_NORMAL_PLACE_STAGEAWARE_BC_FORMAL';SCEN=('NORMAL','PLACE_RECOVERY')
 if a.audit_only:
  split=json.loads((OUT/'split_manifest.json').read_text());dump(OUT/'active_stage_audit.json',audit(split['episode_paths'],[i for v in split['splits'].values() for i in v]));return
 if a.init:init(SCEN)
 split=json.loads((OUT/'split_manifest.json').read_text());paths=split['episode_paths'];tr=Data(paths,split['splits']['train']);va=Data(paths,split['splits']['validation']);te=Data(paths,split['splits']['test']);n=np.load(OUT/'normalizer.npz');mean,std=n['physical_mean'],n['physical_std'];state=torch.load(OUT/'training_state.pt',map_location='cpu',weights_only=False);m=RecoveryBCPolicy(48);m.load_state_dict(state['model']);o=AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);o.load_state_dict(state['optimizer']);mt=torch.from_numpy(mean);st=torch.from_numpy(std)
 for _ in range(a.epochs):
  if state['epoch']>=50 or state['patience']>=5:break
  ep=state['epoch']+1;m.train();ls=[];ms=[];gs=[];loader=DataLoader(tr,batch_size=1024,sampler=WeightedRandomSampler(tr.weights(),len(tr),replacement=True,generator=torch.Generator().manual_seed(SEED+ep)))
  for s,p,y in loader:
   o.zero_grad(set_to_none=True);l,ml,gl,_,_=los(m,s,p,y,mt,st);l.backward();o.step();ls+=[float(l.detach())];ms+=[float(ml.detach())];gs+=[float(gl.detach())]
  v=metrics(m,va,mean,std);r={'epoch':ep,'train_total_loss':float(np.mean(ls)),'train_motion_loss':float(np.mean(ms)),'train_gripper_loss':float(np.mean(gs)),'val_total_loss':v['total_loss'],'val_motion_loss':v['motion_loss'],'val_gripper_loss':v['gripper_loss'],'val_gripper_accuracy':v['overall']['gripper_accuracy']};state['history'].append(r);state['epoch']=ep;print(json.dumps(r),flush=True)
  if v['total_loss']<state['best']:state['best']=v['total_loss'];state['best_epoch']=ep;state['patience']=0;torch.save({'model':m.state_dict(),'physical_mean':mean,'physical_std':std,'best_epoch':ep,'best_val_total_loss':v['total_loss'],'val_motion_loss':v['motion_loss'],'val_gripper_accuracy':v['overall']['gripper_accuracy']},OUT/'best_val.pt')
  else:state['patience']+=1
 state['model']=m.state_dict();state['optimizer']=o.state_dict();torch.save(state,OUT/'training_state.pt');write(OUT/'training_history.csv',state['history'])
 if state['epoch']>=50 or state['patience']>=5:
  z=torch.load(OUT/'best_val.pt',map_location='cpu',weights_only=False);m=RecoveryBCPolicy(48);m.load_state_dict(z['model']);q=metrics(m,te,mean,std);dump(OUT/'offline_test_summary.json',q)
  def flat(label,x):return {'Group':label,'Motion MSE':x['motion_mse'],'Motion MAE':x['motion_mae'],'Translation MAE':x['translation_mae'],'Rotation MAE':x['rotation_mae'],'Gripper Accuracy':x['gripper_accuracy'],'OPEN F1':x['OPEN']['f1'],'CLOSE F1':x['CLOSE']['f1']}
  write(OUT/'offline_test_summary.csv',[flat('OVERALL',q['overall'])]);write(OUT/'offline_test_by_scenario.csv',[flat(x,q['by_scenario'][x]) for x in SCEN]);write(OUT/'offline_test_by_stage.csv',[flat(('APPROACH','GRASP_LIFT','TRANSPORT','PLACE_RELEASE','RETREAT')[i],q['by_stage'][str(i)]) for i in range(5)]);print(json.dumps({'complete':True,'best_epoch':state['best_epoch']}))
if __name__=='__main__':main()
