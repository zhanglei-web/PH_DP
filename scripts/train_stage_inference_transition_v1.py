#!/usr/bin/env python3
"""Stage inference audit: frozen TCN architecture with optional transition loss."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mujoco_shared_control.stage.dataset import load_split, fit_normalization, StageWindowDataset, HISTORY
from mujoco_shared_control.stage.tcn import StageTCNV1

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/stage_dataset/stage_dataset_v1_hysteresis_20260817T052000Z'
MATRIX=ROOT/'outputs/stage_transition_v1/transition_matrix.npy'
OUT=ROOT/'outputs/stage_inference_transition_v1'
NAMES=('APPROACH','GRASP_LIFT','TRANSPORT','PLACE_RELEASE','RETREAT')
ALLOWED={(0,0),(0,1),(1,0),(1,1),(1,2),(2,0),(2,2),(2,3),(3,0),(3,3),(3,4),(4,4)}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def stage_metrics(y,p):
    cm=np.zeros((5,5),int)
    for a,b in zip(y,p): cm[int(a),int(b)]+=1
    per={}; f=[]
    for i,n in enumerate(NAMES):
        tp=cm[i,i]; precision=tp/max(1,cm[:,i].sum()); recall=tp/max(1,cm[i].sum()); f1=2*precision*recall/max(1e-12,precision+recall); f.append(f1)
        per[n]={'precision':float(precision),'recall':float(recall),'f1':float(f1),'support':int(cm[i].sum())}
    return {'accuracy':float(np.mean(y==p)),'macro_f1':float(np.mean(f)),'per_stage':per,'confusion_matrix':cm.tolist()}

def sequence_predictions(ds,p):
    out={e.episode_id:np.full(len(e.labels),-1,int) for e in ds.episodes}
    for k,v in enumerate(p): out[ds.episodes[int(ds.episode_indices[k])].episode_id][int(ds.time_indices[k])]=int(v)
    return out

def transition_audit(ds,p):
    seq=sequence_predictions(ds,p); total=correct=illegal=flicker=0; by={}
    for old,new in sorted({(int(a),int(b)) for e in ds.episodes for a,b in zip(e.labels[:-1],e.labels[1:]) if int(a)!=int(b)}):
        mask=[]; hit=0
        for e in ds.episodes:
            for t,(a,b) in enumerate(zip(e.labels[:-1],e.labels[1:]),1):
                if (int(a),int(b))==(old,new) and seq[e.episode_id][t]>=0:
                    total+=1; hit+=int(seq[e.episode_id][t]==new); mask.append(t)
        by[f'{old}->{new}']={'GT_EVENT_COUNT':len(mask),'IMMEDIATE_CORRECT_COUNT':hit,'IMMEDIATE_CORRECT_RATE':hit/max(1,len(mask))}
    delays=[]
    for e in ds.episodes:
        pred=seq[e.episode_id]
        for t in range(1,len(e.labels)):
            if pred[t-1]>=0 and pred[t]>=0:
                if (int(pred[t-1]),int(pred[t])) not in ALLOWED: illegal+=1
                if t+1<len(pred) and pred[t-1]==pred[t+1]!=pred[t] and (int(e.labels[t-1]),int(e.labels[t+1]))==(int(e.labels[t-1]),int(e.labels[t-1])): flicker+=1
            if e.labels[t-1]!=e.labels[t] and pred[t]>=0:
                target=int(e.labels[t]); found=None
                for j in range(t,min(len(pred),t+10)):
                    if pred[j]==target: found=j;break
                if found is not None: delays.append(found-t)
    return {'by_transition':by,'transition_detection_rate':correct/max(1,total),'transition_delay_frames':{'mean':float(np.mean(delays)) if delays else None,'p95':float(np.quantile(delays,.95)) if delays else None},'PREDICTED_ILLEGAL_TRANSITION_COUNT':illegal,'STAGE_FLICKER_COUNT':flicker,'STAGE_FLICKER_RATE':flicker/max(1,total)}

@torch.no_grad()
def predict(model,ds,device,batch):
    model.eval(); out=[]
    for windows,_,_ in DataLoader(ds,batch_size=batch,shuffle=False): out.extend(model(windows.to(device)).argmax(-1).cpu().numpy())
    return np.asarray(out,int)

def train_model(name, train_ds, val_ds, test_ds, norm, T, cfg, lam, device, out):
    seed_all(cfg['seed']); model=StageTCNV1().to(device); opt=AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']); criterion=nn.CrossEntropyLoss();
    tmat=torch.as_tensor(T,device=device)
    loader=DataLoader(train_ds,batch_size=cfg['batch'],shuffle=True,generator=torch.Generator().manual_seed(cfg['seed']))
    prev=np.full(len(train_ds),-1,np.int64); lookup={(int(train_ds.episode_indices[k]),int(train_ds.time_indices[k])):k for k in range(len(train_ds))}
    active=np.zeros(len(train_ds),bool)
    for k in range(len(train_ds)):
        key=(int(train_ds.episode_indices[k]),int(train_ds.time_indices[k])-1); prev[k]=lookup.get(key,-1)
        if prev[k]>=0: active[k]=train_ds.labels[k]!=train_ds.labels[prev[k]]
    calib_stage=[];calib_trans=[]; gen=torch.Generator(device=device).manual_seed(cfg['seed']+17)
    model.eval()
    for _ in range(100):
        ix=torch.randint(len(train_ds),(cfg['batch'],),device=device,generator=gen); valid=torch.as_tensor(active[ix.cpu().numpy()],device=device)
        cur_windows=torch.from_numpy(train_ds.windows[ix.cpu().numpy()]).to(device); labels=torch.as_tensor(train_ds.labels[ix.cpu().numpy()],device=device); logits=model(cur_windows); ls=criterion(logits,labels);
        pv=torch.as_tensor(prev[ix.cpu().numpy()],device=device); pv_safe=pv.clamp_min(0); prior=model.posterior(torch.from_numpy(train_ds.windows[pv_safe.cpu().numpy()]).to(device)).detach()@tmat.T; cur=model.posterior(cur_windows); lt=-(prior[valid]*torch.log(cur[valid].clamp_min(1e-8))).sum(-1).mean() if valid.any() else torch.zeros((),device=device); calib_stage.append(float(ls));calib_trans.append(float(lt))
    calibration={'L_STAGE_MEDIAN':float(np.median(calib_stage)),'L_TRANS_MEDIAN':float(np.median(calib_trans)),'LAMBDA_TRANSITION':lam,'TARGET_TRANSITION_LOSS_RATIO':.05,'TRANSITION_PREVIOUS_POSTERIOR_DETACHED':True}
    logs=[]; best=-1.; best_path=out/f'{name}_best_macro_f1.pt'
    for epoch in range(1,cfg['epochs']+1):
        model.train(); losses=[]; stages=[]; transes=[]
        for windows,labels,idx in loader:
            ix=idx.numpy(); logits=model(windows.to(device)); ls=criterion(logits,labels.to(device)); valid=torch.as_tensor(active[ix],device=device); pv=torch.as_tensor(prev[ix],device=device).clamp_min(0); prior=model.posterior(torch.from_numpy(train_ds.windows[pv.cpu().numpy()]).to(device)).detach()@tmat.T; cur=torch.softmax(logits,-1); lt=-(prior[valid]*torch.log(cur[valid].clamp_min(1e-8))).sum(-1).mean() if valid.any() else torch.zeros((),device=device); loss=ls+lam*lt; opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach()));stages.append(float(ls.detach()));transes.append(float(lt.detach()))
        vp=predict(model,val_ds,device,cfg['batch']); vm=stage_metrics(val_ds.labels,vp); va=transition_audit(val_ds,vp); rec={'epoch':epoch,'train_total_loss':float(np.mean(losses)),'train_stage_loss':float(np.mean(stages)),'train_transition_loss':float(np.mean(transes)), 'val_accuracy':vm['accuracy'],'val_macro_f1':vm['macro_f1'],'val_transition_detection_rate':va['transition_detection_rate'],'val_illegal_edges':va['PREDICTED_ILLEGAL_TRANSITION_COUNT']};logs.append(rec)
        if vm['macro_f1']>best:best=vm['macro_f1'];torch.save({'model':model.state_dict(),'epoch':epoch,'validation_macro_f1':best,'config':cfg,'lambda_transition':lam,'transition_matrix':T},best_path)
    model.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['model']); testp=predict(model,test_ds,device,cfg['batch']); result={'name':name,'checkpoint':str(best_path.resolve()),'best_val_macro_f1':best,'validation':stage_metrics(val_ds.labels,predict(model,val_ds,device,cfg['batch'])),'test':stage_metrics(test_ds.labels,testp),'transition_audit':transition_audit(test_ds,testp),'calibration':calibration,'logs':logs}; return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--output',type=Path,default=OUT);a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    dev=torch.device('cuda:0');torch.cuda.set_device(dev);a.output.mkdir(parents=True,exist_ok=True)
    if not MATRIX.exists(): raise FileNotFoundError(f'frozen transition matrix missing: {MATRIX}')
    T=np.asarray(np.load(MATRIX),np.float32)
    if T.shape!=(5,5) or not np.allclose(T.sum(1),1.,atol=1e-6): raise RuntimeError('invalid frozen transition matrix')
    train,val,test=load_split(DATA,'train'),load_split(DATA,'validation'),load_split(DATA,'test');norm=fit_normalization(train);td,vd,ted=[StageWindowDataset(x,norm) for x in (train,val,test)];cfg={'seed':42,'lr':3e-4,'weight_decay':1e-4,'batch':512,'epochs':a.epochs};
    np.save(a.output/'frozen_transition_matrix.npy',T);(a.output/'config.json').write_text(json.dumps({'architecture':'StageTCNV1 unchanged','data_root':str(DATA.resolve()),'transition_matrix_source':str(MATRIX.resolve()),'no_matrix_reestimation':True,'optimizer':'AdamW','training_config':cfg},indent=2)+'\n')
    # Calibration is performed per model from identical initialization; same fixed lambda is used for B.
    # Use a temporary baseline-equivalent model to obtain initial scale without changing persisted checkpoints.
    tmp=StageTCNV1().to(dev); crit=nn.CrossEntropyLoss(); stages=[];trans=[]; lookup={(int(td.episode_indices[k]),int(td.time_indices[k])):k for k in range(len(td))}; prev=np.asarray([lookup.get((int(td.episode_indices[k]),int(td.time_indices[k])-1),-1) for k in range(len(td))]); active=(prev>=0) & (td.labels != td.labels[np.maximum(prev,0)]); g=torch.Generator(device=dev).manual_seed(59)
    for _ in range(100):
        ix=torch.randint(len(td),(cfg['batch'],),device=dev,generator=g); cur_windows=torch.from_numpy(td.windows[ix.cpu().numpy()]).to(dev); labels=torch.as_tensor(td.labels[ix.cpu().numpy()],device=dev); logits=tmp(cur_windows);ls=crit(logits,labels); pv=torch.as_tensor(np.maximum(prev[ix.cpu().numpy()],0),device=dev);pr=tmp.posterior(torch.from_numpy(td.windows[pv.cpu().numpy()]).to(dev)).detach()@torch.as_tensor(T,device=dev).T;cur=torch.softmax(logits,-1);mask=torch.as_tensor(active[ix.cpu().numpy()],device=dev);lt=-(pr[mask]*torch.log(cur[mask].clamp_min(1e-8))).sum(-1).mean() if mask.any() else torch.zeros((),device=dev);stages.append(float(ls));trans.append(float(lt))
    lam=.05*float(np.median(np.asarray(stages))/(np.median(np.asarray(trans))+1e-12));(a.output/'lambda_calibration.json').write_text(json.dumps({'TRANSITION_CALIBRATION_BATCHES':100,'L_STAGE_MEDIAN':float(np.median(stages)),'L_TRANS_MEDIAN':float(np.median(trans)),'LAMBDA_TRANSITION':lam,'TARGET_TRANSITION_LOSS_RATIO':.05,'TRANSITION_PREVIOUS_POSTERIOR_DETACHED':True},indent=2)+'\n')
    A=train_model('tcn_a',td,vd,ted,norm,T,cfg,0.,dev,a.output);B=train_model('tcn_b',td,vd,ted,norm,T,cfg,lam,dev,a.output);(a.output/'validation_comparison.json').write_text(json.dumps({'TCN_A':A,'TCN_B':B,'CUDA_TRAINING_VALID':True,'TRANSITION_MATRIX_REESTIMATED':False},indent=2)+'\n');(a.output/'summary.md').write_text(json.dumps({'TCN_A':A['test'],'TCN_B':B['test'],'lambda_transition':lam,'STEP2_READY':False},indent=2)+'\n')
if __name__=='__main__':main()
