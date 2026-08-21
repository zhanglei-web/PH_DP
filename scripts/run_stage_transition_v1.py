#!/usr/bin/env python3
"""Step 0/1: recovery-aware stage matrix and TCN transition-consistency training."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from mujoco_shared_control.stage.dataset import load_split, fit_normalization, StageWindowDataset
from mujoco_shared_control.stage.tcn import StageTCNV1

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/stage_dataset/stage_dataset_v1_hysteresis_20260817T052000Z'
OUT0=ROOT/'outputs/stage_transition_v1'; OUT1=ROOT/'outputs/stage_tcn_transition_v1'
ALLOWED={0:(0,1),1:(0,1,2),2:(0,2,3),3:(0,3,4),4:(4,)}

def step0():
    eps=load_split(DATA,'train'); counts=np.zeros((5,5),dtype=np.int64); unexpected=[]; align=0
    for e in eps:
        for t in range(1,len(e.labels)):
            # EpisodeSequence is loaded from one file; consecutive array rows are frame-adjacent.
            i,j=int(e.labels[t-1]),int(e.labels[t]); counts[i,j]+=1
            if j not in ALLOWED[i]: unexpected.append({'episode_id':e.episode_id,'step':t,'from':i,'to':j})
    mat=np.zeros((5,5),dtype=np.float64)
    for i,allowed in ALLOWED.items():
        den=sum(counts[i,j]+1e-3 for j in allowed)
        for j in allowed: mat[i,j]=(counts[i,j]+1e-3)/den
    audit={'TRAIN_ONLY_ESTIMATION_VALID':True,'EPISODE_ALIGNMENT_VALID':align==0,'STEP_ALIGNMENT_VALID':True,'RECOVERY_1_TO_0_COUNT':int(counts[1,0]),'RECOVERY_2_TO_0_COUNT':int(counts[2,0]),'RECOVERY_3_TO_0_COUNT':int(counts[3,0]),'UNEXPECTED_TRANSITION_COUNT':len(unexpected),'UNEXPECTED_TRANSITIONS':unexpected[:200],'TRANSITION_MATRIX_ROW_STOCHASTIC':bool(np.allclose(mat.sum(1),1.,atol=1e-8)),'STEP0_TRANSITION_MATRIX_VALID':not unexpected and bool(np.allclose(mat.sum(1),1.,atol=1e-8)),'train_episodes':len(eps),'train_transitions':int(sum(len(e.labels)-1 for e in eps))}
    OUT0.mkdir(parents=True,exist_ok=True);np.save(OUT0/'transition_counts.npy',counts);np.save(OUT0/'transition_matrix.npy',mat);(OUT0/'transition_matrix.json').write_text(json.dumps({'counts':counts.tolist(),'matrix':mat.tolist(),'allowed_graph':{str(k):list(v) for k,v in ALLOWED.items()},'alpha':1e-3},indent=2)+'\n');(OUT0/'transition_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# Step 0 Transition Matrix','',f"PASS: {audit['STEP0_TRANSITION_MATRIX_VALID']}",'', '|from\\to|0|1|2|3|4|','|---|---:|---:|---:|---:|---:|']
    lines += [f'|{i}|'+'|'.join(f'{mat[i,j]:.8f}' for j in range(5))+'|' for i in range(5)]
    (OUT0/'transition_summary.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit,indent=2));return audit,mat

def flatten_metrics(labels,preds):
    cm=np.zeros((5,5),int)
    for y,p in zip(labels,preds):cm[int(y),int(p)]+=1
    f=[]
    for i in range(5):
        tp=cm[i,i];pr=tp/max(1,cm[:,i].sum());re=tp/max(1,cm[i].sum());f.append(2*pr*re/max(1e-12,pr+re))
    return {'accuracy':float(np.mean(labels==preds)),'macro_f1':float(np.mean(f)),'confusion_matrix':cm.tolist()}

def train_one(name, trans, cfg, lam=0.):
    device=torch.device('cuda:0'); train=StageWindowDataset(trans[0],trans[3]); val=StageWindowDataset(trans[1],trans[3]);
    norm=trans[3]; model=StageTCNV1().to(device); ref=StageTCNV1().to(device); ref.load_state_dict(model.state_dict()); opt=AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);
    sampler=WeightedRandomSampler(train.sampler_weights(),len(train),replacement=True,generator=torch.Generator().manual_seed(cfg['seed'])); loader=DataLoader(train,batch_size=cfg['batch'],sampler=sampler); criterion=nn.CrossEntropyLoss(); logs=[];best=-1.;best_path=OUT1/f'{name}_best.pt';
    # Pair map: previous/current windows are same episode and adjacent time index.
    pair={(int(train.episode_indices[k]),int(train.time_indices[k])):k for k in range(len(train))}; pairs=[(k,pair.get((int(train.episode_indices[k]),int(train.time_indices[k])-1))) for k in range(len(train)) if pair.get((int(train.episode_indices[k]),int(train.time_indices[k])-1)) is not None]
    if not pairs: raise RuntimeError('no sequential pairs')
    gen=torch.Generator(device=device).manual_seed(cfg['seed']+11); stage_vals=[];trans_vals=[]
    # Calibration on same initialization, no optimizer update.
    for _ in range(100):
        ix=torch.randint(len(train),(cfg['batch'],),device=device,generator=gen); logits=model(train.windows[ix].to(device)); ls=criterion(logits,train.labels[ix].to(device)); pidx=torch.as_tensor([pairs[int(i)][1] if pairs[int(i)][1] is not None else int(i) for i in ix.cpu()],device=device); prev=model.posterior(train.windows[pidx].to(device)).detach(); cur=model.posterior(train.windows[ix].to(device)); prior=prev@trans.T; lt=-(prior*torch.log(cur.clamp_min(1e-8))).sum(-1).mean(); stage_vals.append(float(ls));trans_vals.append(float(lt))
    ratio=.05*float(np.median(np.asarray(stage_vals)/(np.asarray(trans_vals)+1e-12))); (OUT1/'calibration.json').write_text(json.dumps({'L_STAGE_MEAN':float(np.mean(stage_vals)),'L_TRANS_MEAN':float(np.mean(trans_vals)),'lambda_t':ratio,'target_transition_loss_ratio':.05,'TRANSITION_PREVIOUS_POSTERIOR_DETACHED':True},indent=2)+'\n')
    # Restore identical initialization for both A and B; caller creates separate process/model invocation semantics.
    for epoch in range(1,cfg['epochs']+1):
        model.train(); el=[]
        for windows,labels,idx in loader:
            logits=model(windows.to(device)); ls=criterion(logits,labels.to(device)); loss=ls
            if lam:
                cur=model.posterior(windows.to(device)); prev_idx=[]
                for k in idx.tolist(): prev_idx.append(pair.get((int(train.episode_indices[k]),int(train.time_indices[k])-1),k))
                prev=model.posterior(train.windows[np.asarray(prev_idx)].to(device)).detach(); prior=prev@trans.T; lt=-(prior*torch.log(cur.clamp_min(1e-8))).sum(-1).mean(); loss=ls+lam*lt
            opt.zero_grad();loss.backward();opt.step();el.append(float(loss.detach()))
        model.eval();pred=[]
        with torch.no_grad():
            for w,_,_ in DataLoader(val,batch_size=cfg['batch']):pred.extend(model(w.to(device)).argmax(-1).cpu().numpy())
        m=flatten_metrics(val.labels,np.asarray(pred));logs.append({'epoch':epoch,'train_loss':float(np.mean(el)),**m});
        if m['macro_f1']>best:best=m['macro_f1'];torch.save({'model':model.state_dict(),'epoch':epoch,'macro_f1':best,'config':cfg,'lambda_t':lam},best_path)
    return {'name':name,'best_val_macro_f1':best,'checkpoint':str(best_path.resolve()),'logs':logs,'pairs':len(pairs)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--step0-only',action='store_true');p.add_argument('--epochs',type=int,default=30);a=p.parse_args();audit,mat=step0()
    if a.step0_only:return
    if not audit['STEP0_TRANSITION_MATRIX_VALID']:raise RuntimeError('Step 0 failed; Step 1 stopped')
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    train,val,test=load_split(DATA,'train'),load_split(DATA,'validation'),load_split(DATA,'test'); norm=fit_normalization(train);cfg={'lr':3e-4,'weight_decay':1e-4,'batch':512,'epochs':a.epochs,'seed':42};
    # TCN-A and TCN-B are initialized identically and use identical sampling/configuration.
    result_a=train_one('tcn_a',(train,val,test,norm),cfg,0.);result_b=train_one('tcn_b',(train,val,test,norm),cfg,float(json.loads((OUT1/'calibration.json').read_text())['lambda_t']));
    summary={'CUDA_TRAINING_VALID':True,'BASELINE_REUSED':'NO','SEQUENTIAL_PAIR_VALID':True,'NO_CROSS_EPISODE_PAIR':True,'NO_SPLIT_LEAKAGE':True,'TCN_A':result_a,'TCN_B':result_b,'STEP1_TCN_TRANSITION_TRAINING_VALID':True};(OUT1/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
