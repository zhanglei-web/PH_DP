#!/usr/bin/env python3
"""Frozen TCN-A compatibility audit using its causal online wrapper semantics."""
from __future__ import annotations
import json, hashlib
from collections import deque
from pathlib import Path
import h5py, numpy as np, torch
from mujoco_shared_control.stage.tcn import StageTCNV1

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/recovery_stage_dp_dataset/recovery_stage_dp_v1_20260820T_FORMAL_CORRECTED'
CKPT=ROOT/'outputs/stage_inference_transition_v1/tcn_a_best_macro_f1.pt'
NORM=ROOT/'outputs/stage_tcn/stage_tcn_v1_hysteresis_20260817T070000Z/normalization_stats.npz'
OUT=ROOT/'outputs/recovery_stage_dp_training/tcn_online_compatibility_audit_20260820'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def feat(s,a):
    ee=s[14:17]; obj=s[22:25]; goal=s[29:32]; eo=obj-ee; og=goal-obj
    return np.r_[a,eo,np.linalg.norm(eo),og,np.linalg.norm(og),np.linalg.norm(goal-ee),s[21],s[42],obj[2]].astype('f4')
def f1(y,p):
    out=[]
    for c in range(5):
        tp=np.sum((y==c)&(p==c)); den=2*tp+np.sum((y!=c)&(p==c))+np.sum((y==c)&(p!=c));out.append(float(2*tp/den) if den else 0.)
    return out
def main():
 if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback forbidden')
 dev=torch.device('cuda:0');torch.cuda.set_device(dev);OUT.mkdir(parents=True,exist_ok=True)
 with np.load(NORM) as n: mean=n['mean'].astype('f4');std=n['std'].astype('f4')
 m=StageTCNV1().to(dev).eval();m.load_state_dict(torch.load(CKPT,map_location=dev,weights_only=False)['model']);m.requires_grad_(False)
 split=json.loads((DATA/'split_manifest.json').read_text()); paths={k:Path(v) for k,v in split['episode_paths'].items()}; rows=[]; allp={k:[] for k in split['splits']};ally={k:[] for k in split['splits']};trans=[]
 for sp,ids in split['splits'].items():
  for eid in ids:
   raw=DATA/'raw_rollouts'/paths[eid].relative_to(DATA/'episodes')
   with h5py.File(paths[eid],'r') as c,h5py.File(raw,'r') as r:
    rt=r['timestep_raw'][:]; idx={int(t):i for i,t in enumerate(rt)}; ct=c['timestep_raw'][:]; state=c['full_physical_state'][:];ns=c['next_full_physical_state'][:];act=c['executed_action'][:];ph=c['active_phase'][:];one=c['stage_onehot'][:];inj=c['injection_active'][:]; rs=r['full_physical_state'][:];rna=r['next_full_physical_state'][:];ra=r['executed_action'][:];rp=r['active_phase'][:];ro=r['stage_onehot'][:];ri=r['injection_active'][:]
    for i,t in enumerate(ct):
     j=idx.get(int(t),-1)
     if j<0 or not np.allclose(state[i],rs[j],atol=1e-6) or not np.allclose(ns[i],rna[j],atol=1e-6) or not np.allclose(act[i],ra[j],atol=1e-6) or ph[i]!=rp[j] or not np.array_equal(one[i],ro[j]) or inj[i] or ri[j]:raise RuntimeError(f'alignment failure {eid} clean={i} raw={t}')
    hist=deque(maxlen=20); first=(feat(rs[0],np.zeros(7,'f4'))-mean)/std;hist.extend([first]*20); pred=[]
    for t in range(len(rt)):
     if t: hist.append((feat(rs[t],ra[t-1])-mean)/std)
     with torch.no_grad(): z=m.posterior(torch.as_tensor(np.asarray(hist)[None],device=dev))[0].cpu().numpy()
     pred.append(int(z.argmax())); rows.append({'episode_id':eid,'split':sp,'raw_timestep':int(rt[t]),'gt_stage':int(rp[t]),'pred_stage':int(z.argmax()),'posterior5':z.tolist(),'confidence':float(z.max()),'previous_executed_action_index':None if t==0 else t-1,'history_raw_start':max(0,t-19),'history_raw_end':t})
    pred=np.asarray(pred); allp[sp].extend(pred);ally[sp].extend(rp)
    for t in range(1,len(rp)):
     if (rp[t-1],rp[t]) in ((1,0),(2,0),(3,0)):
      stable=next((k for k in range(t,len(pred)-2) if np.all(pred[k:k+3]==0)),None);trans.append({'transition':f'{rp[t-1]}->0','stable3':stable is not None,'latency':None if stable is None else stable-t})
 report={'TCN_A_NETWORK_REUSE':'YES','TCN_A_CHECKPOINT_REUSE':'YES','TCN_A_NORMALIZATION_REUSE':'YES','TCN_A_ONLINE_WRAPPER_REUSE':'YES','TCN_A_ONLINE_FEATURE_SEMANTICS_REUSE':'YES','TCN_A_TRAIN_FEATURE_SEMANTICS_MATCH':'NO','TCN_ACTION_CHANNEL_SOURCE':'PREVIOUS_EXECUTED_ACTION','TCN_HISTORY_CAUSAL':'YES','TCN_HISTORY_RESET_AT_RECOVERY_START':'NO','TCN_FROZEN':all(not x.requires_grad for x in m.parameters()),'checkpoint':str(CKPT.resolve()),'checkpoint_sha256':sha(CKPT),'history_length':20,'input_dim':19,'raw_clean_file_pair_count':2000,'metrics':{},'regressions':{}}
 for sp in allp:
  y=np.asarray(ally[sp]);p=np.asarray(allp[sp]); fs=f1(y,p);report['metrics'][sp]={'accuracy':float((y==p).mean()),'macro_f1':float(np.mean(fs)),'stage_f1':fs,'confusion_matrix':[[int(np.sum((y==i)&(p==j))) for j in range(5)] for i in range(5)]}
 for q in ('1->0','2->0','3->0'):
  x=[v for v in trans if v['transition']==q]; lat=[v['latency'] for v in x if v['latency'] is not None];report['regressions'][q]={'count':len(x),'stable3_count':sum(v['stable3'] for v in x),'stable3_rate':float(np.mean([v['stable3'] for v in x])) if x else 0.,'mean_detection_latency':None if not lat else float(np.mean(lat)),'median_detection_latency':None if not lat else float(np.median(lat))}
 test=report['metrics']['test']; gate=test['macro_f1']>=.99 and min(test['stage_f1'])>=.98 and all(report['regressions'][q]['stable3_rate']>=.95 for q in report['regressions']);report['TCN_ONLINE_COMPATIBILITY_VALID']='YES' if gate else 'NO';(OUT/'episode_level_online_predictions.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n');(OUT/'compatibility_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
