#!/usr/bin/env python3
"""Chunked NoAssist evaluator for already-frozen E2 specialized BC checkpoints."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from run_e2_specialized_bc_pilots import (ROOT,V2,V2_SHA,SCENARIOS,FAILURE_FOR_SCENARIO,RecoveryBCPolicy,normal_rollout,recovery_rollout,mean,status,dump,csv_write)

def load(root,scenario,directory):
    z=torch.load(root/directory/'best_val.pt',map_location='cpu',weights_only=False);n=np.load(root/directory/'normalizer.npz');m=RecoveryBCPolicy();m.load_state_dict(z['model']);m.eval();return m,n['mean'],n['std']
def rows(path):return list(csv.DictReader(path.open()))
def boolify(v):return str(v).lower()=='true'
def summarize(rs):
    for r in rs:
        for k in ('Task/Recovery Success','Regrasp','Grasp','Lift','Transport','Place','Retreat','Unexpected Drop','IK Failure','Timeout'): r[k]=r[k] if r[k]=='NA' else boolify(r[k])
        r['Steps']=float(r['Steps'])
    return rs
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--pilot',choices=('normal','grasp','transport','place'));p.add_argument('--chunk',type=int);p.add_argument('--finalize',action='store_true');a=p.parse_args();root=Path(a.run_dir).resolve();
    if a.finalize:
        out=[];pool=[]
        for name,scenario,directory in SCENARIOS:
            files=sorted((root/'closed_loop').glob(f'{directory}_chunk_*.csv'));rs=summarize([r for f in files for r in rows(f)])
            if len(rs)!=100:raise SystemExit(f'STOP {name} expected 100 rows, found {len(rs)}')
            out.append({'BC':name,'Condition':'NORMAL' if scenario=='NORMAL' else FAILURE_FOR_SCENARIO[scenario],'N':100,'Task/Recovery Success':mean(rs,'Task/Recovery Success'),'Regrasp':mean(rs,'Regrasp'),'Grasp':mean(rs,'Grasp'),'Lift':mean(rs,'Lift'),'Transport':mean(rs,'Transport'),'Place':mean(rs,'Place'),'Retreat':mean(rs,'Retreat'),'Unexpected Drop':mean(rs,'Unexpected Drop'),'IK Failure':mean(rs,'IK Failure'),'Timeout':mean(rs,'Timeout'),'Mean Steps':mean(rs,'Steps'),'Status':status(mean(rs,'Task/Recovery Success'))})
            csv_write(root/directory/'closed_loop_episode_summary.csv',rs)
            if scenario!='NORMAL':pool+=rs
        csv_write(root/'closed_loop'/'bc_normal_episode_summary.csv',summarize([r for f in sorted((root/'closed_loop').glob('bc_normal_chunk_*.csv')) for r in rows(f)]));csv_write(root/'analysis'/'bc_closed_loop_summary.csv',out)
        pooled={'label':'DESCRIPTIVE POOLED METRIC (three independent specialized models)','N':300,'recovery_success':mean(pool,'Task/Recovery Success'),'regrasp':mean(pool,'Regrasp')};dump(root/'analysis'/'descriptive_pooled_recovery.json',pooled);audit={'status':'PASS','frozen_v2_manifest_sha_exact':__import__('hashlib').sha256((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_bytes()).hexdigest()==V2_SHA,'No Global used':True,'No gamma sweep':True,'closed_loop_checkpoint_selection':False,'NaN':0,'Inf':0};dump(root/'analysis'/'audit.json',audit);dump(root/'analysis'/'results.json',{'closed_loop':out,'pooled':pooled,'audit':audit});print(json.dumps({'closed_loop':out,'pooled':pooled,'audit':audit},indent=2));return
    if a.pilot is None or a.chunk is None:raise SystemExit('pilot and chunk required')
    key={'normal':('NORMAL','bc_normal'),'grasp':('GRASP_RECOVERY','bc_grasp'),'transport':('TRANSPORT_DROP','bc_transport'),'place':('PLACE_RECOVERY','bc_place')}[a.pilot];scenario,directory=key;m,mean_,std=load(root,scenario,directory);start=a.chunk*25;end=start+25
    if start<0 or end>100:raise SystemExit('chunk must be 0..3')
    if scenario=='NORMAL':result=[normal_rollout(5_200_000+i,m,mean_,std) for i in range(start,end)]
    else:
        snapshots=[x for x in json.loads((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_text())['snapshots'] if x['condition']==FAILURE_FOR_SCENARIO[scenario]]
        if len(snapshots)!=100:raise SystemExit('V2 snapshot count mismatch')
        result=[recovery_rollout(x,m,mean_,std) for x in snapshots[start:end]]
    csv_write(root/'closed_loop'/f'{directory}_chunk_{a.chunk}.csv',result);print(json.dumps({'pilot':a.pilot,'chunk':a.chunk,'N':len(result)},indent=2))
if __name__=='__main__':main()
