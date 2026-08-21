#!/usr/bin/env python3
"""Resumable, immutable V2 dataset builder for the unified E2 Recovery BC."""
from __future__ import annotations
import argparse,collections,hashlib,json,multiprocessing as mp,shutil
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import h5py,numpy as np
import collect_stage_dataset_v1 as c

ROOT=Path(__file__).resolve().parents[1]; OLD=ROOT/'outputs/stage_dataset/stage_dataset_v1_hysteresis_20260817T052000Z';OUT=ROOT/'outputs/stage_dataset';V2=ROOT/'outputs/experiments/e2_failure_snapshot_bank_v2/run_20260818T030000Z';PSHA='30ee3d2e0e9386afd24952e0270f654d690ed9cfed45b7d838dacdcb79458e58';V2SHA='d06c1b95b821ab797ef83506c4bfec952d861313e4cab68d19017878a545496f';TARGET={'NORMAL':1000,'GRASP_RECOVERY':1000,'TRANSPORT_DROP':1000,'PLACE_RECOVERY':1000};ADD={'GRASP_RECOVERY':700,'TRANSPORT_DROP':600,'PLACE_RECOVERY':700};BASE={'GRASP_RECOVERY':5300000,'TRANSPORT_DROP':5400000,'PLACE_RECOVERY':5500000};BINS={0:'EARLY',1:'EARLY_MID',2:'MID'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p,default):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def candidates(kind):
    # Fixed candidate stream, with transport bucket determined before collection.
    out=[]
    for i in range(30000):
        seed=BASE[kind]+i;bucket=int(np.random.default_rng(seed+101).integers(0,3)) if kind=='TRANSPORT_DROP' else None
        out.append({'scenario':kind,'candidate_index':i,'seed':seed,'transport_bin':None if bucket is None else BINS[bucket],'transport_bucket':bucket})
    return out
def valid(row):return bool(row['valid'] and row['final_success'] and (row['trajectory_type']=='NORMAL' or row['regression_found']))
def collect(root,kind,new_limit,workers):
    accepted_path=root/f'{kind}_accepted_manifest.json';rejected_path=root/f'{kind}_rejected_manifest.json';accepted=load(accepted_path,[]);rejected=load(rejected_path,[]);seen={int(x['seed']) for x in accepted+rejected};have=list(accepted);goal=min(ADD[kind],len(have)+new_limit);need=goal-len(have)
    if need<=0:print(json.dumps({'scenario':kind,'accepted_existing':len(have),'new_accepted':0}));return
    plan=[x for x in candidates(kind) if x['seed'] not in seen]
    if kind=='TRANSPORT_DROP':
        quota={'EARLY':240,'EARLY_MID':180,'MID':180}; current=collections.Counter(x['transport_bin'] for x in have);plan=[x for x in plan if current[x['transport_bin']]<quota[x['transport_bin']]]
    attempts=[];idx=0;ctx=mp.get_context('spawn')
    with ctx.Pool(workers,initializer=c._init_worker) as pool:
        while need>0:
            batch=plan[idx:idx+workers];idx+=len(batch)
            if not batch:raise RuntimeError(f'candidate stream exhausted for {kind}')
            tasks=[]
            for x in batch:
                path=root/'new_episodes'/kind/f"stage_{kind.lower()}_{x['seed']}.h5";tasks.append((kind,x['seed'],str(path)))
            for x,row in zip(batch,pool.map(c._worker,tasks)):
                event={'scenario':kind,'seed':x['seed'],'candidate_index':x['candidate_index'],'transport_bin':x['transport_bin'],**row}
                if valid(row) and (kind!='TRANSPORT_DROP' or collections.Counter(y['transport_bin'] for y in have)[x['transport_bin']]<{'EARLY':240,'EARLY_MID':180,'MID':180}[x['transport_bin']]):
                    accepted.append(event);have.append(event);need-=1;event['accepted']=True
                else:
                    event['accepted']=False;event['reason']='invalid_regression_or_final_failure' if not valid(row) else 'transport_bin_quota_full';rejected.append(event)
                    src=Path(row['path']);dst=root/'rejected_episodes'/kind/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
                    if src.exists():src.replace(dst);event['path']=str(dst.resolve())
                attempts.append(event)
            dump(accepted_path,accepted);dump(rejected_path,rejected)
            if len(attempts)%50==0:print(json.dumps({'scenario':kind,'accepted_total':len(have),'target_this_call':goal,'attempts':len(attempts)}),flush=True)
    print(json.dumps({'scenario':kind,'accepted_total':len(have),'target_total':ADD[kind]}))
def repair(root):
    """Recover scenario manifests from H5s written before an interrupted collector."""
    for kind in ADD:
        records=[]
        for path in sorted((root/'new_episodes'/kind).glob('*.h5')):
            with h5py.File(path,'r') as f:
                row={k:(v.item() if hasattr(v,'item') else v) for k,v in f.attrs.items() if k in {'episode_id','trajectory_type','seed','failure_step','final_success','episode_length','valid','reset_count','control_dt','drop_timing_bucket','drop_progress_threshold','transport_start_object_goal_distance','object_initial_x','object_initial_y','goal_x','goal_y'}};phase=f['active_phase'][:].astype(int)
            row['path']=str(path.resolve());row['regression_found']=True if kind=='NORMAL' else bool(np.any((phase[:-1]=={'GRASP_RECOVERY':1,'TRANSPORT_DROP':2,'PLACE_RECOVERY':3}[kind])&(phase[1:]==0)));row['scenario']=kind;row['candidate_index']=int(row['seed'])-BASE[kind];row['transport_bucket']=None if kind!='TRANSPORT_DROP' else int(row['drop_timing_bucket']);row['transport_bin']=None if kind!='TRANSPORT_DROP' else BINS[int(row['drop_timing_bucket'])]
            if valid(row):records.append(row)
        all_valid=list(records)
        if kind=='TRANSPORT_DROP':
            quota={'EARLY':240,'EARLY_MID':180,'MID':180};chosen=[]
            for label in ('EARLY','EARLY_MID','MID'):chosen += [x for x in records if x['transport_bin']==label][:quota[label]]
            records=sorted(chosen,key=lambda x:x['seed'])
        else:records=records[:ADD[kind]]
        rejected_records=[{**x,'accepted':False,'reason':'over_target_or_transport_quota'} for x in all_valid if x['path'] not in {y['path'] for y in records}]
        for path in sorted((root/'rejected_episodes'/kind).glob('*.h5')):
            with h5py.File(path,'r') as f: rejected_records.append({'scenario':kind,'seed':int(f.attrs['seed']),'path':str(path.resolve()),'accepted':False,'reason':'invalid_regression_or_final_failure'})
        dump(root/f'{kind}_accepted_manifest.json',records);dump(root/f'{kind}_rejected_manifest.json',rejected_records)
        print(json.dumps({'repaired':kind,'accepted':len(records)}))
def finalize(root):
    old=json.loads((OLD/'episode_manifest.json').read_text());new=[x for k in ADD for x in load(root/f'{k}_accepted_manifest.json',[])];rej=[x for k in ADD for x in load(root/f'{k}_rejected_manifest.json',[])]
    dump(root/'rejected_manifest.json',rej)
    if {k:sum(x['trajectory_type']==k for x in new) for k in ADD}!={k:ADD[k] for k in ADD}:raise RuntimeError('new targets incomplete')
    allrows=old+new;counts={k:sum(x['trajectory_type']==k for x in allrows) for k in TARGET}
    if counts!=TARGET or any(not valid(x) for x in allrows):raise RuntimeError('formal V2 count or validity failure')
    auditrows=[];transport=[]
    for x in allrows:
        with h5py.File(x['path'],'r') as f:
            state=f['full_physical_state'][:];action=f['raw_pilot_action'][:];phase=f['active_phase'][:].astype(int);event=f['event'][:].astype(int)
        scenario=x['trajectory_type']; expected={'GRASP_RECOVERY':(1,0),'TRANSPORT_DROP':(2,0),'PLACE_RECOVERY':(3,0)}.get(scenario);reg=0 if expected is None else int(np.sum((phase[:-1]==expected[0])&(phase[1:]==expected[1])));failure=int(np.sum(event=={'GRASP_RECOVERY':1,'TRANSPORT_DROP':2,'PLACE_RECOVERY':3}.get(scenario,-1)));regrasp=False;success_without=False
        if scenario!='NORMAL':
            drops=np.flatnonzero(event==({'GRASP_RECOVERY':1,'TRANSPORT_DROP':2,'PLACE_RECOVERY':3}[scenario])); start=int(drops[0]) if len(drops) else -1;g=state[:,42].astype(bool);regrasp=bool(start>=0 and np.any((~g[:-1])&g[1:] & (np.arange(len(g)-1)>=start)));success_without=bool(start>=0 and not regrasp and bool(x['final_success']))
        if not np.isfinite(state).all() or not np.isfinite(action).all() or failure!=(0 if scenario=='NORMAL' else 1) or reg!=(0 if scenario=='NORMAL' else 1):raise RuntimeError(f'audit failed {x["episode_id"]}')
        auditrows.append({'Scenario':scenario,'episode_id':x['episode_id'],'length':len(state),'failure_events':failure,'regressions':reg,'regrasp':regrasp,'success_without_regrasp':success_without})
        if scenario=='TRANSPORT_DROP':transport.append({'episode_id':x['episode_id'],'failure_timestamp':int(np.flatnonzero(event==2)[0]),'post_drop_object_goal_distance':float(np.linalg.norm(state[min(int(np.flatnonzero(event==2)[0])+1,len(state)-1),22:25]-state[min(int(np.flatnonzero(event==2)[0])+1,len(state)-1),29:32])),'first_regrasp_timestamp':next((int(i) for i in np.flatnonzero((~state[:-1,42].astype(bool))&state[1:,42].astype(bool)) if i>=int(np.flatnonzero(event==2)[0])),None),'success_without_regrasp':success_without,'transport_bin':x.get('transport_bin')})
    frozen=json.loads((V2/'e2_failure_snapshot_bank_v2_manifest.json').read_text())['snapshots'];train_seeds={int(x['seed']) for x in allrows};frozen_seeds={int(x['environment_seed']) for x in frozen};leak={'status':'PASS' if not train_seeds&frozen_seeds else 'FAIL','overlap_seeds':sorted(train_seeds&frozen_seeds),'training_seed_range':[min(train_seeds),max(train_seeds)],'frozen_v2_manifest_sha256':sha(V2/'e2_failure_snapshot_bank_v2_manifest.json'),'frozen_seed_range':[min(frozen_seeds),max(frozen_seeds)]};dump(root/'train_vs_frozen_v2_leakage_audit.json',leak)
    summary=[]
    for k in TARGET:
        r=[x for x in auditrows if x['Scenario']==k];summary.append({'Scenario':k,'Requested':TARGET[k],'Accepted':len(r),'Rejected':sum(x['scenario']==k for x in rej),'Final Success':len(r),'Final Failure':0,'Transitions':sum(x['length'] for x in r),'Mean Episode Length':float(np.mean([x['length'] for x in r])),'Median Episode Length':float(np.median([x['length'] for x in r])),'Failure Event Count':sum(x['failure_events'] for x in r),'Stage Regression Count':sum(x['regressions'] for x in r),'Regrasp Rate':'NA' if k=='NORMAL' else float(np.mean([x['regrasp'] for x in r])),'Success Without Regrasp':'NA' if k=='NORMAL' else float(np.mean([x['success_without_regrasp'] for x in r]))})
    import csv
    def write(p,rows):
        keys=sorted({k for x in rows for k in x});
        with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    write(root/'dataset_v2_summary.csv',summary);write(root/'transport_recovery_statistics.csv',transport);manifest={'format':'unified-recovery-bc-dataset-v2','source_dataset':str(OLD.resolve()),'new_episode_streams':BASE,'counts':counts,'episodes':allrows,'teacher_sha256':sha(ROOT/'src/mujoco_shared_control/experts/recovery_pilot.py'),'v2_frozen_benchmark_sha256':V2SHA};dump(root/'dataset_manifest.json',manifest);audit={'status':'PASS' if leak['status']=='PASS' else 'FAIL','counts':counts,'formal_episodes':len(allrows),'all_final_success':True,'invalid_place_attempts_from_old_excluded':9,'nan':0,'inf':0,'regressions_valid':True,'teacher_sha_exact':manifest['teacher_sha256']==PSHA,'transport_success_without_regrasp_rate':float(np.mean([x['success_without_regrasp'] for x in transport]))};dump(root/'dataset_v2_audit.json',audit);print(json.dumps({'output':str(root),'audit':audit,'summary':summary},indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--scenario',choices=tuple(ADD));ap.add_argument('--accepted-limit',type=int,default=100);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--finalize',action='store_true');ap.add_argument('--repair-manifests',action='store_true');a=ap.parse_args();root=OUT/f'unified_bc_dataset_v2_4000_{a.run_id}';root.mkdir(parents=True,exist_ok=True);(root/'new_episodes').mkdir(exist_ok=True);(root/'rejected_episodes').mkdir(exist_ok=True)
    if sha(ROOT / "src/mujoco_shared_control/experts/recovery_pilot.py") != PSHA or sha(V2 / "e2_failure_snapshot_bank_v2_manifest.json") != V2SHA: raise SystemExit("STOP frozen hash mismatch")
    if a.repair_manifests:repair(root)
    elif a.finalize:finalize(root)
    elif a.scenario:collect(root,a.scenario,a.accepted_limit,a.workers)
    else:raise SystemExit('scenario required unless --finalize')
if __name__=='__main__':main()
