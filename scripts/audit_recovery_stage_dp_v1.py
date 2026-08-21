#!/usr/bin/env python3
"""Audit a clean Recovery-Stage-DP-V1 collection without changing episodes."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import h5py,numpy as np

TYPES=('NORMAL_SUCCESS','GRASP_RECOVERY_SUCCESS','TRANSPORT_RECOVERY_SUCCESS','PLACE_RECOVERY_SUCCESS')
EDGES={'GRASP_RECOVERY_SUCCESS':(1,0),'TRANSPORT_RECOVERY_SUCCESS':(2,0),'PLACE_RECOVERY_SUCCESS':(3,0)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();root=a.dataset; files=sorted((root/'episodes').rglob('*.h5')); counts=Counter(); transitions=Counter(); edges=Counter(); edge_episodes=Counter(); nan=inf=bad_dim=bad_onehot=bad_stage=bad_source=bad_injection=bad_time=bad_align=bad_success=0; ids=[]; seeds=[]; samples={k:[] for k in EDGES}
 for path in files:
  with h5py.File(path,'r') as f:
   typ=str(f.attrs['episode_type']); counts[typ]+=1; ids.append(str(f.attrs['episode_id']));seeds.append(int(f.attrs['seed']))
   s=f['full_physical_state'][:];ns=f['next_full_physical_state'][:];action=f['executed_action'][:];phase=f['active_phase'][:].astype(int);one=f['stage_onehot'][:];source=f['action_source'][:];inj=f['injection_active'][:];time=f['timestep_dp'][:];transitions[typ]+=len(phase)
   finite=np.isfinite(s).all() and np.isfinite(ns).all() and np.isfinite(action).all();nan+=int(not finite);inf+=int(not finite)
   bad_dim+=int(s.shape[1:]!=(43,) or ns.shape!=s.shape or action.shape!=(len(s),7) or one.shape!=(len(s),5));bad_onehot+=int(not np.allclose(one.sum(1),1) or not np.array_equal(one,np.eye(5,dtype=np.float32)[phase]));bad_stage+=int(np.any(~np.isin(phase,range(5))))
   bad_source+=int(np.any(source!=0));bad_injection+=int(np.any(inj));bad_time+=int(not np.array_equal(time,np.arange(len(time),dtype=time.dtype)));bad_align+=int(len(s)>1 and not np.allclose(ns[:-1],s[1:]));bad_success+=int(not bool(f.attrs['final_success']))
   if typ in EDGES:
    raw_path=root/'raw_rollouts'/typ/path.name
    with h5py.File(raw_path,'r') as raw_file:
     raw_phase=raw_file['active_phase'][:].astype(int)
    edge=EDGES[typ]; n=int(np.sum((raw_phase[:-1]==edge[0])&(raw_phase[1:]==edge[1]))); edge_key='%d_to_%d'%edge; edges[edge_key]+=n; edge_episodes[edge_key]+=int(n>0)
    if len(samples[typ])<20: samples[typ].append(' -> '.join(map(str,[int(raw_phase[0])]+[int(x) for i,x in enumerate(raw_phase[1:],1) if x!=raw_phase[i-1]])))
 split=json.loads((root/'split_manifest.json').read_text()); flat=sum((split['splits'][k] for k in ('train','validation','test')),[]); leak=len(flat)!=len(set(flat)) or set(flat)!=set(ids)
 split_counts={k:Counter(x.split('_SUCCESS_')[0]+'_SUCCESS' if False else next(t for t in TYPES if t.lower() in x.lower()) for x in split['splits'][k]) for k in split['splits']}
 edge_coverage_ok=edge_episodes['1_to_0']==300 and edge_episodes['2_to_0']==400 and edge_episodes['3_to_0']==300
 report={'RECOVERY_STAGE_DP_DATASET_VALID':'YES' if len(files)==2000 and all(counts[k]==n for k,n in zip(TYPES,(1000,300,400,300))) and edge_coverage_ok and not any((nan,inf,bad_dim,bad_onehot,bad_stage,bad_source,bad_injection,bad_time,bad_align,bad_success,leak)) else 'NO','NORMAL_SUCCESS_EPISODES':counts['NORMAL_SUCCESS'],'GRASP_RECOVERY_SUCCESS_EPISODES':counts['GRASP_RECOVERY_SUCCESS'],'TRANSPORT_RECOVERY_SUCCESS_EPISODES':counts['TRANSPORT_RECOVERY_SUCCESS'],'PLACE_RECOVERY_SUCCESS_EPISODES':counts['PLACE_RECOVERY_SUCCESS'],'TOTAL_EPISODES':len(files),'TOTAL_DP_TRANSITIONS':int(sum(transitions.values())),'TRANSITIONS_BY_TYPE':dict(transitions),'PHYSICAL_DIM':43,'STAGE_DIM':5,'ACTION_DIM':7,'TRAINING_INJECTION_TRANSITIONS':bad_injection,'TRAINING_NON_EXPERT_ACTIONS':bad_source,'NAN_COUNT':nan,'INF_COUNT':inf,'STATE_ACTION_ALIGNMENT_FAILURES':bad_align,'TIMESTEP_CONTINUITY_FAILURES':bad_time,'INVALID_STAGE_OR_ONEHOT_EPISODES':bad_stage+bad_onehot,'FINAL_SUCCESS_FAILURES':bad_success,'COUNT_STAGE_1_TO_0':edges['1_to_0'],'COUNT_STAGE_2_TO_0':edges['2_to_0'],'COUNT_STAGE_3_TO_0':edges['3_to_0'],'EPISODES_WITH_STAGE_1_TO_0':edge_episodes['1_to_0'],'EPISODES_WITH_STAGE_2_TO_0':edge_episodes['2_to_0'],'EPISODES_WITH_STAGE_3_TO_0':edge_episodes['3_to_0'],'RECOVERY_STAGE_SEQUENCE_VALID':'YES' if edge_coverage_ok else 'NO','RECOVERY_STAGE_SAMPLES':samples,'TRAIN_VAL_TEST_LEAKAGE':'YES' if leak else 'NO','SPLIT_EPISODES':{k:len(v) for k,v in split['splits'].items()},'E2_SEED_OVERLAP':sum(6_500_000<=s<6_500_300 for s in seeds),'E2_SNAPSHOT_OVERLAP':'UNKNOWN','DATASET_PATH':str(root.resolve())}
 (root/'audit_report.json').write_text(json.dumps(report,indent=2)+'\n');(root/'dataset_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
