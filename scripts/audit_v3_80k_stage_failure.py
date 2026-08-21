#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_srdp_condition_v3_20260818/checkpoint_sweep_n100/step_00080000/evaluation_report.json'
OUT=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_v3_srdp_optimizer_20260819'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads(SRC.read_text())['rows']; out=[]
    for r in rows:
        if r['success']: cat='OTHER'
        elif r['illegal_drop']: cat='ILLEGAL_DROP'
        elif r['ik_failure']: cat='IK_FAILURE'
        elif not r['grasp']: cat='GRASP'
        elif not r['transport']: cat='TRANSPORT'
        elif not r['place'] or not r['release']: cat='PLACE_RELEASE'
        elif not r['retreat']: cat='RETREAT'
        else: cat='APPROACH'
        out.append({'seed':r['environment_seed'],'success':r['success'],'approach':r['grasp'],'grasp':r['grasp'],'lift':r['lift'],'transport':r['transport'],'place':r['place'],'release':r['release'],'retreat':r['retreat'],'illegal_drop':r['illegal_drop'],'ik_failure':r['ik_failure'],'timeout':r['timeout'],'terminal_active_stage':r.get('failure_phase') or ('COMPLETE' if r['success'] else 'UNKNOWN'),'failure_category':cat})
    with (OUT/'v3_80k_stage_failure_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    summary={'episodes':len(out),'approach_reached':sum(x['approach'] for x in out),'grasp_reached':sum(x['grasp'] for x in out),'lift_reached':sum(x['lift'] for x in out),'transport_reached':sum(x['transport'] for x in out),'place_reached':sum(x['place'] for x in out),'release_reached':sum(x['release'] for x in out),'retreat_success':sum(x['retreat'] for x in out),'illegal_drop':sum(x['illegal_drop'] for x in out),'ik_failure':sum(x['ik_failure'] for x in out),'timeout':sum(x['timeout'] for x in out),'failure_categories':{k:sum(x['failure_category']==k for x in out) for k in ('APPROACH','GRASP','TRANSPORT','PLACE_RELEASE','RETREAT','ILLEGAL_DROP','IK_FAILURE','OTHER')},'source':str(SRC.resolve())}
    (OUT/'v3_80k_stage_failure_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
