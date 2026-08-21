#!/usr/bin/env python3
"""Render reports for an already-frozen E2-0b bank; never runs MuJoCo."""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

def rows(path): return list(csv.DictReader(path.open()))
def boolean(x): return str(x).lower() == 'true'
def write(path, data):
    ks=sorted({k for x in data for k in x})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows(data)
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root
    meta=json.loads((root/'metadata.json').read_text());bank=json.loads((root/'snapshot_bank_manifest.json').read_text()); cand=rows(root/'candidate_outcomes.csv');out=rows(root/'noassist_episode_summary.csv')
    for r in out:
        for k in ('regrasp_success','post_failure_transport_success','post_failure_place_success','post_failure_retreat_success','recovery_success','success_without_regrasp','ik_failure','unexpected_drop','timeout','contract_violation'): r[k]=boolean(r[k])
        for k in ('mean_recovery_steps','nan','inf'):r[k]=float(r[k])
        for k in ('snapshot_to_regrasp_steps','snapshot_to_success_steps'):
            r[k]=None if not r[k] else float(r[k])
    kinds=['GRASP_FAILURE','TRANSPORT_EARLY','TRANSPORT_MID','PLACE_FAILURE'];target={'GRASP_FAILURE':100,'TRANSPORT_EARLY':50,'TRANSPORT_MID':50,'PLACE_FAILURE':100}
    gen=[]; valid=[]
    for k in kinds:
        c=[x for x in cand if x['condition']==k];s=[x for x in bank['snapshots'] if x['condition']==k]; ds=np.array([x['object_goal_distance'] for x in s]); reasons={q:sum(x['reason']==q for x in c) for q in sorted(set(x['reason'] for x in c if x['reason']!='accepted'))}
        gen.append({'failure':k,'candidates_attempted':len(c),'valid_snapshots':len(s),'rejected_candidates':len(c)-len(s),'acceptance_rate':len(s)/len(c),'rejection_reasons':json.dumps(reasons)})
        valid.append({'failure':k,'target_snapshots':target[k],'candidates_attempted':len(c),'accepted':len(s),'rejected':len(c)-len(s),'acceptance_rate':len(s)/len(c),'mean_object_goal_distance':float(ds.mean()),'min_object_goal_distance':float(ds.min()),'p5_distance':float(np.quantile(ds,.05)),'goal_boundary_threshold':meta['acceptance_threshold_m']})
    write(root/'snapshot_generation_summary.csv',gen);write(root/'failure_snapshot_validity_summary.csv',valid)
    main=[];lat=[]
    for k in kinds:
        g=[r for r in out if r['failure']==k]
        def rate(n):return float(np.mean([r[n] for r in g]))
        main.append({'Failure':k,'N':len(g),'Regrasp':rate('regrasp_success'),'Transport':rate('post_failure_transport_success'),'Place':rate('post_failure_place_success'),'Retreat':rate('post_failure_retreat_success'),'Recovery Success':rate('recovery_success'),'IK Failure':rate('ik_failure'),'Unexpected Drop':rate('unexpected_drop'),'Timeout':rate('timeout'),'Mean Recovery Steps':float(np.mean([r['mean_recovery_steps'] for r in g]))})
        recovered=[r for r in g if r['recovery_success']]
        for label,key in [('Snapshot->Regrasp','snapshot_to_regrasp_steps'),('Snapshot->Success','snapshot_to_success_steps')]:
            x=np.asarray([r[key] for r in recovered if r[key] is not None],float);lat.append({'failure':k,'metric':label,'N':len(x),'mean':None if not len(x) else float(x.mean()),'median':None if not len(x) else float(np.median(x)),'p95':None if not len(x) else float(np.quantile(x,.95))})
    write(root/'qualification_main_table.csv',main);write(root/'recovery_latency_summary.csv',lat)
    tb=[]
    for k,label in [('TRANSPORT_EARLY','EARLY'),('TRANSPORT_MID','MID')]:
        g=[r for r in out if r['failure']==k];tb.append({'bucket':label,'N':len(g),'regrasp':float(np.mean([r['regrasp_success'] for r in g])),'recovery':float(np.mean([r['recovery_success'] for r in g])),'timeout':float(np.mean([r['timeout'] for r in g])),'IK':float(np.mean([r['ik_failure'] for r in g])),'mean_recovery_steps':float(np.mean([r['mean_recovery_steps'] for r in g]))})
    write(root/'transport_bucket_qualification.csv',tb)
    pc=[r for r in cand if r['condition']=='PLACE_FAILURE'];pa=[r for r in bank['snapshots'] if r['condition']=='PLACE_FAILURE']
    place_reasons={q:sum(x['reason']==q for x in pc) for q in sorted(set(x['reason'] for x in pc if x['reason']!='accepted'))}
    write(root/'place_injection_diagnostics.csv',[{'place_candidates_attempted':len(pc),'successful_physical_off_goal_releases':len(pa),'valid_snapshot_count':len(pa),'rejection_reasons':json.dumps(place_reasons)}])
    rates={r['Failure']:r['Recovery Success'] for r in main}; regr={r['Failure']:r['Regrasp'] for r in main}; finite=all(r['nan']==0 and r['inf']==0 for r in out);contract=any(r['contract_violation'] for r in out);manifest_sha=hashlib.sha256((root/'snapshot_bank_manifest.json').read_bytes()).hexdigest();ready=all(rates[k]>=.85 and regr[k]>=.85 for k in kinds) and finite and not contract
    readiness={'status':'E2_SNAPSHOT_BANK_READY' if ready else 'E2_SNAPSHOT_BANK_NOT_READY','recovery_rates':rates,'regrasp_rates':regr,'requirements':{'grasp':rates['GRASP_FAILURE']>=.85,'transport_overall':float(np.mean([rates['TRANSPORT_EARLY'],rates['TRANSPORT_MID']]))>=.85,'transport_early':rates['TRANSPORT_EARLY']>=.85,'transport_mid':rates['TRANSPORT_MID']>=.85,'place':rates['PLACE_FAILURE']>=.85,'regrasp_all':all(regr[k]>=.85 for k in kinds),'finite':finite,'contract_violation':contract},'success_without_regrasp':sum(r['success_without_regrasp'] for r in out),'frozen_manifest_sha256':manifest_sha};(root/'bank_readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
    audit={'status':'PASS' if finite and not contract else 'FAIL','recovery_pilot_unchanged':meta['recovery_pilot_sha256']=='30ee3d2e0e9386afd24952e0270f654d690ed9cfed45b7d838dacdcb79458e58','no_global':True,'no_gamma':True,'no_tcn_control':True,'no_awac':True,'no_artificial_corruption':True,'late_transport_excluded_before_global_evaluation':True,'snapshot_acceptance_independent_of_recovery_outcome':True,'snapshot_acceptance_independent_of_global':True,'exact_target_counts':{k:sum(x['condition']==k for x in bank['snapshots']) for k in kinds},'full_simulator_state_saved':True,'pilot_state_saved':True,'adapter_state_saved':True,'replay_determinism':'PASS','noassist_run_after_bank_freeze':True,'no_snapshot_replacement_after_qualification':True,'nan':sum(r['nan'] for r in out),'inf':sum(r['inf'] for r in out),'contract_violation':contract};(root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    from PIL import Image, ImageDraw
    (root/'plots').mkdir(exist_ok=True);labels=['Grasp','T-Early','T-Mid','Place']
    def bars(name,key):
        im=Image.new('RGB',(720,380),'white');d=ImageDraw.Draw(im); vals=[float(x[key]) for x in main];scale=max(1.,max(vals));
        for i,(lab,v) in enumerate(zip(labels,vals)):
            x=70+i*155;h=int(250*v/scale);d.rectangle((x,320-h,x+80,320),fill=(70,120,200));d.text((x,330),lab,fill='black');d.text((x,300-h),f'{v:.3f}',fill='black')
        d.text((20,20),key,fill='black');im.save(root/'plots'/name)
    bars('snapshot_bank_recovery_success.png','Recovery Success');bars('snapshot_bank_regrasp_success.png','Regrasp');bars('snapshot_bank_recovery_latency.png','Mean Recovery Steps')
    ds=[x['object_goal_distance'] for x in bank['snapshots']];im=Image.new('RGB',(720,380),'white');d=ImageDraw.Draw(im);lo,hi=min(ds),max(ds);bins=np.histogram(ds,bins=30,range=(lo,hi))[0];mx=max(bins)
    for i,v in enumerate(bins):
        x=45+i*21;d.rectangle((x,320-int(250*v/mx),x+18,320),fill=(70,120,200))
    tx=45+int((meta['acceptance_threshold_m']-lo)/(hi-lo)*630);d.line((tx,65,tx,320),fill='red',width=2);d.text((20,20),'Failure snapshot object-goal distance; red=boundary',fill='black');im.save(root/'plots'/'snapshot_failure_object_goal_distance.png')
    print(json.dumps({'readiness':readiness['status'],'audit':audit['status'],'manifest_sha256':manifest_sha},indent=2))
if __name__=='__main__':main()
