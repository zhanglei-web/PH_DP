#!/usr/bin/env python3
"""CUDA-only fixed-case Normal/Place growth evaluator for Global and Oracle-V2."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
import validate_recovery_stage_checkpoints as oracle_protocol
import validate_recovery_global_checkpoints as global_protocol
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/experiment1_growth_analysis';MANIFEST=ROOT/'outputs/final_stage_ambiguity_experiments_20260820/growth_analysis/growth_normal_place_manifest.json';GLOBAL=ROOT/'outputs/recovery_stage_dp_training/recovery_global_120k_20260820';ORACLE=ROOT/'outputs/recovery_stage_dp_training/recovery_stage_v2_120k_20260820';STEPS=tuple(range(10000,120001,10000));STAGES=['APPROACH','GRASP_LIFT','TRANSPORT','PLACE_RELEASE','RETREAT']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cases():
 p=json.loads(MANIFEST.read_text());x=p['cases'];c={k:[q for q in x if q['kind']==k] for k in ('NORMAL','PLACE_RECOVERY')}
 if {k:len(v) for k,v in c.items()}!={'NORMAL':50,'PLACE_RECOVERY':50}:raise RuntimeError('growth manifest must be 50+50')
 return p,c
def predictor(name,ck,dev):
 d=GLOBAL if name=='Global' else ORACLE;n=d/'normalization_stats.npz'
 return global_protocol.GlobalPredictor(ck,n,dev) if name=='Global' else oracle_protocol.Predictor('V2',ck,n,dev)
def run_case(name,case,model,traces):
 r=global_protocol.evaluate_case(case,model,traces) if name=='Global' else oracle_protocol.evaluate_case(case,model,traces)
 # evaluator returns GT-driven stage milestones; trace preserves exact final stage.
 trace=json.loads((traces/f"{case['case_id']}.json").read_text());rows=trace.get('rows',[]);last=int(rows[-1].get('stage',0)) if rows else 0
 return {'checkpoint':None,'model_name':name,'scenario':case['kind'],'episode_id':case['case_id'],'success':bool(r['success']),'timeout':bool(r['timeout']),'illegal_drop':bool(r['illegal_drop']),'ik_failure':bool(r['ik'] if 'ik' in r else r.get('ik_failure',False)),'other_failure':not(bool(r['success']) or bool(r['timeout']) or bool(r['illegal_drop']) or bool(r.get('ik',r.get('ik_failure',False)))),'steps':int(r['steps']),'approach_reached':True,'grasp_reached':bool(r.get('grasp',False)),'lift_reached':bool(r.get('lift',False)),'transport_reached':bool(r.get('transport',False)),'place_reached':bool(r.get('place_release',r.get('place',False))),'retreat_reached':bool(r.get('retreat',False)),'last_active_stage':last,'recovery_success':bool(r.get('success',False)) if case['kind']=='PLACE_RECOVERY' else None,'regression_seen':bool(r.get('regression')) if case['kind']=='PLACE_RECOVERY' else None,'3_to_0_seen':bool(r.get('regression') and r['regression'].get('from')==3 and r['regression'].get('to')==0) if case['kind']=='PLACE_RECOVERY' else None}
def graph(path,title,series):
 im=Image.new('RGB',(1000,600),'white');d=ImageDraw.Draw(im);d.text((20,15),title,fill='black');L,R,T,B=70,950,60,520;cols=['#1f77b4','#d62728','#2ca02c','#9467bd']
 for n,(name,vals) in enumerate(series.items()):
  pts=[]
  for i,v in enumerate(vals):pts.append((L+i*(R-L)/(len(vals)-1),B-v*(B-T)))
  d.line(pts,fill=cols[n],width=3);d.text((L+220*n,550),name,fill=cols[n])
 d.line((L,B,R,B),fill='black');d.line((L,T,L,B),fill='black');im.save(path)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--check-only',action='store_true');a=p.parse_args();manifest,groups=cases();check={'manifest_sha256':sha(MANIFEST),'case_counts':{k:len(v) for k,v in groups.items()},'steps':STEPS}
 for name,d in {'Global':GLOBAL,'Oracle-V2':ORACLE}.items():
  for s in STEPS:
   ck=d/'checkpoints'/f'step_{s:06d}.pt'
   if not ck.is_file():raise FileNotFoundError(ck)
   q=torch.load(ck,map_location='cpu',weights_only=False)
   if 'model' not in q:raise ValueError(f'{name} invalid {ck}')
 if a.check_only:print(json.dumps({'READY_FOR_CUDA_ROLLOUT':'YES',**check},indent=2));return
 if not torch.cuda.is_available():raise RuntimeError('CUDA_REQUIRED_BUT_UNAVAILABLE')
 dev=torch.device('cuda:0');torch.cuda.set_device(dev);a.output.mkdir(parents=True,exist_ok=True);(a.output/'growth_normal_place_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');allrows=[];stream=(a.output/'checkpoint_results.jsonl').open('w')
 try:
  for name,d in {'Global':GLOBAL,'Oracle-V2':ORACLE}.items():
   for s in STEPS:
    ck=d/'checkpoints'/f'step_{s:06d}.pt';model=predictor(name,ck,dev)
    for scenario,cc in groups.items():
     for case in cc:
      row=run_case(name,case,model,a.output/'traces'/name/f'{s:06d}'/scenario);row['checkpoint']=s;row['checkpoint_path']=str(ck.resolve());row['checkpoint_sha256']=sha(ck);stream.write(json.dumps(row)+'\n');stream.flush();allrows.append(row)
  (a.output/'checkpoint_results.json').write_text(json.dumps(allrows,indent=2)+'\n')
 finally:stream.close()
 growth=[];failure=[];stages=[]
 for name in ('Global','Oracle-V2'):
  for s in STEPS:
   x=[r for r in allrows if r['model_name']==name and r['checkpoint']==s]
   for scenario in groups:
    q=[r for r in x if r['scenario']==scenario];growth.append({'model':name,'checkpoint':s,'scenario':scenario,'success':float(np.mean([r['success'] for r in q])),'timeout':float(np.mean([r['timeout'] for r in q])),'illegal_drop':float(np.mean([r['illegal_drop'] for r in q])),'ik_failure':float(np.mean([r['ik_failure'] for r in q]))})
   bad=[r for r in x if not r['success']];failure.append({'model':name,'checkpoint':s,'N_failed':len(bad),**{k:float(np.mean([r[k] for r in bad])) if bad else 0. for k in ('timeout','illegal_drop','ik_failure','other_failure')}});stages.append({'model':name,'checkpoint':s,**{f'Stage{i}':sum(r['last_active_stage']==i for r in bad) for i in range(5)}})
 for path,rows in ((a.output/'growth_results.json',growth),(a.output/'failure_migration.json',failure),(a.output/'failure_stage_distribution.json',stages)):(path).write_text(json.dumps(rows,indent=2)+'\n')
 graph(a.output/'FIG_A1_NORMAL_SUCCESS_GROWTH.png','FIG A1 Normal success growth',{n:[next(r['success'] for r in growth if r['model']==n and r['checkpoint']==s and r['scenario']=='NORMAL') for s in STEPS] for n in ('Global','Oracle-V2')});graph(a.output/'FIG_A2_PLACE_RECOVERY_SUCCESS_GROWTH.png','FIG A2 Place Recovery success growth',{n:[next(r['success'] for r in growth if r['model']==n and r['checkpoint']==s and r['scenario']=='PLACE_RECOVERY') for s in STEPS] for n in ('Global','Oracle-V2')});graph(a.output/'FIG_A3_FAILURE_CAUSE_MIGRATION.png','FIG A3 Timeout among failures',{n:[next(r['timeout'] for r in failure if r['model']==n and r['checkpoint']==s) for s in STEPS] for n in ('Global','Oracle-V2')});graph(a.output/'FIG_A4_FAILURE_STAGE_MIGRATION.png','FIG A4 Late-stage failures',{n:[next((r['Stage3']+r['Stage4'])/max(r['Stage0']+r['Stage1']+r['Stage2']+r['Stage3']+r['Stage4'],1) for r in stages if r['model']==n and r['checkpoint']==s) for s in STEPS] for n in ('Global','Oracle-V2')});print(json.dumps({'GROWTH_EVALUATION_VALID':'YES','episodes':len(allrows),'output':str(a.output.resolve())},indent=2))
if __name__=='__main__':main()
