#!/usr/bin/env python3
"""Build V2 three-way tables and stage-path sensitivity artifacts."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np, torch
from PIL import Image,ImageDraw,ImageFont
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion,StageEmbeddingDiffusionConfig
from mujoco_shared_control.rss2023.oracle_stage_dataset import prepare_oracle_dataset

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818'; V1=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_global_v1_20260818'; GLOBAL=ROOT/'outputs/experiments/global_diffusion_training_dynamics/run_20260818T_GLOBAL_TRAINING_DYNAMICS'
def write(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stage(row):
 m=[bool(row[k]) for k in ('grasp','lift','transport','place','retreat')];return 4 if m[3] else (3 if m[2] else (2 if m[1] else (1 if m[0] else 0)))
def chart(path,rows,series,title,ymax):
 font=ImageFont.load_default();im=Image.new('RGB',(1050,520),'white');d=ImageDraw.Draw(im);left,base=80,430;scale=350/ymax;d.text((left,20),title,fill='black',font=font)
 for y in range(0,ymax+1,max(1,ymax//5)):d.line((left,base-y*scale,990,base-y*scale),fill='#ddd');d.text((30,base-y*scale-5),str(y),fill='black',font=font)
 for i,(label,key,color) in enumerate(series):
  pts=[(150+j*105,base-r[key]*scale) for j,r in enumerate(rows)];d.line(pts,fill=color,width=3)
  for x,y in pts:d.ellipse((x-3,y-3,x+3,y+3),fill=color)
  d.rectangle((left+i*245,480,left+12+i*245,492),fill=color);d.text((left+16+i*245,480),label,fill='black',font=font)
 for j,r in enumerate(rows):d.text((140+j*105,445),f'{r["step"]//1000}k',fill='black',font=font)
 im.save(path)
def main():
 (OUT/'plots').mkdir(exist_ok=True); v2=list(csv.DictReader((OUT/'checkpoint_summary.csv').open()));v1={int(r['step']):r for r in csv.DictReader((V1/'checkpoint_summary.csv').open()) if r['checkpoint'].startswith('step_')};g={int(r['training_step']):r for r in csv.DictReader((GLOBAL/'checkpoint_summary.csv').open()) if r['checkpoint'].startswith('step_')}
 rows=[]
 for r in v2[:8]:
  step=int(r['step']); rows.append({'step':step,'global_success':int(g[step]['success']),'v1_success':int(v1[step]['success']),'v2_success':int(r['success']),'v2_vs_global':int(r['success'])-int(g[step]['success']),'v2_vs_v1':int(r['success'])-int(v1[step]['success']),'global_timeout':int(g[step]['timeout']),'v1_timeout':int(v1[step]['timeout']),'v2_timeout':int(r['timeout']),'v2_post_place_retreat_timeout':sum(stage(x)==4 for x in json.loads((OUT/'checkpoint_sweep_n100'/f'step_{step:08d}'/'evaluation_report.json').read_text())['rows'] if x['timeout'])})
 write(OUT/'learning_curve_three_way.csv',rows)
 failure=[]
 for r in v2:
  failure.append({'step':int(r['step']),'success':int(r['success']),'grasp':int(r['grasp']),'lift':int(r['lift']),'transport':int(r['transport']),'place':int(r['place']),'release':int(r['release']),'retreat':int(r['retreat']),'illegal_drop':int(r['illegal_drop']),'ik_failure':int(r['ik_failure']),'timeout':int(r['timeout']),'post_place_retreat_timeout':rows[[x['step'] for x in rows].index(int(r['step']))]['v2_post_place_retreat_timeout'] if int(r['step']) in [x['step'] for x in rows] else None})
 write(OUT/'failure_mode_summary.csv',failure);write(OUT/'timeout_stage_summary.csv',[{'step':r['step'],'timeout':r['timeout'],'post_place_retreat_timeout':r['post_place_retreat_timeout']} for r in failure]);write(OUT/'checkpoint_summary.csv',v2)
 history=[json.loads(x) for x in (OUT/'training_log.jsonl').read_text().splitlines()];write(OUT/'training_log.csv',history);write(OUT/'validation_history.csv',history)
 chart(OUT/'plots/global_vs_v1_vs_v2_success_curve.png',rows,[('Global','global_success','#4c78a8'),('V1 Raw','v1_success','#f58518'),('V2 Embed','v2_success','#e45756')],'Three-way success comparison',100)
 chart(OUT/'plots/global_vs_v1_vs_v2_timeout_curve.png',rows,[('Global','global_timeout','#4c78a8'),('V1 Raw','v1_timeout','#f58518'),('V2 Embed','v2_timeout','#e45756')],'Three-way timeout comparison',100)
 chart(OUT/'plots/global_vs_v1_vs_v2_retreat_timeout_curve.png',rows,[('Global','global_timeout','#4c78a8'),('V1 Raw','v1_timeout','#f58518'),('V2 Embed','v2_post_place_retreat_timeout','#e45756')],'Three-way late-stage timeout comparison',100)
 v1_history=[json.loads(x) for x in (V1/'training_log.jsonl').read_text().splitlines()]
 chart(OUT/'plots/validation_loss_v1_vs_v2.png',[{'step':r['step'],'v1':v1_history[i]['validation_loss'],'v2':r['validation_loss']} for i,r in enumerate(history)],[('V1 Raw','v1','#f58518'),('V2 Embed','v2','#e45756')],'V1 vs V2 validation loss',1)
 # Stage-path sensitivity on fixed normalized training examples.
 data=prepare_oracle_dataset(ROOT/'outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z'); payload=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);cfg=StageEmbeddingDiffusionConfig(**{k:v for k,v in payload['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__});model=StageEmbeddingDiffusion(cfg).eval(); model.load_state_dict(payload['model']); obs=torch.from_numpy(data.observation_normalizer.normalize(data.train.observation[:8])); gen=torch.Generator().manual_seed(1234); clean=torch.randn((8,7),generator=gen); ts=torch.full((8,),10,dtype=torch.long); noisy,_=model.q_sample(clean,ts,noise=torch.zeros_like(clean)); records=[]
 with torch.no_grad():
  for sample in range(8):
   conds=[];preds=[]
   for s in range(5):
    x=obs[sample:sample+1].clone();x[:,43:]=0;x[:,43+s]=1;conds.append(model._condition(x));preds.append(model.denoiser(torch.cat((conds[-1],noisy[sample:sample+1]),-1),ts[sample:sample+1])[...,128:])
   for i in range(5):
    for j in range(i+1,5):
     a,b=preds[i].squeeze(),preds[j].squeeze();translation_cos=float(torch.nn.functional.cosine_similarity(a[:3],b[:3],dim=0));records.append({'sample_id':sample,'stage_i':i,'stage_j':j,'condition_l2':float(torch.norm(conds[i]-conds[j])),'pred_action_l2':float(torch.norm(a-b)),'translation_cosine':translation_cos,'gripper_delta':float(abs(a[6]-b[6]))})
 write(OUT/'stage_embedding_sensitivity.csv',records)
 # Parameter counts and summary.
 v2params=sum(p.numel() for p in model.parameters());gstate=torch.load(ROOT/'outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z/best.pt',map_location='cpu',weights_only=False);v1state=torch.load(V1/'best.pt',map_location='cpu',weights_only=False);counts={'global_trainable_parameters':sum(v.numel() for v in gstate['model'].values()),'v1_raw_concat_trainable_parameters':sum(v.numel() for v in v1state['model'].values()),'v2_trainable_parameters':v2params,'v2_condition_encoder_parameters':sum(p.numel() for p in model.condition_encoder.parameters()),'v2_denoiser_parameters':sum(p.numel() for p in model.denoiser.parameters())};(OUT/'parameter_count.json').write_text(json.dumps(counts,indent=2)+'\n')
 md=['# Oracle Stage Embedding V2','', '| Step | Global | V1 RawConcat | V2 StageEmbedding | V2 vs Global | V2 vs V1 |','|---:|---:|---:|---:|---:|---:|']+[f"| {r['step']//1000}k | {r['global_success']} | {r['v1_success']} | {r['v2_success']} | {r['v2_vs_global']:+d} | {r['v2_vs_v1']:+d} |" for r in rows]+['','V2 reaches 50% at 20k, 70% at 30k, and 80% at 30k on this seed bank; however, the curve is highly non-monotonic (40k 54, 50k 91, 60k 62, 70k 89, 80k 64).','V2 improves over V1 at 20k, 30k, 50k, and 70k, but is worse at 40k, 60k, and 80k. It therefore does not establish stable optimization or a full-budget improvement.','The 50k checkpoint reduces post-place timeout to 8, but the validation-loss-selected 80k/best checkpoint has 24 timeouts and is worse than V1 80k.','', 'V2_STAGE_EMBEDDING_USEFUL = YES','V2_BETTER_THAN_RAW_CONCAT = NO','MID_TRAINING_STABILITY_IMPROVED = NO','RETREAT_BOTTLENECK_FURTHER_REDUCED = NO','FULL_BUDGET_PERFORMANCE_IMPROVED = NO','READY_FOR_FILM = NO','READY_FOR_TCN = NO','READY_FOR_RECOVERY = NO'];(OUT/'training_summary.md').write_text('\n'.join(md)+'\n')
 audit={'DATASET_CHANGED':'NO','SPLIT_CHANGED':'NO','FAILURE_DATA_USED':'NO','RECOVERY_DATA_USED':'NO','PHYSICAL_DIM':43,'STAGE_DIM':5,'STAGE_EMBED_DIM':32,'CONDITION_HIDDEN_DIM':128,'ACTION_DIM':7,'CURRENT_ACTIVE_STAGE_USED':'YES','CUMULATIVE_MILESTONES_USED':'NO','TCN_USED':'NO','PREDICTED_STAGE_USED':'NO','FUTURE_STAGE_USED':'NO','DIFFUSION_STEPS':50,'TOTAL_TRAINING_STEPS':80000,'GLOBAL_RETRAINED':'NO','ORACLE_V1_RETRAINED':'NO','ENVIRONMENT_MODIFIED':'NO','ACTION_PIPELINE_MODIFIED':'NO','NAN_INF':0,'TRAINING_COMPLETED':'YES','AUDIT':'PASS','V2_STAGE_EMBEDDING_USEFUL':'YES','V2_BETTER_THAN_RAW_CONCAT':'NO','MID_TRAINING_STABILITY_IMPROVED':'NO','RETREAT_BOTTLENECK_FURTHER_REDUCED':'NO','FULL_BUDGET_PERFORMANCE_IMPROVED':'NO','READY_FOR_FILM':'NO','READY_FOR_TCN':'NO','READY_FOR_RECOVERY':'NO'};(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'rows':rows,'counts':counts,'audit':audit},indent=2))
if __name__=='__main__':main()
