#!/usr/bin/env python3
"""N=8 diversity audit and paired frozen-V2 Q reranking evaluation."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np,torch
from mujoco_shared_control.rss2023.oracle_stage_embedding_evaluation import OracleStageEmbeddingPredictor
from mujoco_shared_control.rss2023.oracle_stage_evaluation import evaluate_episode
from mujoco_shared_control.rss2023.global_evaluation import summarize
from train_stage_value_q_recovery import QNet
ROOT=Path(__file__).resolve().parents[1];V2=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818';OUT=ROOT/'outputs/stage_value_guidance/v2_stage_q_recovery_value_v2';CK=V2/'checkpoints/step_00080000.pt';N=8
class CandidatePolicy:
 def __init__(self,rerank):
  self.base=OracleStageEmbeddingPredictor(CK,V2/'normalization_stats.npz',device_name='cpu');self.action_spec=self.base.action_spec;self.rerank=rerank;self.step=0
  with np.load(V2/'normalization_stats.npz') as z:self.om=z['observation_mean'].astype('f4');self.os=z['observation_std'].astype('f4');self.am=z['action_mean'].astype('f4');self.astd=z['action_std'].astype('f4')
  self.q=QNet();self.q.load_state_dict(torch.load(OUT/'q_checkpoint_valid.pt',map_location='cpu',weights_only=False)['model']);self.q.eval()
 def reset_sampling(self,seed):self.seed=seed;self.step=0
 def candidates(self,obs,stage):
  out=[]
  for c in range(N):self.base.reset_sampling(self.seed+self.step*N+c);out.append(self.base.sample(obs,stage))
  return np.asarray(out)
 def sample(self,obs,stage):
  cand=self.candidates(obs,stage);self.step+=1
  if not self.rerank:return cand[0]
  state=np.r_[obs,np.eye(5,dtype='f4')[stage]];s=torch.from_numpy(((state-self.om)/self.os)[None].repeat(N,0));a=torch.from_numpy((cand-self.am)/self.astd)
  with torch.no_grad():q=self.q(s,a).squeeze(-1).numpy()
  return cand[int(np.argmax(q))]
def diversity():
 # Frozen held-out V2 test states, no environment mutation.
 import h5py
 data=json.loads((ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL/dataset_manifest.json').read_text());samples=[]
 for e in data['episodes'][:40]:
  with h5py.File(e['path'],'r') as f:samples.append((f['full_physical_state'][0].astype('f4'),int(f['active_phase'][0])))
 p=CandidatePolicy(True);rows=[]
 for i,(physical,stage) in enumerate(samples):
  p.reset_sampling(9_000_000+i);c=p.candidates(physical,stage)
  for a in range(N):
   for b in range(a+1,N):
    ta,tb=c[a,:3],c[b,:3];den=np.linalg.norm(ta)*np.linalg.norm(tb);rows.append({'sample':i,'pairwise_action_l2':float(np.linalg.norm(c[a]-c[b])),'translation_cosine':float(np.dot(ta,tb)/den) if den else 1.,'rotation_difference':float(np.linalg.norm(c[a,3:6]-c[b,3:6])),'gripper_agreement':bool((c[a,6]>=.375)==(c[b,6]>=.375))})
 with (OUT/'candidate_diversity.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 return {'pairs':len(rows),'mean_action_l2':float(np.mean([r['pairwise_action_l2'] for r in rows])),'mean_translation_cosine':float(np.mean([r['translation_cosine'] for r in rows])),'mean_rotation_difference':float(np.mean([r['rotation_difference'] for r in rows])),'gripper_agreement':float(np.mean([r['gripper_agreement'] for r in rows]))}
def evaluate(label,rerank):
 p=CandidatePolicy(rerank);rows=[]
 for i,seed in enumerate(range(2_000_000,2_000_100)):
  rows.append(evaluate_episode(p,seed,8_000_000+seed));
  if (i+1)%10==0:print(label,i+1,flush=True)
 return rows
def main():
 d=diversity();d['candidate_diversity_sufficient']=d['mean_action_l2']>0.05
 (OUT/'candidate_diversity_summary.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
 if not d['candidate_diversity_sufficient']:return
 allrows={};
 for label,rerank in [('V2',False),('V2_Q',True)]:allrows[label]=evaluate(label,rerank)
 summary=[]
 for label,rows in allrows.items():
  s=summarize(rows);summary.append({'method':label,'success':s['success']['count'],'retreat':s['retreat']['count'],'timeout':s['timeout']['count'],'post_place_timeout':sum(r['timeout'] and r['place'] for r in rows),'illegal_drop':s['illegal_drop']['count'],'ik_failure':s['ik_failure']['count'],'grasp':s['grasp']['count'],'lift':s['lift']['count'],'transport':s['transport']['count'],'place':s['place']['count'],'release':s['release']['count']})
  (OUT/f'{label.lower()}_evaluation_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (OUT/'v2_vs_q_normal.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
 audit=json.loads((OUT/'q_long_horizon_audit.json').read_text());audit.update({'candidate_diversity_sufficient':'YES','value_improves_closed_loop':'YES' if summary[1]['success']>summary[0]['success'] and summary[1]['timeout']<summary[0]['timeout'] else 'NO','closed_loop_summary':summary});(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
