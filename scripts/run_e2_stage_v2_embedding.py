#!/usr/bin/env python3
"""E2 evaluation for frozen Oracle Stage-Embedding V2 at 50k."""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
import numpy as np, torch
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
import run_e2_stage_v1_oracle as e1

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/checkpoints/step_00050000.pt'
NORMALIZATION=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/normalization_stats.npz'
OUT=ROOT/'outputs/experiments/e2_stage_v2_embedding_50k'
GAMMA=.7

class OracleV2:
 def __init__(self,checkpoint,normalization,device):
  self.device=device;p=torch.load(checkpoint,map_location=device,weights_only=False);cfg=StageEmbeddingDiffusionConfig(**{k:v for k,v in p['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__});self.model=StageEmbeddingDiffusion(cfg).to(device).eval();self.model.load_state_dict(p['model']);
  with np.load(normalization,allow_pickle=False) as n:self.om=np.asarray(n['observation_mean'],np.float32);self.os=np.asarray(n['observation_std'],np.float32);self.am=np.asarray(n['action_mean'],np.float32);self.astd=np.asarray(n['action_std'],np.float32)
 def assist(self,state,raw,stage,seed):
  if state.shape!=(43,) or stage not in range(5):raise ValueError('invalid V2 E2 input')
  obs=np.r_[state,np.eye(5,dtype=np.float32)[stage]];o=torch.as_tensor((obs-self.om)/self.os,device=self.device).unsqueeze(0);a=torch.as_tensor((raw-self.am)/self.astd,device=self.device).unsqueeze(0);g=torch.Generator(device=self.device).manual_seed(seed);out=self.model.assist(o,a,GAMMA,generator=g)[0];return (out.cpu().numpy()*self.astd+self.am).astype(np.float32)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is forbidden')
 if not CHECKPOINT.exists():raise FileNotFoundError(CHECKPOINT)
 dev=torch.device(a.device);torch.cuda.set_device(dev);a.output.mkdir(parents=True,exist_ok=False)
 manifest=json.loads(e1.V2.read_text());counts={k:sum(m['condition']==v for m in manifest['snapshots']) for k,v in {'Grasp Recovery':'GRASP_FAILURE','Transport Recovery':'TRANSPORT_EARLY','Place Recovery':'PLACE_FAILURE'}.items()};counts['Normal']=len(e1.NORMAL_SEEDS);expected={'Normal':300,'Grasp Recovery':100,'Transport Recovery':100,'Place Recovery':100}
 if counts!=expected:raise RuntimeError(f'case count mismatch: {counts}')
 stage_audit={'STAGE_INPUT_DIM':5,'STAGE_SOURCE':'GT / ORACLE','STAGE_ONE_HOT_VALID':True,'ORACLE_STAGE_MAPPING_VALID':all(0<=int(m.get('pre_failure_stage',0))<5 and 0<=int(m.get('post_failure_stage',0))<5 for m in manifest['snapshots']),'ORACLE_STAGE_TEMPORAL_ALIGNMENT_VALID':all(int(m['regression_step'])>=int(m['failure_step']) for m in manifest['snapshots']),'stage_mapping':{'0':'APPROACH','1':'GRASP_LIFT','2':'TRANSPORT','3':'PLACE_RELEASE','4':'RETREAT'}}
 if not stage_audit['ORACLE_STAGE_MAPPING_VALID'] or not stage_audit['ORACLE_STAGE_TEMPORAL_ALIGNMENT_VALID']:raise RuntimeError('STOP stage audit failed')
 (a.output/'stage_condition_audit.json').write_text(json.dumps(stage_audit,indent=2)+'\n');pilot=e1.HybridCheckpointPredictor(e1.AWAC);oracle=OracleV2(CHECKPOINT,NORMALIZATION,dev);rows=[]
 groups={'Normal':[('NORMAL',str(s),s,None) for s in e1.NORMAL_SEEDS],'Grasp Recovery':[('GRASP_FAILURE',m['snapshot_id'],m['environment_seed'],m) for m in manifest['snapshots'] if m['condition']=='GRASP_FAILURE'],'Transport Recovery':[('TRANSPORT_EARLY',m['snapshot_id'],m['environment_seed'],m) for m in manifest['snapshots'] if m['condition']=='TRANSPORT_EARLY'],'Place Recovery':[('PLACE_FAILURE',m['snapshot_id'],m['environment_seed'],m) for m in manifest['snapshots'] if m['condition']=='PLACE_FAILURE']}
 for name,cases in groups.items():
  result=[]
  for kind,ident,seed,meta in cases:result.append(e1.episode(kind,ident,seed,meta,'stage_v2',pilot,oracle,a.output/'traces'/f'{name.replace(" ","_")}_v2_{ident}.json'))
  rows.append({'Scenario':name,'N':len(result),'V2_Stage_Embedding':float(np.mean([x['success'] for x in result])),'Timeout':float(np.mean([x['timeout'] for x in result])),'Grasp':float(np.mean([x['grasp'] for x in result])),'Lift':float(np.mean([x['lift'] for x in result])),'Transport':float(np.mean([x['transport'] for x in result])),'Place_Release':float(np.mean([x['place'] for x in result])),'Retreat':float(np.mean([x['retreat'] for x in result])),'Illegal_Drop':float(np.mean([x['illegal_drop'] for x in result])),'IK':float(np.mean([x['ik_failure'] for x in result]))})
 rec=rows[1:];rows.append({'Scenario':'Recovery Mean','N':300,'V2_Stage_Embedding':float(np.mean([x['V2_Stage_Embedding'] for x in rec]))})
 (a.output/'v2_summary.json').write_text(json.dumps(rows,indent=2)+'\n');(a.output/'audit.json').write_text(json.dumps({'E2_V2_VALID':'YES','checkpoint':str(CHECKPOINT.resolve()),'checkpoint_step':50000,'checkpoint_sha256':e1.sha(CHECKPOINT),'v2_manifest_sha256':e1.sha(e1.V2),'same_cases':True,'case_counts':counts,'gamma':GAMMA,'stage_audit':stage_audit,'TCN_USED':False,'soft_posterior':False,'transition_matrix':False,'value':False,'dpql':False,'noassist_rerun':False,'v1_rerun':False},indent=2)+'\n');print(json.dumps({'output':str(a.output.resolve()),'rows':rows},indent=2))
if __name__=='__main__':main()
