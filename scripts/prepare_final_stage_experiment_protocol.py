#!/usr/bin/env python3
"""Freeze the no-retraining Normal/Place protocol and summarize completed offline analysis."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/final_stage_ambiguity_experiments_20260820'
SOURCE=ROOT/'outputs/recovery_stage_dp_validation_80k_120k/validation_case_manifest.json'
AMB=OUT/'ambiguity_analysis/ambiguity_report.json'
MODELS={'GLOBAL':ROOT/'outputs/recovery_stage_dp_training/recovery_global_120k_20260820','ORACLE_V2':ROOT/'outputs/recovery_stage_dp_training/recovery_stage_v2_120k_20260820','TCN_V2':ROOT/'outputs/recovery_stage_dp_training/recovery_tcn_v2_120k_20260820'}
STEPS=list(range(10000,120001,10000))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True); source=json.loads(SOURCE.read_text()); cases=[c for c in source['cases'] if c['kind'] in ('NORMAL','PLACE_RECOVERY')]
 counts={k:sum(c['kind']==k for c in cases) for k in ('NORMAL','PLACE_RECOVERY')}
 if counts!={'NORMAL':50,'PLACE_RECOVERY':50}:raise RuntimeError(f'frozen source manifest cannot supply required 50+50: {counts}')
 growth={'version':'final-stage-growth-normal-place-v1','source_manifest':str(SOURCE.resolve()),'source_manifest_sha256':sha(SOURCE),'same_cases_for_all_models_and_checkpoints':True,'independent_from_e2_formal':True,'counts':counts,'cases':cases}
 (OUT/'growth_analysis').mkdir(exist_ok=True);(OUT/'growth_analysis/growth_normal_place_manifest.json').write_text(json.dumps(growth,indent=2)+'\n')
 model_audit={}
 for name,d in MODELS.items():
  checkpoints=[str((d/'checkpoints'/f'step_{s:06d}.pt').resolve()) for s in STEPS]
  missing=[x for x in checkpoints if not Path(x).is_file()]
  model_audit[name]={'training_dir':str(d.resolve()),'all_10k_to_120k_checkpoints_present':not missing,'missing_checkpoints':missing,'training_report_present':(d/'training_report.json').is_file() or (d/'final_training_report.json').is_file()}
 if not all(x['all_10k_to_120k_checkpoints_present'] and x['training_report_present'] for x in model_audit.values()):raise RuntimeError(f'model checkpoint freeze audit failed: {model_audit}')
 ambiguity=json.loads(AMB.read_text()); cuda=bool(torch.cuda.is_available())
 report={'PROTOCOL_FROZEN':'YES','NO_MODEL_RETRAINING':'YES','growth_manifest':str((OUT/'growth_analysis/growth_normal_place_manifest.json').resolve()),'growth_model_audit':model_audit,'ambiguity_analysis':'COMPLETE','ambiguity_key_results':{'cross_stage_state_distance_mean':ambiguity['cross_stage_state_distance']['mean'],'random_cross_stage_state_distance_mean':ambiguity['random_cross_stage_state_distance']['mean'],'cross_stage_action_distance_mean':ambiguity['cross_stage_action_distance']['mean'],'same_stage_action_distance_mean':ambiguity['same_stage_action_distance']['mean'],'local_action_variance_reduction_mean':ambiguity['statistics']['local_variance_reduction_mean']},'growth_evaluation':'WAITING_FOR_CUDA','e2_gamma_calibration':'WAITING_FOR_CUDA','e2_final':'WAITING_FOR_CUDA','CUDA_AVAILABLE':cuda,'BLOCKER':'CUDA unavailable in this environment; no rollout metrics or gamma selection were fabricated.' if not cuda else None}
 (OUT/'FINAL_STAGE_AMBIGUITY_EXPERIMENT_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
 (OUT/'FINAL_STAGE_AMBIGUITY_EXPERIMENT_REPORT.md').write_text('# Final Stage Ambiguity Experiments\n\n- Protocol frozen: YES\n- Model retraining: prohibited and not performed\n- Ambiguity analysis: complete\n- Growth/E2 rollout execution: waiting for CUDA\n\n'+json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
