#!/usr/bin/env python3
"""Finalize artifacts after the formal run's structural protective stop."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from mujoco_shared_control.sac.agent import SACCore,SACCoreConfig
from mujoco_shared_control.sac.evaluation import evaluate_sac

ACTOR=Path("outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt")
STEPS=(0,25000,50000,100000,150000)
def main():
 p=argparse.ArgumentParser();p.add_argument("run",type=Path);a=p.parse_args();run=a.run.resolve();results={}
 for step in STEPS:
  d=torch.load(run/"checkpoints"/f"step_{step:06d}.pt",map_location="cpu",weights_only=False)
  core=SACCore(ACTOR,SACCoreConfig());core.actor.load_state_dict(d["actor_state_dict"]);core.critics.load_state_dict(d["critic_state_dict"]);core.target_critics.load_state_dict(d["target_critic_state_dict"])
  results[str(step)]=evaluate_sac(core,list(range(300000,300100)),reward_version="sac_reward_v2_candidate")
  print(step,results[str(step)]["success"],flush=True)
 (run/"primary_evaluation.json").write_text(json.dumps({"checkpoints":results,"best_step":0,"protective_stop_step":165000},indent=2)+"\n")
 initial=results["0"]
 (run/"secondary_evaluation.json").write_text(json.dumps({"not_run":"no trained checkpoint qualified; selected checkpoint is unchanged step0","best_step":0},indent=2)+"\n")
 drift={}
 initial_state=torch.load(run/"checkpoints/step_000000.pt",map_location="cpu",weights_only=False)["actor_state_dict"]
 for step in STEPS:
  state=torch.load(run/"checkpoints"/f"step_{step:06d}.pt",map_location="cpu",weights_only=False)["actor_state_dict"]
  diffs=torch.cat([(state[k]-initial_state[k]).reshape(-1) for k in state])
  drift[str(step)]={"parameter_mae":float(diffs.abs().mean()),"parameter_max_abs":float(diffs.abs().max())}
 (run/"actor_drift.json").write_text(json.dumps(drift,indent=2)+"\n")
 failure={"status":"LEARNED_EXPERT_JOINT_V1_BLOCKED","protective_stop_step":165000,"reason":"non-finite joint metric after sustained Q explosion and 0/100 success","step0_success":initial["success"],"completed_checkpoints":list(STEPS),"missing_checkpoint":200000}
 (run/"failure_statistics.json").write_text(json.dumps(failure,indent=2)+"\n")
 (run/"training_manifest.json").write_text(json.dumps({"planned_env_steps":200000,"completed_env_steps":165000,"formal_run_complete":False,"protective_stop":True},indent=2)+"\n")
 (run/"config.json").write_text(json.dumps({"algorithm":"deterministic Twin-Q + online -Q Actor + nominal BC anchor","gamma":.995,"tau":.005,"actor_lr":3e-4,"critic_lr":3e-4,"batch_size":256,"expert_online_critic_batch":[128,128],"replay_seed_transitions":256,"lambda_calibration":{"g_q":25.89212989807129,"g_bc":.0028325372841209173,"lambda_bc":9140.96701712648},"alpha":False,"entropy":False,"final_test_seeds_used":False,"protective_stop":failure},indent=2)+"\n")
if __name__=="__main__":main()
