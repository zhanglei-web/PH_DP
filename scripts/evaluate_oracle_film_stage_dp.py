#!/usr/bin/env python3
"""CUDA-only ordinary closed-loop sweep for Oracle FiLM Stage-DP checkpoints."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from mujoco_shared_control.awac.milestones import phase_from_milestones
from mujoco_shared_control.awac.reward import AWACRewardV1Config, AWACRewardV1Online
from mujoco_shared_control.collection.automatic import CollectionConfig
from mujoco_shared_control.control.expert_command_adapter import ExpertCommandAdapter
from mujoco_shared_control.envs.pick_place_env import PickPlaceEnv
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.rss2023.global_evaluation import summarize
from mujoco_shared_control.rss2023.oracle_film_stage_model import OracleFiLMStageConfig, OracleFiLMStageDiffusion

ROOT=Path(__file__).resolve().parents[1];SEEDS=range(2_000_000,2_000_100);THRESH=.375
class Predictor:
 def __init__(self,ck,dev):
  p=torch.load(ck,map_location=dev,weights_only=False);self.payload=p;self.dev=dev;cfg=OracleFiLMStageConfig(**p['diffusion_config']);self.model=OracleFiLMStageDiffusion(cfg).to(dev).eval();self.model.load_state_dict(p['model']);n=p['normalization'];self.om=np.asarray(n['observation_mean'],np.float32);self.os=np.asarray(n['observation_std'],np.float32);self.am=np.asarray(n['action_mean'],np.float32);self.astd=np.asarray(n['action_std'],np.float32);self.gen=None
 def reset(self,seed):self.gen=torch.Generator(device=self.dev).manual_seed(seed)
 @torch.inference_mode()
 def sample(self,state,stage,seed):
  self.reset(seed);x=np.r_[state,np.eye(5,dtype=np.float32)[stage]];physical=torch.as_tensor(((x[:43]-self.om[:43])/self.os[:43]),device=self.dev).unsqueeze(0);z=torch.as_tensor(x[43:],device=self.dev).unsqueeze(0);human=torch.zeros((1,7),device=self.dev);a=self.model.assist(physical,z,human,1.,generator=self.gen)[0];return (a.cpu().numpy()*self.astd+self.am).astype(np.float32)
def episode(pred,env_seed,sampling_seed):
 c=CollectionConfig();env=PickPlaceEnv(render_mode=None,control_timestep=c.control_timestep_s,max_episode_steps=c.max_steps,enable_camera=False);ad=ExpertCommandAdapter(env.ik_controller,ExpertActionSpec());pred.reset(sampling_seed)
 try:
  ob,_=env.reset(seed=env_seed,options={'randomize_arm':c.randomize_arm,'arm_joint_noise_scale':c.arm_joint_noise_scale,'randomize_object':c.randomize_object,'randomize_goal':c.randomize_goal});ad.reset(ob['ee_pose'],ob['q_obs']);state=np.r_[env.get_policy_observation(ob),np.float32(bool(ob['object_grasped']))].astype(np.float32);rew=AWACRewardV1Online(state,AWACRewardV1Config());con=0;reason='timeout';ret=0.;clip=vals=aclip=0
  for step in range(c.max_steps):
   state=np.r_[env.get_policy_observation(ob),np.float32(bool(ob['object_grasped']))].astype(np.float32);stage=int(phase_from_milestones(rew.tracker.current));raw=pred.sample(state,stage,sampling_seed+step);out=(raw < -1)|(raw>1);vals+=int(out.sum());clip+=int(out.any());bounded=np.clip(raw,-1,1);bounded[6]=-1. if bounded[6]<THRESH else 1.;adapt=ad.adapt(ExpertActionSpec().denormalize(bounded));aclip+=int(adapt.action_clipped);con=0 if adapt.accepted else con+1;nob,*_=env.step(adapt.joint_target);ns=np.r_[env.get_policy_observation(nob),np.float32(bool(nob['object_grasped']))].astype(np.float32);rs=rew.step(state,ns,ik_failure=con>=c.max_consecutive_ik_failures,time_limit=step+1>=c.max_steps);ret+=rs.reward;ob=nob
   if rs.task_success:reason='task_success';break
   if rs.terminated or rs.truncated:reason=rs.termination_reason;break
  m=rew.tracker.current;success=reason=='task_success';return {'environment_seed':env_seed,'sampling_seed':sampling_seed,'success':success,'grasp':bool(m[0]),'lift':bool(m[1]),'transport':bool(m[2]),'place':bool(m[3]),'release':bool(m[3]),'retreat':bool(m[4]),'illegal_drop':reason=='illegal_drop','ik_failure':reason=='ik_failure_limit','timeout':reason=='timeout','termination_reason':reason,'failure_phase':None if success else ('RETREAT' if m[3] else 'APPROACH'),'episode_return':float(ret),'episode_length':step+1,'nan_count':0,'inf_count':0,'out_of_bounds_steps':clip,'out_of_bounds_values':vals,'policy_clip_steps':clip,'adapter_clip_steps':aclip}
 finally:env.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoints',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'outputs/oracle_film_stage_dp_v1/closed_loop');a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is forbidden')
 dev=torch.device('cuda:0');torch.cuda.set_device(dev);a.output.mkdir(parents=True,exist_ok=True);summary=[]
 for ck in a.checkpoints:
  pred=Predictor(ck,dev);step=int(pred.payload['step']);rows=[episode(pred,s,8_000_000+s) for s in SEEDS];report={'policy':'Oracle-FiLM-Stage-DP-MLP','checkpoint':str(ck.resolve()),'step':step,'stage_source':'GT one-hot','summary':summarize(rows),'rows':rows};out=a.output/f'step_{step:06d}';out.mkdir(exist_ok=True);(out/'evaluation_report.json').write_text(json.dumps(report,indent=2)+'\n');s=report['summary'];summary.append({'Step':step,'Success':s['success']['rate'],'Grasp':s['grasp']['rate'],'Lift':s['lift']['rate'],'Transport':s['transport']['rate'],'Place_Release':s['release']['rate'],'Retreat':s['retreat']['rate'],'Illegal_Drop':s['illegal_drop']['rate'],'IK':s['ik_failure']['rate'],'Timeout':s['timeout']['rate']});print(summary[-1],flush=True)
 (a.output/'checkpoint_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
