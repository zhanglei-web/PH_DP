from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2
from mujoco_shared_control.sac.critic_pretraining import build_arrays_from_semantic_run
from mujoco_shared_control.sac.learned_expert import (
    LearnedExpertConfig, actor_losses, actor_step, calibrate_lambda,
    critic_step, deterministic_td_target, freeze_log_std,
)
from mujoco_shared_control.sac.replay_buffer import ReplayBatch


ALIGNED=Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")


def assets():
    actor,critic,target,payload=load_aligned_v2(ALIGNED)
    arrays,_=build_arrays_from_semantic_run(
        Path("manifests/rule_expert_v1_formal.json"),
        Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z"),
        reward_version="sac_reward_v2_candidate")
    return actor,critic,target,payload,arrays["train"]


def batch(data,start=0,n=256):
    sl=slice(start,start+n)
    return ReplayBatch(*(torch.from_numpy(getattr(data,name)[sl]) for name in
        ("observation","action","reward","next_observation","terminated","truncated")))


def test_deterministic_target_terminal_truncated_and_no_entropy() -> None:
    actor,_critic,target,payload,data=assets();b=batch(data,0,2)
    b=ReplayBatch(b.observation,b.action,torch.ones(2,1),b.next_observation,
                  torch.tensor([[True],[False]]),torch.tensor([[False],[True]]))
    y=deterministic_td_target(actor,target,b,payload["observation_mean"],payload["observation_std"])
    assert y[0].item()==1
    assert y[1].item()!=1


def test_actor_q_uses_online_and_bc_uses_nominal_expert() -> None:
    actor,critic,_target,payload,data=assets();freeze_log_std(actor)
    online=torch.from_numpy(data.observation[:32]);expert=torch.from_numpy(data.observation[64:96])
    action=torch.from_numpy(data.action[64:96])
    q,bc,_=actor_losses(actor,critic,online,expert,action,payload["observation_mean"],payload["observation_std"])
    parameters=[p for p in actor.parameters() if p.requires_grad]
    calibration=calibrate_lambda(q,bc,parameters)
    assert calibration["lambda_bc"]>0 and np.isfinite(list(calibration.values())).all()
    before={k:v.clone() for k,v in actor.log_std_head.state_dict().items()}
    optimizer=torch.optim.Adam(parameters,lr=3e-4)
    actor_step(actor,critic,optimizer,online,expert,action,payload["observation_mean"],
               payload["observation_std"],calibration["lambda_bc"])
    for k,v in before.items():torch.testing.assert_close(v,actor.log_std_head.state_dict()[k],rtol=0,atol=0)


def test_joint_critic_batch_is_exact_50_50_and_updates_finite() -> None:
    actor,critic,target,payload,data=assets();config=LearnedExpertConfig()
    left=batch(data,0,128);right=batch(data,128,128)
    mixed=ReplayBatch(*(torch.cat((getattr(left,name),getattr(right,name))) for name in
        ("observation","action","reward","next_observation","terminated","truncated")))
    assert len(left.observation)==len(right.observation)==128 and len(mixed.observation)==256
    optimizer=torch.optim.Adam(critic.parameters(),lr=config.critic_lr)
    metrics=critic_step(critic,target,actor,optimizer,mixed,payload["observation_mean"],
                        payload["observation_std"],config)
    assert np.isfinite(list(metrics.values())).all()
