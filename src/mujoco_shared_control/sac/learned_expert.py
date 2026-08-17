"""Conservative deterministic joint Actor-Critic training primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from mujoco_shared_control.sac.agent import bootstrap_mask, polyak_update
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.critic import TwinSACCritic
from mujoco_shared_control.sac.replay_buffer import ReplayBatch


@dataclass(frozen=True)
class LearnedExpertConfig:
    gamma: float = .995
    tau: float = .005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    replay_seed_transitions: int = 256
    total_env_steps: int = 200_000
    gradient_clip: float = 1.0
    seed: int = 20260815

    def __post_init__(self) -> None:
        if self.gamma != .995 or self.tau != .005:
            raise ValueError("Learned Expert v1 gamma/tau are frozen")
        if self.batch_size != 256 or self.replay_seed_transitions < self.batch_size:
            raise ValueError("Learned Expert v1 requires a normal 256-row replay seed")


@torch.no_grad()
def deterministic_td_target(
    actor: SACConstrainedGaussianActor, target: TwinSACCritic, batch: ReplayBatch,
    mean: torch.Tensor, std: torch.Tensor, gamma: float = .995,
) -> torch.Tensor:
    next_obs=(batch.next_observation-mean)/std
    next_action=actor.deterministic_action(next_obs)
    q1,q2=target(next_obs,next_action)
    return batch.reward+gamma*bootstrap_mask(batch.terminated)*torch.minimum(q1,q2)


def critic_step(critic: TwinSACCritic,target: TwinSACCritic,actor: SACConstrainedGaussianActor,
                optimizer: torch.optim.Optimizer,batch: ReplayBatch,mean: torch.Tensor,
                std: torch.Tensor,config: LearnedExpertConfig) -> dict[str,float]:
    y=deterministic_td_target(actor,target,batch,mean,std,config.gamma)
    obs=(batch.observation-mean)/std;q1,q2=critic(obs,batch.action)
    loss1=F.mse_loss(q1,y);loss2=F.mse_loss(q2,y);loss=loss1+loss2
    optimizer.zero_grad(set_to_none=True);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(critic.parameters(),config.gradient_clip)
    optimizer.step();polyak_update(critic,target,config.tau)
    return {"critic_loss":float(loss.detach()),"critic_gradient_norm":float(grad),
            "q1_mean":float(q1.mean().detach()),"q1_std":float(q1.std(unbiased=False).detach()),
            "q2_mean":float(q2.mean().detach()),"q2_std":float(q2.std(unbiased=False).detach()),
            "target_mean":float(y.mean()),"target_std":float(y.std(unbiased=False)),
            "q_disagreement":float((q1-q2).abs().mean().detach())}


def actor_losses(actor:SACConstrainedGaussianActor,critic:TwinSACCritic,
                 online_observation:torch.Tensor,expert_observation:torch.Tensor,
                 expert_action:torch.Tensor,mean:torch.Tensor,std:torch.Tensor
                 )->tuple[torch.Tensor,torch.Tensor,dict[str,float]]:
    online=(online_observation-mean)/std;expert=(expert_observation-mean)/std
    policy=actor.deterministic_action(online);q1,q2=critic(online,policy)
    q_loss=-torch.minimum(q1,q2).mean()
    bc_action=actor.deterministic_action(expert);bc_loss=F.mse_loss(bc_action,expert_action)
    return q_loss,bc_loss,{"q_pi_mean":float(torch.minimum(q1,q2).mean().detach()),
                           "bc_mse":float(bc_loss.detach())}


def gradient_norm(loss:torch.Tensor,parameters:list[torch.nn.Parameter])->float:
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=False)
    return float(torch.sqrt(sum((gradient.detach()**2).sum() for gradient in gradients)))


def calibrate_lambda(q_loss:torch.Tensor,bc_loss:torch.Tensor,
                     parameters:list[torch.nn.Parameter],epsilon:float=1e-12)->dict[str,float]:
    gq=gradient_norm(q_loss,parameters);gbc=gradient_norm(bc_loss,parameters)
    value=gq/(gbc+epsilon)
    if not torch.isfinite(torch.tensor(value)) or value<=0:raise FloatingPointError("invalid lambda_BC")
    return {"g_q":gq,"g_bc":gbc,"lambda_bc":value}


def actor_step(actor:SACConstrainedGaussianActor,critic:TwinSACCritic,
               optimizer:torch.optim.Optimizer,online_observation:torch.Tensor,
               expert_observation:torch.Tensor,expert_action:torch.Tensor,
               mean:torch.Tensor,std:torch.Tensor,lambda_bc:float,
               gradient_clip:float=1.0)->dict[str,float]:
    critic.requires_grad_(False)
    q_loss,bc_loss,metrics=actor_losses(actor,critic,online_observation,
                                       expert_observation,expert_action,mean,std)
    loss=q_loss+lambda_bc*bc_loss
    optimizer.zero_grad(set_to_none=True);loss.backward()
    grad=torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],gradient_clip)
    optimizer.step();critic.requires_grad_(True)
    return {**metrics,"actor_q_loss":float(q_loss.detach()),"actor_bc_loss":float(bc_loss.detach()),
            "actor_loss":float(loss.detach()),"actor_gradient_norm":float(grad)}


def freeze_log_std(actor:SACConstrainedGaussianActor)->None:
    actor.log_std_head.requires_grad_(False)
