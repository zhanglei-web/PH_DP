"""SRDP-style lightweight Oracle stage conditioning for RSS2023 Diffusion."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mujoco_shared_control.rss2023.model import ConditionalDenoiser, _extract, make_beta_schedule

@dataclass(frozen=True)
class SRDPStageDiffusionConfig:
    physical_dim: int = 43
    stage_posterior_dim: int = 5
    stage_condition_dim: int = 32
    action_dim: int = 7
    num_diffusion_steps: int = 50
    beta_schedule: str = 'sigmoid'
    beta_min: float = 1e-4
    beta_max: float = 0.26
    hidden_dim: int = 128
    @property
    def observation_dim(self): return self.physical_dim + self.stage_posterior_dim
    @property
    def denoiser_condition_dim(self): return self.physical_dim + self.stage_condition_dim
    def state_dict(self) -> dict[str, Any]:
        value=asdict(self);value['observation_dim']=self.observation_dim;value['denoiser_condition_dim']=self.denoiser_condition_dim;return value

class StageMLP(nn.Module):
    def __init__(self, config: SRDPStageDiffusionConfig):
        super().__init__(); self.net=nn.Sequential(nn.Linear(config.stage_posterior_dim,config.stage_condition_dim),nn.SiLU())
    def forward(self, stage: Tensor) -> Tensor: return self.net(stage)

class SRDPStageDiffusion(nn.Module):
    def __init__(self, config: SRDPStageDiffusionConfig=SRDPStageDiffusionConfig()):
        super().__init__();self.config=config;self.stage_mlp=StageMLP(config)
        denoise_cfg=type('DenoiseConfig',(),{'observation_dim':config.denoiser_condition_dim,'action_dim':config.action_dim,'num_diffusion_steps':config.num_diffusion_steps,'hidden_dim':config.hidden_dim})()
        self.denoiser=ConditionalDenoiser(denoise_cfg)
        schedule_cfg=type('Schedule',(),{'num_diffusion_steps':config.num_diffusion_steps,'beta_min':config.beta_min,'beta_max':config.beta_max,'beta_schedule':config.beta_schedule})()
        betas=make_beta_schedule(schedule_cfg);self.register_buffer('betas',betas);self.register_buffer('alphas',1-betas);cum=torch.cumprod(1-betas,dim=0);self.register_buffer('alphas_cumprod',cum);self.register_buffer('sqrt_alphas_cumprod',torch.sqrt(cum));self.register_buffer('sqrt_one_minus_alphas_cumprod',torch.sqrt(1-cum))
    def _parts(self, observation):
        if observation.shape[-1]!=self.config.observation_dim: raise ValueError('unexpected V3 observation dimension')
        return observation[...,:self.config.physical_dim],self.stage_mlp(observation[...,self.config.physical_dim:])
    def q_sample(self,clean_action,timesteps,noise=None):
        noise=torch.randn_like(clean_action) if noise is None else noise;noisy=_extract(self.sqrt_alphas_cumprod,timesteps,clean_action)*clean_action+_extract(self.sqrt_one_minus_alphas_cumprod,timesteps,clean_action)*noise;return noisy,noise
    def predicted_noise(self,observation,noisy_action,timesteps):
        physical,stage=self._parts(observation);value=torch.cat((physical,stage,noisy_action),dim=-1);return self.denoiser(value,timesteps)[...,self.config.denoiser_condition_dim:]
    def loss(self,observation,clean_action,timesteps=None):
        batch=observation.shape[0];timesteps=torch.randint(self.config.num_diffusion_steps,(batch,),device=observation.device) if timesteps is None else timesteps;noisy,target=self.q_sample(clean_action,timesteps);return F.mse_loss(self.predicted_noise(observation,noisy,timesteps),target)
    @torch.no_grad()
    def assist(self,observation,human_action,gamma,*,generator=None):
        if not 0<=gamma<=1:raise ValueError('gamma must be between zero and one')
        squeeze=observation.ndim==1
        if squeeze:observation,human_action=observation.unsqueeze(0),human_action.unsqueeze(0)
        step=int((self.config.num_diffusion_steps-1)*gamma)
        if step==0:return human_action.squeeze(0) if squeeze else human_action.clone()
        ts=torch.full((human_action.shape[0],),step,dtype=torch.long,device=human_action.device);action,_=self.q_sample(human_action,ts,torch.randn(human_action.shape,dtype=human_action.dtype,device=human_action.device,generator=generator))
        for t in reversed(range(step)):
            current=torch.full((action.shape[0],),t,dtype=torch.long,device=action.device);eps=self.predicted_noise(observation,action,current);alpha=_extract(self.alphas,current,action);scale=_extract(self.sqrt_one_minus_alphas_cumprod,current,action);mean=(action-((1-alpha)/scale)*eps)/torch.sqrt(alpha);action=mean+torch.sqrt(_extract(self.betas,current,action))*torch.randn(action.shape,dtype=action.dtype,device=action.device,generator=generator) if t>0 else mean
        return action.squeeze(0) if squeeze else action
