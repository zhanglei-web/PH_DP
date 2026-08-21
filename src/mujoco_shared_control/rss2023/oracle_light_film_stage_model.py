"""Single-middle-layer Light-FiLM variant of the Oracle V1 vector denoiser."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mujoco_shared_control.rss2023.model import _extract, make_beta_schedule

@dataclass(frozen=True)
class OracleLightFiLMConfig:
    physical_dim: int = 43
    stage_dim: int = 5
    stage_embedding_dim: int = 16
    action_dim: int = 7
    num_diffusion_steps: int = 50
    beta_schedule: str = "sigmoid"
    beta_min: float = 1e-4
    beta_max: float = 0.26
    hidden_dim: int = 128
    def state_dict(self) -> dict[str, Any]: return asdict(self)

class OracleLightFiLMDenoiser(nn.Module):
    def __init__(self, config: OracleLightFiLMConfig):
        super().__init__(); self.config=config
        self.stage_encoder=nn.Sequential(nn.Linear(config.stage_dim,config.stage_embedding_dim),nn.SiLU())
        input_dim=config.physical_dim+config.action_dim
        self.layer1=nn.Linear(input_dim,config.hidden_dim); self.layer1_embedding=nn.Embedding(config.num_diffusion_steps,config.hidden_dim)
        self.layer2=nn.Linear(config.hidden_dim,config.hidden_dim); self.layer2_embedding=nn.Embedding(config.num_diffusion_steps,config.hidden_dim)
        self.layer3=nn.Linear(config.hidden_dim,config.hidden_dim); self.layer3_embedding=nn.Embedding(config.num_diffusion_steps,config.hidden_dim)
        self.film_head=nn.Linear(config.stage_embedding_dim,2*config.hidden_dim); nn.init.zeros_(self.film_head.weight); nn.init.zeros_(self.film_head.bias)
        self.output=nn.Linear(config.hidden_dim,input_dim)
    def forward(self,physical:Tensor,noisy_action:Tensor,timesteps:Tensor,stage:Tensor)->Tensor:
        e=self.stage_encoder(stage); values=torch.cat((physical,noisy_action),dim=-1)
        values=F.softplus(self.layer1(values)*self.layer1_embedding(timesteps))
        values=self.layer2(values)*self.layer2_embedding(timesteps)
        gamma,beta=self.film_head(e).chunk(2,dim=-1); values=F.softplus((1+gamma)*values+beta)
        values=F.softplus(self.layer3(values)*self.layer3_embedding(timesteps))
        return self.output(values)

class OracleLightFiLMStageDiffusion(nn.Module):
    def __init__(self,config:OracleLightFiLMConfig=OracleLightFiLMConfig()):
        super().__init__();self.config=config;self.denoiser=OracleLightFiLMDenoiser(config)
        schedule=type('Schedule',(),{'num_diffusion_steps':config.num_diffusion_steps,'beta_min':config.beta_min,'beta_max':config.beta_max,'beta_schedule':config.beta_schedule})();betas=make_beta_schedule(schedule);self.register_buffer('betas',betas);self.register_buffer('alphas',1-betas);ac=torch.cumprod(1-betas,0);self.register_buffer('sqrt_alphas_cumprod',torch.sqrt(ac));self.register_buffer('sqrt_one_minus_alphas_cumprod',torch.sqrt(1-ac))
    def q_sample(self,clean_action:Tensor,timesteps:Tensor,noise:Tensor|None=None):
        noise=torch.randn_like(clean_action) if noise is None else noise;return _extract(self.sqrt_alphas_cumprod,timesteps,clean_action)*clean_action+_extract(self.sqrt_one_minus_alphas_cumprod,timesteps,clean_action)*noise,noise
    def loss(self,physical:Tensor,stage:Tensor,clean_action:Tensor,timesteps:Tensor|None=None):
        timesteps=torch.randint(self.config.num_diffusion_steps,(physical.shape[0],),device=physical.device) if timesteps is None else timesteps;noisy,target=self.q_sample(clean_action,timesteps);return F.mse_loss(self.denoiser(physical,noisy,timesteps,stage)[...,-self.config.action_dim:],target)
    @torch.no_grad()
    def assist(self,physical:Tensor,stage:Tensor,human_action:Tensor,gamma:float,*,generator:torch.Generator|None=None):
        squeeze=physical.ndim==1
        if squeeze: physical,stage,human_action=physical.unsqueeze(0),stage.unsqueeze(0),human_action.unsqueeze(0)
        step=int((self.config.num_diffusion_steps-1)*gamma)
        if step==0:return human_action.squeeze(0) if squeeze else human_action.clone()
        ts=torch.full((human_action.shape[0],),step,dtype=torch.long,device=human_action.device);action,_=self.q_sample(human_action,ts,torch.randn(human_action.shape,device=human_action.device,generator=generator))
        for t in reversed(range(step)):
            cur=torch.full((action.shape[0],),t,dtype=torch.long,device=action.device);eps=self.denoiser(physical,action,cur,stage)[...,-self.config.action_dim:];alpha=_extract(self.alphas,cur,action);scale=_extract(self.sqrt_one_minus_alphas_cumprod,cur,action);mean=(action-((1-alpha)/scale)*eps)/torch.sqrt(alpha);action=mean if t==0 else mean+torch.sqrt(_extract(self.betas,cur,action))*torch.randn(action.shape,device=action.device,generator=generator)
        return action.squeeze(0) if squeeze else action
