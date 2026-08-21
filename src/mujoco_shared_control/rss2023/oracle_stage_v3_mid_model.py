"""V3-Mid: frozen SRDP-style conditioning with a 32->64 stage MLP."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mujoco_shared_control.rss2023.model import ConditionalDenoiser, _extract, make_beta_schedule

@dataclass(frozen=True)
class V3MidConfig:
    physical_dim:int=43; stage_posterior_dim:int=5; stage_condition_dim:int=64; action_dim:int=7
    num_diffusion_steps:int=50; beta_schedule:str='sigmoid'; beta_min:float=1e-4; beta_max:float=.26; hidden_dim:int=128
    @property
    def observation_dim(self): return self.physical_dim+self.stage_posterior_dim
    @property
    def denoiser_condition_dim(self): return self.physical_dim+self.stage_condition_dim
    def state_dict(self)->dict[str,Any]:
        x=asdict(self); x['observation_dim']=self.observation_dim; x['denoiser_condition_dim']=self.denoiser_condition_dim; return x

class StageMLP(nn.Module):
    def __init__(self,c:V3MidConfig): super().__init__(); self.net=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,64),nn.SiLU())
    def forward(self,x): return self.net(x)

class V3MidDiffusion(nn.Module):
    def __init__(self,c:V3MidConfig=V3MidConfig()):
        super().__init__(); self.config=c; self.stage_mlp=StageMLP(c)
        dc=type('D',(),{'observation_dim':c.denoiser_condition_dim,'action_dim':c.action_dim,'num_diffusion_steps':c.num_diffusion_steps,'hidden_dim':c.hidden_dim})(); self.denoiser=ConditionalDenoiser(dc)
        sc=type('S',(),{'num_diffusion_steps':c.num_diffusion_steps,'beta_min':c.beta_min,'beta_max':c.beta_max,'beta_schedule':c.beta_schedule})(); b=make_beta_schedule(sc); self.register_buffer('betas',b); self.register_buffer('alphas',1-b); ac=torch.cumprod(1-b,0); self.register_buffer('alphas_cumprod',ac); self.register_buffer('sqrt_alphas_cumprod',torch.sqrt(ac)); self.register_buffer('sqrt_one_minus_alphas_cumprod',torch.sqrt(1-ac))
    def _parts(self,o):
        if o.shape[-1]!=48: raise ValueError('V3-Mid expects state48')
        return o[...,:43],self.stage_mlp(o[...,43:])
    def q_sample(self,a,t,noise=None):
        noise=torch.randn_like(a) if noise is None else noise; return _extract(self.sqrt_alphas_cumprod,t,a)*a+_extract(self.sqrt_one_minus_alphas_cumprod,t,a)*noise,noise
    def predicted_noise(self,o,a,t):
        p,s=self._parts(o); x=torch.cat((p,s,a),-1); return self.denoiser(x,t)[...,self.config.denoiser_condition_dim:]
    def loss(self,o,a,t=None):
        t=torch.randint(self.config.num_diffusion_steps,(o.shape[0],),device=o.device) if t is None else t; n,z=self.q_sample(a,t); return F.mse_loss(self.predicted_noise(o,n,t),z)
    @torch.no_grad()
    def assist(self,o,h,gamma,*,generator=None):
        squeeze=o.ndim==1
        if squeeze:o,h=o.unsqueeze(0),h.unsqueeze(0)
        step=int((self.config.num_diffusion_steps-1)*gamma)
        if step==0:return h.squeeze(0) if squeeze else h.clone()
        t=torch.full((h.shape[0],),step,dtype=torch.long,device=h.device); a,_=self.q_sample(h,t,torch.randn(h.shape,device=h.device,generator=generator))
        for k in reversed(range(step)):
            tt=torch.full((a.shape[0],),k,dtype=torch.long,device=a.device); eps=self.predicted_noise(o,a,tt); alpha=_extract(self.alphas,tt,a); scale=_extract(self.sqrt_one_minus_alphas_cumprod,tt,a); mean=(a-((1-alpha)/scale)*eps)/torch.sqrt(alpha); a=mean+torch.sqrt(_extract(self.betas,tt,a))*torch.randn(a.shape,device=a.device,generator=generator) if k>0 else mean
        return a.squeeze(0) if squeeze else a
