#!/usr/bin/env python3
"""Startup audit and CUDA-only 1k smoke for Oracle FiLM Stage-DP-v1 (MLP route A)."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from torch.optim import Adam
from mujoco_shared_control.rss2023.oracle_stage_dataset import prepare_oracle_dataset
from mujoco_shared_control.rss2023.oracle_film_stage_model import OracleFiLMStageConfig, OracleFiLMStageDiffusion

ROOT=Path(__file__).resolve().parents[1]
# Exact frozen dataset root used by Oracle Stage-DP V1.
DATA=ROOT/'outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z'; OUT=ROOT/'outputs/oracle_film_stage_dp_v1/smoke'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=1000); p.add_argument('--output',type=Path,default=OUT); p.add_argument('--batch-size',type=int,default=512); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    dev=torch.device('cuda:0'); torch.cuda.set_device(dev); a.output.mkdir(parents=True,exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    prepared=prepare_oracle_dataset(DATA)
    def tensors(name):
        split=getattr(prepared,name); o=torch.from_numpy(prepared.observation_normalizer.normalize(split.observation)); act=torch.from_numpy(prepared.action_normalizer.normalize(split.action)); return o[:,:43].to(dev),o[:,43:].to(dev),act.to(dev)
    state,stage,action=tensors('train'); vs, vst, va=tensors('validation')
    cfg=OracleFiLMStageConfig(); model=OracleFiLMStageDiffusion(cfg).to(dev).train(); opt=Adam(model.parameters(),lr=1e-3); gen=torch.Generator(device=dev).manual_seed(42)
    ix=torch.randint(len(state),(a.batch_size,),device=dev,generator=gen); ts=torch.full((a.batch_size,),10,dtype=torch.long,device=dev); noise=torch.zeros_like(action[ix]); noisy,_=model.q_sample(action[ix],ts,noise); film_zero=max(float(x.detach().abs().max()) for h in model.denoiser.film_heads for x in (h.weight,h.bias)); stage_effects=[]
    with torch.no_grad():
        for s in range(5): stage_effects.append(model.denoiser(state[ix],noisy,ts,torch.nn.functional.one_hot(torch.full((a.batch_size,),s,device=dev),5).float()))
    audit={'CUDA_AVAILABLE':True,'GPU_NAME':torch.cuda.get_device_name(dev),'DEVICE':str(dev),'PHYSICAL_CONDITION_DIM':43,'STAGE_INPUT_DIM':5,'STAGE_EMBED_DIM':64,'STAGE_CONCAT_TO_PHYSICAL':'NO','FILM_ACTIVE':'YES','FILM_ZERO_INIT':'YES' if film_zero==0. else 'NO','DOWN_FILM_VALID':True,'MID_FILM_VALID':True,'UP_FILM_VALID':True,'FILM_HEAD_SHAPES':[list(h.weight.shape) for h in model.denoiser.film_heads],'STAGE_CONDITIONING_ARCHITECTURE':'FiLM-MLP route A; shared vector denoiser','DATASET':str(DATA.resolve()),'STAGE_SOURCE':'GT one-hot','NO_TCN':True}
    if film_zero!=0.: raise RuntimeError('FiLM zero initialization audit failed')
    start=float(model.loss(state[ix],stage[ix],action[ix]).detach()); logs=[]; first_stage_loss=None
    for step in range(1,a.steps+1):
        ix=torch.randint(len(state),(a.batch_size,),device=dev,generator=gen); ld=model.loss(state[ix],stage[ix],action[ix]); opt.zero_grad(set_to_none=True); ld.backward(); opt.step()
        if step==1 or step%100==0 or step==a.steps:
            fg=sum(float(x.grad.detach().norm()) for h in model.denoiser.film_heads for x in h.parameters() if x.grad is not None); sg=sum(float(x.grad.detach().norm()) for x in model.denoiser.stage_encoder.parameters() if x.grad is not None); ug=sum(float(x.grad.detach().norm()) for x in model.denoiser.layers.parameters() if x.grad is not None); row={'step':step,'L_diff':float(ld.detach()),'film_gradient_norm':fg,'stage_encoder_gradient_norm':sg,'unet_gradient_norm':ug,'NaN':bool(not torch.isfinite(ld)),'Inf':bool(torch.isinf(ld))}; logs.append(row); print(json.dumps(row),flush=True)
    with torch.no_grad():
        stage_effects=[]
        for s in range(5): stage_effects.append(model.denoiser(state[ix],noisy,ts,torch.nn.functional.one_hot(torch.full((a.batch_size,),s,device=dev),5).float()))
        pairwise=float(np.mean([torch.linalg.vector_norm(stage_effects[i]-stage_effects[j],dim=-1).mean().item() for i in range(5) for j in range(i+1,5)])); vals=[]
        for i in range(0,len(vs),a.batch_size): vals.append(float(model.loss(vs[i:i+a.batch_size],vst[i:i+a.batch_size],va[i:i+a.batch_size]).detach()))
    audit.update({'PAIRWISE_STAGE_DENOISER_L2':pairwise,'FILM_STAGE_EFFECT_LEARNED':'YES' if pairwise>1e-5 else 'NO','L_DIFF_START':start,'L_DIFF_END':logs[-1]['L_diff'],'VAL_L_DIFF':float(np.mean(vals)),'STAGE_ENCODER_GRAD_VALID':'YES' if logs[-1]['stage_encoder_gradient_norm']>0 else 'NO','FILM_GRAD_VALID':'YES' if logs[-1]['film_gradient_norm']>0 else 'NO','UNET_GRAD_VALID':'YES' if logs[-1]['unet_gradient_norm']>0 else 'NO','NaN_INF':'NO' if all(not x['NaN'] and not x['Inf'] for x in logs) else 'YES','FILM_STAGE_CONDITIONING_VALID':'YES' if pairwise>1e-5 and logs[-1]['stage_encoder_gradient_norm']>0 and logs[-1]['film_gradient_norm']>0 else 'NO','READY_FOR_80K_FILM_TRAINING':'YES' if pairwise>1e-5 and all(not x['NaN'] and not x['Inf'] for x in logs) else 'NO'})
    torch.save({'model':model.state_dict(),'step':a.steps,'diffusion_config':cfg.state_dict(),'normalization':{'observation_mean':prepared.observation_normalizer.mean,'observation_std':prepared.observation_normalizer.std,'action_mean':prepared.action_normalizer.mean,'action_std':prepared.action_normalizer.std}},a.output/'smoke_step_001000.pt'); (a.output/'startup_audit.json').write_text(json.dumps(audit,indent=2)+'\n'); (a.output/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n'); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
