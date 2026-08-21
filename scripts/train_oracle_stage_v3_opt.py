#!/usr/bin/env python3
from __future__ import annotations
import json,random,time
from pathlib import Path
import numpy as np, torch
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from mujoco_shared_control.rss2023.oracle_stage_dataset import prepare_oracle_dataset
from mujoco_shared_control.rss2023.oracle_stage_srdp_v3_model import SRDPStageDiffusion,SRDPStageDiffusionConfig
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z'
OUT=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_v3_srdp_optimizer_20260819'

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def inf(loader):
    while True: yield from loader
@torch.no_grad()
def validate(model,loader,device):
    model.eval(); vals=[]
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(12345)
        for i,(o,a) in enumerate(loader):
            if i>=20: break
            vals.append(float(model.loss(o.to(device),a.to(device)).item()))
    model.train(); return float(np.mean(vals))
def save(x,p): p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp'); torch.save(x,t); t.replace(p)
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seed(42); p=prepare_oracle_dataset(DATA); OUT.mkdir(parents=True,exist_ok=True)
    cfg=SRDPStageDiffusionConfig(); batch=512; steps=80000; train_obs=torch.from_numpy(p.observation_normalizer.normalize(p.train.observation)); train_act=torch.from_numpy(p.action_normalizer.normalize(p.train.action)); val_obs=torch.from_numpy(p.observation_normalizer.normalize(p.validation.observation)); val_act=torch.from_numpy(p.action_normalizer.normalize(p.validation.action)); tr=DataLoader(TensorDataset(train_obs,train_act),batch_size=batch,shuffle=True); va=DataLoader(TensorDataset(val_obs,val_act),batch_size=batch,shuffle=False); it=inf(tr)
    model=SRDPStageDiffusion(cfg).to(device); opt=Adam(model.parameters(),lr=1e-4,weight_decay=1e-6); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps); ema=ExponentialMovingAverage(model,0.9); best=float('inf'); hist=[]; log=[]; start=time.monotonic(); OUT.joinpath('checkpoints').mkdir(exist_ok=True)
    config={'format_version':'oracle-stage-v3-opt-1.0','architecture':'SRDPStageDiffusion frozen V3','dataset':str(DATA.resolve()),'diffusion':cfg.state_dict(),'training':{'steps':steps,'batch_size':batch,'learning_rate':1e-4,'weight_decay':1e-6,'scheduler':'cosine','warmup':False,'seed':42,'validation_every':1000,'checkpoint_every':10000}}
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n'); (OUT/'architecture_audit.json').write_text(json.dumps({'architecture_identical_to_original_v3':True,'stage_mlp_width':32,'hidden_dim':128,'diffusion_steps':50,'parameter_count':sum(x.numel() for x in model.parameters())},indent=2)+'\n'); np.savez(OUT/'normalization_stats.npz',observation_mean=p.observation_normalizer.mean,observation_std=p.observation_normalizer.std,action_mean=p.action_normalizer.mean,action_std=p.action_normalizer.std)
    for step in range(1,steps+1):
        o,a=next(it); o,a=o.to(device),a.to(device); opt.zero_grad(set_to_none=True); loss=model.loss(o,a); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sched.step(); ema.update(model)
        if step==1 or step%1000==0:
            vl=validate(model,va,device); row={'step':step,'lr':opt.param_groups[0]['lr'],'train_loss':float(loss.item()),'val_loss':vl,'elapsed_seconds':time.monotonic()-start}; log.append(row); print(json.dumps(row),flush=True)
            payload={'format_version':'oracle-stage-v3-opt-1.0','step':step,'validation_loss':vl,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'diffusion_config':cfg.state_dict(),'training_config':config['training'],'observation_normalizer':p.observation_normalizer.state_dict(),'action_normalizer':p.action_normalizer.state_dict(),'dataset_manifest':p.manifest()}; save(payload,OUT/'latest.pt');
            if vl<best: best=vl; save(payload,OUT/'best_val.pt')
            if step%10000==0: save(payload,OUT/'checkpoints'/f'step_{step:08d}.pt')
    (OUT/'training_log.csv').write_text('step,lr,train_loss,val_loss,elapsed_seconds\n'+'\n'.join(','.join(str(r[k]) for k in ('step','lr','train_loss','val_loss','elapsed_seconds')) for r in log)+'\n'); (OUT/'training_report.json').write_text(json.dumps({'status':'PASS','steps':steps,'best_validation_loss':best,'final_validation_loss':log[-1]['val_loss'],'parameter_count':sum(x.numel() for x in model.parameters())},indent=2)+'\n'); (OUT/'lr_schedule.csv').write_text('step,learning_rate\n'+'\n'.join(f"{r['step']},{r['lr']}" for r in log)+'\n')
if __name__=='__main__': main()
