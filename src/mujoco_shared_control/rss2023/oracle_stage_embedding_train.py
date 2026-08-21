"""Training loop for Oracle Stage Embedding V2."""
from __future__ import annotations
from dataclasses import asdict
import json, random, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from mujoco_shared_control.rss2023.oracle_stage_dataset import prepare_oracle_dataset
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage, TrainingConfig

def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _loader(prepared, split_name, batch_size, shuffle):
    split=getattr(prepared,split_name); obs=torch.from_numpy(prepared.observation_normalizer.normalize(split.observation)); act=torch.from_numpy(prepared.action_normalizer.normalize(split.action))
    return DataLoader(TensorDataset(obs,act),batch_size=min(batch_size,len(split)),shuffle=shuffle,drop_last=False,num_workers=0,pin_memory=torch.cuda.is_available())

def _infinite(loader):
    while True: yield from loader

@torch.no_grad()
def _val(model,loader,device,max_batches):
    model.eval(); values=[]
    with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
        torch.manual_seed(12345)
        for i,(obs,act) in enumerate(loader):
            if i>=max_batches: break
            values.append(float(model.loss(obs.to(device),act.to(device)).item()))
    model.train(); return float(np.mean(values))

def _save(payload,path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(payload,tmp); tmp.replace(path)

def train_v2(dataset_dir: Path, output: Path, *, device_name='auto', smoke_steps=500, training_config=TrainingConfig(steps=80000,validation_every=1000,checkpoint_every=10000)):
    prepared=prepare_oracle_dataset(dataset_dir); output.mkdir(parents=True,exist_ok=True)
    (output/'stage_label_audit.json').write_text(json.dumps(prepared.stage_audit,indent=2)+'\n')
    with (output/'stage_distribution.csv').open('w') as f:
        f.write('split,stage0,stage1,stage2,stage3,stage4\n')
        for name in ('train','validation','test'): f.write(name+','+','.join(map(str,prepared.stage_audit[name]['stage_counts']))+'\n')
    cfg=StageEmbeddingDiffusionConfig(); training_config=TrainingConfig(**{**asdict(training_config),'steps':80000,'validation_every':1000,'checkpoint_every':10000})
    (output/'dataset_adapter_report.json').write_text(json.dumps(prepared.manifest(),indent=2)+'\n'); np.savez(output/'normalization_stats.npz',observation_mean=prepared.observation_normalizer.mean,observation_std=prepared.observation_normalizer.std,action_mean=prepared.action_normalizer.mean,action_std=prepared.action_normalizer.std)
    global_cfg=json.loads((Path(__file__).resolve().parents[3]/'outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z/training_config.json').read_text()); v1_cfg=json.loads((Path(__file__).resolve().parents[3]/'outputs/oracle_stage_diffusion/oracle_stage_global_v1_20260818/config.json').read_text())
    config={'implementation':'mujoco_shared_control.rss2023.oracle_stage_embedding_model.StageEmbeddingDiffusion','observation_mode':'physical43_stage_encoder32_latent_concat','physical_dim':43,'stage_dim':5,'stage_embedding_dim':32,'condition_hidden_dim':128,'action_dim':7,'diffusion':cfg.state_dict(),'training':asdict(training_config)}; (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    diff={'global':global_cfg,'v1_raw_concat':v1_cfg,'v2':config,'architectural_change':'physical43->128; stage5->32; concat160->projection128; denoiser hidden remains128'}; (output/'v1_vs_v2_architecture_diff.json').write_text(json.dumps(diff,indent=2)+'\n')
    model=StageEmbeddingDiffusion(cfg); global_model_params=sum(p.numel() for p in torch.nn.ModuleList([]).parameters())
    parameter_count={'global_or_v1_reference':'see frozen config/checkpoint','v2_total_trainable_parameters':sum(p.numel() for p in model.parameters()),'v2_condition_encoder_parameters':sum(p.numel() for p in model.condition_encoder.parameters()),'v2_denoiser_parameters':sum(p.numel() for p in model.denoiser.parameters())}; (output/'parameter_count.json').write_text(json.dumps(parameter_count,indent=2)+'\n')
    device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name)
    _seed(7001); smoke=StageEmbeddingDiffusion(cfg).to(device).train(); opt=Adam(smoke.parameters(),lr=training_config.learning_rate); batches=_infinite(_loader(prepared,'train',training_config.batch_size,True)); last=float('nan')
    for _ in range(smoke_steps):
        obs,act=next(batches);obs,act=obs.to(device),act.to(device);opt.zero_grad(set_to_none=True);loss=smoke.loss(obs,act)
        if not torch.isfinite(loss): raise FloatingPointError('V2 smoke loss NaN/Inf')
        loss.backward();torch.nn.utils.clip_grad_norm_(smoke.parameters(),1.0);opt.step();last=float(loss.item())
    _save({'model':smoke.state_dict(),'diffusion_config':cfg.state_dict()},output/'smoke.pt');(output/'smoke_report.json').write_text(json.dumps({'status':'PASS','steps':smoke_steps,'last_loss':last,'nan_inf':0},indent=2)+'\n')
    _seed(training_config.seed); train_loader=_loader(prepared,'train',training_config.batch_size,True);val_loader=_loader(prepared,'validation',training_config.batch_size,False);batches=_infinite(train_loader);model=StageEmbeddingDiffusion(cfg).to(device).train();optimizer=Adam(model.parameters(),lr=training_config.learning_rate);ema=ExponentialMovingAverage(model,training_config.ema_decay);best=float('inf');final=float('nan');window=[];log_path=output/'training_log.jsonl';started=time.monotonic()
    def payload(step,value): return {'format_version':'oracle-stage-embedding-rss2023-1.0','step':step,'validation_loss':value,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'diffusion_config':cfg.state_dict(),'training_config':asdict(training_config),'observation_normalizer':prepared.observation_normalizer.state_dict(),'action_normalizer':prepared.action_normalizer.state_dict(),'dataset_manifest':prepared.manifest()}
    with log_path.open('w') as log:
        for step in range(1,training_config.steps+1):
            obs,act=next(batches);obs,act=obs.to(device),act.to(device);optimizer.zero_grad(set_to_none=True);loss=model.loss(obs,act)
            if not torch.isfinite(loss): raise FloatingPointError(f'V2 loss NaN/Inf at {step}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();ema.update(model);window.append(float(loss.item()))
            if step==1 or step%training_config.validation_every==0 or step==training_config.steps:
                final=_val(model,val_loader,device,training_config.validation_batches);record={'step':step,'train_loss':float(np.mean(window)),'validation_loss':final,'elapsed_seconds':time.monotonic()-started};log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True);window.clear();_save(payload(step,final),output/'latest.pt')
                if final<best:best=final;_save(payload(step,final),output/'best.pt')
            if step%training_config.checkpoint_every==0:_save(payload(step,final),output/'checkpoints'/f'step_{step:08d}.pt')
    history=[json.loads(x) for x in log_path.read_text().splitlines()];best_step=next((x['step'] for x in reversed(history) if x['validation_loss']==best),None);(output/'training_report.json').write_text(json.dumps({'status':'PASS','best_validation_loss':best,'final_validation_loss':final,'best_step':best_step,'steps':80000,'nan_inf':0,'device':str(device)},indent=2)+'\n');return output/'best.pt'
