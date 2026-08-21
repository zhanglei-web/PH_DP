"""80k training loop for SRDP-style lightweight Oracle Stage V3."""
from __future__ import annotations
from dataclasses import asdict
import json,random,time
from pathlib import Path
import numpy as np,torch
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from mujoco_shared_control.rss2023.oracle_stage_dataset import prepare_oracle_dataset
from mujoco_shared_control.rss2023.oracle_stage_srdp_v3_model import SRDPStageDiffusion,SRDPStageDiffusionConfig
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage,TrainingConfig
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def loader(p,name,batch,shuffle):
 x=getattr(p,name);o=torch.from_numpy(p.observation_normalizer.normalize(x.observation));a=torch.from_numpy(p.action_normalizer.normalize(x.action));return DataLoader(TensorDataset(o,a),batch_size=min(batch,len(x)),shuffle=shuffle,num_workers=0)
def infinite(l):
 while True:yield from l
@torch.no_grad()
def val(m,l,d,n):
 m.eval();z=[]
 with torch.random.fork_rng(devices=[d.index or 0] if d.type=='cuda' else []):
  torch.manual_seed(12345)
  for i,(o,a) in enumerate(l):
   if i>=n:break
   z.append(float(m.loss(o.to(d),a.to(d)).item()))
 m.train();return float(np.mean(z))
def save(payload,path):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)
def train_v3(dataset_dir,output,*,device_name='auto',smoke_steps=500,training_config=TrainingConfig(steps=80000,validation_every=1000,checkpoint_every=10000)):
 p=prepare_oracle_dataset(dataset_dir);output.mkdir(parents=True,exist_ok=True);(output/'stage_label_audit.json').write_text(json.dumps(p.stage_audit,indent=2)+'\n')
 with (output/'stage_distribution.csv').open('w') as f:
  f.write('split,stage0,stage1,stage2,stage3,stage4\n')
  for n in ('train','validation','test'):f.write(n+','+','.join(map(str,p.stage_audit[n]['stage_counts']))+'\n')
 cfg=SRDPStageDiffusionConfig();training_config=TrainingConfig(**{**asdict(training_config),'steps':80000,'validation_every':1000,'checkpoint_every':10000});(output/'dataset_adapter_report.json').write_text(json.dumps(p.manifest(),indent=2)+'\n');np.savez(output/'normalization_stats.npz',observation_mean=p.observation_normalizer.mean,observation_std=p.observation_normalizer.std,action_mean=p.action_normalizer.mean,action_std=p.action_normalizer.std)
 global_cfg=json.loads((Path(__file__).resolve().parents[3]/'outputs/global_diffusion/global_diffusion_v2_expert20000_20260816T230000Z/training_config.json').read_text());v1_cfg=json.loads((Path(__file__).resolve().parents[3]/'outputs/oracle_stage_diffusion/oracle_stage_global_v1_20260818/config.json').read_text());v2_cfg=json.loads((Path(__file__).resolve().parents[3]/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/config.json').read_text());cfgout={'implementation':'mujoco_shared_control.rss2023.oracle_stage_srdp_v3_model.SRDPStageDiffusion','observation_mode':'physical43_plus_additional_stage_condition','physical_dim':43,'stage_posterior_dim':5,'stage_condition_dim':32,'action_dim':7,'diffusion':cfg.state_dict(),'training':asdict(training_config)};(output/'config.json').write_text(json.dumps(cfgout,indent=2)+'\n');(output/'v3_architecture_audit.json').write_text(json.dumps({'global_forward_path':'physical43 + noisy_action -> ConditionalDenoiser(hidden128)','v1_forward_path':'physical43 + raw_stage5 + noisy_action -> ConditionalDenoiser(hidden128)','v2_forward_path':'physical43 encoder128 + stage encoder32 -> concat160 -> projection128 -> denoiser','v3_forward_path':'physical43 + StageMLP(5->32) + noisy_action -> shared ConditionalDenoiser(hidden128)','global_observation_encoder_preserved':True,'stage_used_as_additional_denoiser_condition':True,'transition_matrix_used':False,'v2_heavy_latent_path_used':False},indent=2)+'\n');(output/'architecture_config_sources.json').write_text(json.dumps({'global':global_cfg,'v1':v1_cfg,'v2':v2_cfg,'v3':cfgout},indent=2)+'\n')
 device=torch.device('cuda' if device_name=='auto' and torch.cuda.is_available() else 'cpu' if device_name=='auto' else device_name);seed(7001);smoke=SRDPStageDiffusion(cfg).to(device);opt=Adam(smoke.parameters(),lr=training_config.learning_rate);b=infinite(loader(p,'train',training_config.batch_size,True));last=float('nan')
 for _ in range(smoke_steps):
  o,a=next(b);o,a=o.to(device),a.to(device);opt.zero_grad(set_to_none=True);loss=smoke.loss(o,a)
  if not torch.isfinite(loss):raise FloatingPointError('V3 smoke NaN/Inf')
  loss.backward();torch.nn.utils.clip_grad_norm_(smoke.parameters(),1.);opt.step();last=float(loss.item())
 save({'model':smoke.state_dict(),'diffusion_config':cfg.state_dict()},output/'smoke.pt');(output/'smoke_report.json').write_text(json.dumps({'status':'PASS','steps':smoke_steps,'last_loss':last,'nan_inf':0},indent=2)+'\n')
 seed(training_config.seed);trainl=loader(p,'train',training_config.batch_size,True);vall=loader(p,'validation',training_config.batch_size,False);b=infinite(trainl);model=SRDPStageDiffusion(cfg).to(device);optimizer=Adam(model.parameters(),lr=training_config.learning_rate);ema=ExponentialMovingAverage(model,training_config.ema_decay);best=float('inf');final=float('nan');window=[];log=output/'training_log.jsonl';start=time.monotonic()
 def payload(step,value):return {'format_version':'oracle-stage-srdp-v3-1.0','step':step,'validation_loss':value,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'diffusion_config':cfg.state_dict(),'training_config':asdict(training_config),'observation_normalizer':p.observation_normalizer.state_dict(),'action_normalizer':p.action_normalizer.state_dict(),'dataset_manifest':p.manifest()}
 with log.open('w') as f:
  for step in range(1,80001):
   o,a=next(b);o,a=o.to(device),a.to(device);optimizer.zero_grad(set_to_none=True);loss=model.loss(o,a)
   if not torch.isfinite(loss):raise FloatingPointError(f'V3 loss NaN/Inf at {step}')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();ema.update(model);window.append(float(loss.item()))
   if step==1 or step%1000==0:
    final=val(model,vall,device,training_config.validation_batches);r={'step':step,'train_loss':float(np.mean(window)),'validation_loss':final,'elapsed_seconds':time.monotonic()-start};f.write(json.dumps(r)+'\n');f.flush();print(json.dumps(r),flush=True);window.clear();save(payload(step,final),output/'latest.pt');
    if step%1000==0 and final<best:best=final;save(payload(step,final),output/'best.pt')
   if step%10000==0:save(payload(step,final),output/'checkpoints'/f'step_{step:08d}.pt')
 history=[json.loads(x) for x in log.read_text().splitlines()];best_step=next((x['step'] for x in reversed(history) if x['validation_loss']==best),None);(output/'training_report.json').write_text(json.dumps({'status':'PASS','best_validation_loss':best,'final_validation_loss':final,'best_step':best_step,'steps':80000,'nan_inf':0,'device':str(device)},indent=2)+'\n');return output/'best.pt'
