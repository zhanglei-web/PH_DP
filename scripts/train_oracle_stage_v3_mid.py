#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from mujoco_shared_control.rss2023.oracle_stage_dataset import prepare_oracle_dataset
from mujoco_shared_control.rss2023.oracle_stage_v3_mid_model import V3MidConfig,V3MidDiffusion
from mujoco_shared_control.rss2023.train import ExponentialMovingAverage
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z'; OUT=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_v3_mid_20260819'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--learning-rate',type=float,default=1e-3);ap.add_argument('--output',type=Path,default=OUT);args=ap.parse_args();out=args.output
 random.seed(42);np.random.seed(42);torch.manual_seed(42); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); p=prepare_oracle_dataset(DATA); c=V3MidConfig(); tr=DataLoader(TensorDataset(torch.from_numpy(p.observation_normalizer.normalize(p.train.observation)),torch.from_numpy(p.action_normalizer.normalize(p.train.action))),batch_size=512,shuffle=True); va=DataLoader(TensorDataset(torch.from_numpy(p.observation_normalizer.normalize(p.validation.observation)),torch.from_numpy(p.action_normalizer.normalize(p.validation.action))),batch_size=512); it=iter(tr); model=V3MidDiffusion(c).to(dev); opt=Adam(model.parameters(),lr=args.learning_rate); ema=ExponentialMovingAverage(model,.9); out.mkdir(parents=True,exist_ok=True); (out/'checkpoints').mkdir(exist_ok=True); np.savez(out/'normalization_stats.npz',observation_mean=p.observation_normalizer.mean,observation_std=p.observation_normalizer.std,action_mean=p.action_normalizer.mean,action_std=p.action_normalizer.std); (out/'config.json').write_text(json.dumps({'architecture':'V3-Mid','stage_mlp':'Linear(5,32),SiLU,Linear(32,64),SiLU','stage_condition_dim':64,'diffusion':c.state_dict(),'training':{'steps':80000,'batch_size':512,'learning_rate':args.learning_rate,'weight_decay':0.0,'scheduler':'none','warmup':False,'seed':42}},indent=2)+'\n'); log=[]; best=1e9; start=time.monotonic()
 def payload(step,vl): return {'format_version':'oracle-stage-v3-mid-1.0','step':step,'validation_loss':vl,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),'diffusion_config':c.state_dict(),'observation_normalizer':p.observation_normalizer.state_dict(),'action_normalizer':p.action_normalizer.state_dict(),'dataset_manifest':p.manifest()}
 for step in range(1,80001):
  try:o,a=next(it)
  except StopIteration:it=iter(tr);o,a=next(it)
  o,a=o.to(dev),a.to(dev);opt.zero_grad(set_to_none=True);loss=model.loss(o,a);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();ema.update(model)
  if step==1 or step%1000==0:
   model.eval(); vals=[]; torch.manual_seed(12345)
   with torch.no_grad():
    for j,(x,y) in enumerate(va):
     if j>=20:break
     vals.append(float(model.loss(x.to(dev),y.to(dev)).item()))
   model.train();vl=float(np.mean(vals)); r={'step':step,'learning_rate':args.learning_rate,'train_loss':float(loss.item()),'val_loss':vl,'elapsed_seconds':time.monotonic()-start};log.append(r);print(json.dumps(r),flush=True);torch.save(payload(step,vl),out/'latest.pt');
   if vl<best:best=vl;torch.save(payload(step,vl),out/'best_val.pt')
   if step%10000==0:torch.save(payload(step,vl),out/'checkpoints'/f'step_{step:08d}.pt')
 (out/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in log)+'\n');(out/'training_report.json').write_text(json.dumps({'status':'PASS','parameter_count':sum(x.numel() for x in model.parameters()),'best_validation_loss':best,'steps':80000},indent=2)+'\n')
if __name__=='__main__':main()
