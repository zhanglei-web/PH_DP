#!/usr/bin/env python3
"""CUDA-only formal 80k Oracle Light-FiLM Stage-DP-MLP training."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.optim import Adam
from mujoco_shared_control.rss2023.oracle_stage_dataset import prepare_oracle_dataset
from mujoco_shared_control.rss2023.oracle_light_film_stage_model import OracleLightFiLMConfig,OracleLightFiLMStageDiffusion
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/learned_expert_collection/final_online_awac20k_formal20000_v2_20260816T200000Z'
def main():
 p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=80000);p.add_argument('--output',type=Path,default=ROOT/'outputs/oracle_light_film_stage_dp_v1/formal');p.add_argument('--batch-size',type=int,default=512);a=p.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required; CPU fallback is forbidden')
 if a.steps<80000:raise ValueError('formal training requires at least 80000 steps')
 dev=torch.device('cuda:0');torch.cuda.set_device(dev);a.output.mkdir(parents=True,exist_ok=True);random.seed(42);np.random.seed(42);torch.manual_seed(42);prepared=prepare_oracle_dataset(DATA)
 def tensors(name):
  s=getattr(prepared,name);o=torch.from_numpy(prepared.observation_normalizer.normalize(s.observation));return o[:,:43].to(dev),o[:,43:].to(dev),torch.from_numpy(prepared.action_normalizer.normalize(s.action)).to(dev)
 state,stage,action=tensors('train');vs,vstage,va=tensors('validation');cfg=OracleLightFiLMConfig();model=OracleLightFiLMStageDiffusion(cfg).to(dev).train();opt=Adam(model.parameters(),lr=1e-3);gen=torch.Generator(device=dev).manual_seed(42);fixed=torch.arange(min(a.batch_size,len(state)),device=dev);ts=torch.full((len(fixed),),10,dtype=torch.long,device=dev);noisy,_=model.q_sample(action[fixed],ts,torch.zeros_like(action[fixed]));logs=[]
 def audit(step,loss,grads):
  model.eval();torch.manual_seed(1000+step);vals=[]
  with torch.no_grad():
   for i in range(0,len(vs),a.batch_size):vals.append(float(model.loss(vs[i:i+a.batch_size],vstage[i:i+a.batch_size],va[i:i+a.batch_size])))
   outs=[model.denoiser(state[fixed],noisy,ts,torch.nn.functional.one_hot(torch.full((len(fixed),),s,device=dev),5).float()) for s in range(5)]
  pair=float(np.mean([torch.linalg.vector_norm(outs[i]-outs[j],dim=-1).mean().item() for i in range(5) for j in range(i+1,5)]));model.train();return {'step':step,'train_L_diff':float(loss.detach()),'val_L_diff':float(np.mean(vals)),'stage_encoder_gradient_norm':grads[0],'film_gradient_norm':grads[1],'denoiser_gradient_norm':grads[2],'PAIRWISE_STAGE_DENOISER_L2':pair,'NaN':bool(not torch.isfinite(loss)),'Inf':bool(torch.isinf(loss))}
 for step in range(1,a.steps+1):
  ix=torch.randint(len(state),(a.batch_size,),device=dev,generator=gen);ld=model.loss(state[ix],stage[ix],action[ix]);opt.zero_grad(set_to_none=True);ld.backward();grads=(sum(float(x.grad.detach().norm()) for x in model.denoiser.stage_encoder.parameters() if x.grad is not None),sum(float(x.grad.detach().norm()) for x in model.denoiser.film_head.parameters() if x.grad is not None),sum(float(x.grad.detach().norm()) for layer in (model.denoiser.layer1,model.denoiser.layer2,model.denoiser.layer3) for x in layer.parameters() if x.grad is not None));opt.step()
  if step%10000==0:
   row=audit(step,ld,grads);logs.append(row);ck=a.output/'checkpoints'/f'light_film_step_{step:06d}.pt';ck.parent.mkdir(exist_ok=True);torch.save({'format_version':'oracle-light-film-stage-dp-mlp-v1','step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'diffusion_config':cfg.state_dict(),'normalization':{'observation_mean':prepared.observation_normalizer.mean,'observation_std':prepared.observation_normalizer.std,'action_mean':prepared.action_normalizer.mean,'action_std':prepared.action_normalizer.std},'dataset_root':str(DATA.resolve()),'stage_source':'GT one-hot','film_zero_init':True},ck);row['checkpoint']=str(ck);print(json.dumps(row),flush=True)
 (a.output/'config.json').write_text(json.dumps({'architecture':'Light-FiLM MLP','physical_dim':43,'stage_dim':5,'stage_embedding_dim':16,'hidden_dim':128,'action_dim':7,'diffusion_steps':50,'batch_size':a.batch_size,'learning_rate':1e-3,'optimizer':'Adam','scheduler':'V1 unchanged','loss':'L_diff only','steps':a.steps,'dataset':str(DATA.resolve()),'stage_source':'GT one-hot','film_layer_count':1,'film_target_layer':'middle_hidden','film_zero_init':True},indent=2)+'\n');(a.output/'checkpoint_summary.json').write_text(json.dumps(logs,indent=2)+'\n');(a.output/'training_log.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n')
if __name__=='__main__':main()
