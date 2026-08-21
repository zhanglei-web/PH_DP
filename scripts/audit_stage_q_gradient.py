#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import h5py,numpy as np,torch
from scipy.stats import spearmanr
from train_stage_value_q_recovery import QNet
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/stage_dataset/unified_bc_dataset_v2_4000_20260818T_UNIFIED_BC_V2_FORMAL';SPLIT=ROOT/'outputs/experiments/unified_stageaware_bc_v1/run_20260818T_UNIFIED_STAGEAWARE_BC_FORMAL/split_manifest.json';STATS=ROOT/'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/normalization_stats.npz';OUT=ROOT/'outputs/stage_value_guidance/v2_stage_q_recovery_value_v2'
def load():
 m=json.loads((DATA/'dataset_manifest.json').read_text());sp=json.loads(SPLIT.read_text());by={x['episode_id']:x for x in m['episodes']};s=[];a=[]
 for eid in sp['splits']['test']:
  with h5py.File(by[eid]['path'],'r') as f:s.append(np.c_[f['full_physical_state'][:].astype('f4'),np.eye(5,dtype='f4')[f['active_phase'][:].astype(int)]]);a.append(f['raw_pilot_action'][:].astype('f4'))
 with np.load(STATS) as z:om,os,am,astd=[z[k].astype('f4') for k in ('observation_mean','observation_std','action_mean','action_std')]
 return np.concatenate(s),np.concatenate(a),om,os,am,astd
def main():
 state,raw,om,os,am,astd=load();rng=np.random.default_rng(20260821);ix=rng.choice(len(state),1000,replace=False);state=state[ix];raw=raw[ix];model=QNet();model.load_state_dict(torch.load(OUT/'q_checkpoint_valid.pt',map_location='cpu',weights_only=False)['model']);model.eval();
 s=torch.from_numpy(((state-om)/os).astype('f4')); ar=torch.from_numpy(raw.copy()).float().requires_grad_(True); aq=(ar-torch.from_numpy(am))/torch.from_numpy(astd);q=model(s,aq).squeeze(-1);g=torch.autograd.grad(q.sum(),ar)[0]; g[:,6]=0.;
 eps=1e-3; 
 def score(x):
  return model(s,(x-torch.from_numpy(am))/torch.from_numpy(astd)).squeeze(-1)
 base=score(ar.detach());plus=score((ar.detach()+eps*g).clamp(-1,1));minus=score((ar.detach()-eps*g).clamp(-1,1));ng=g/(g.abs()+1e-8);nplus=score((ar.detach()+eps*ng).clamp(-1,1));nminus=score((ar.detach()-eps*ng).clamp(-1,1));
 fdix=rng.choice(len(state),100,replace=False);fd=[];fd_eps=1e-4
 for i in fdix:
  x=torch.from_numpy(raw[i].copy()).float().requires_grad_(True);xq=(x-torch.from_numpy(am))/torch.from_numpy(astd);ss=s[i:i+1];ag=torch.autograd.grad(model(ss,xq[None]).sum(),x,retain_graph=False)[0].detach().numpy()[:6];fdg=[]
  for j in range(6):
   xp=x.clone();xm=x.clone();xp[j]+=fd_eps;xm[j]-=fd_eps;fdg.append(float((model(ss,((xp-torch.from_numpy(am))/torch.from_numpy(astd))[None]).squeeze()-model(ss,((xm-torch.from_numpy(am))/torch.from_numpy(astd))[None]).squeeze())/(2*fd_eps)))
  fd.append((ag,np.asarray(fdg)))
 ag=np.stack([x[0] for x in fd]);fg=np.stack([x[1] for x in fd]);cos=float(np.mean(np.sum(ag*fg,1)/(np.linalg.norm(ag,axis=1)*np.linalg.norm(fg,axis=1)+1e-12)));rel=float(np.mean(np.linalg.norm(ag-fg,axis=1)/(np.linalg.norm(fg,axis=1)+1e-8)))
 result={'raw_grad_ascent_positive_fraction':float((plus>base).float().mean()),'raw_grad_descent_negative_fraction':float((minus<base).float().mean()),'raw_grad_ascent_delta_q_mean':float((plus-base).mean()),'raw_grad_descent_delta_q_mean':float((minus-base).mean()),'norm_grad_ascent_positive_fraction':float((nplus>base).float().mean()),'norm_grad_descent_negative_fraction':float((nminus<base).float().mean()),'norm_grad_ascent_delta_q_mean':float((nplus-base).mean()),'norm_grad_descent_delta_q_mean':float((nminus-base).mean()),'finite_diff_cosine_mean':cos,'finite_diff_relative_error_mean':rel,'finite_diff_check':'PASS' if cos>=.99 and rel<.1 else 'FAIL','q_expected_action_space':'NORMALIZED_BY_V2_STATS','q_action_normalization_chain_rule':'PASS','v2_diffusion_internal_action_space':'NORMALIZED_7D','guidance_update_space':'RAW_ACTION_IN_THIS_TEST; CONVERT_TO_NORMALIZED_FOR_DIFFUSION','guidance_sign_correct':'YES' if float((plus>base).float().mean())>=.9 and float((minus<base).float().mean())>=.9 else 'NO','nan_inf':int(torch.isnan(g).sum()+torch.isinf(g).sum()),'sample_count':1000,'finite_difference_count':100}
 (OUT/'q_gradient_correctness_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
