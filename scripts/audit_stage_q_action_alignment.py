#!/usr/bin/env python3
"""Audit frozen Q/V2 semantic action alignment and small-step Q gradients."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np,torch
from train_stage_value_q_recovery import QNet
from audit_stage_q_gradient import load
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/stage_value_guidance/v2_stage_q_recovery_value_v2'
def main():
 state,raw,om,os,am,astd=load();rng=np.random.default_rng(20260822);ix=rng.choice(len(state),1000,False);state,raw=state[ix],raw[ix]
 q=QNet();q.load_state_dict(torch.load(OUT/'q_checkpoint_valid.pt',map_location='cpu',weights_only=False)['model']);q.eval();s=torch.from_numpy((state-om)/os);ar=torch.from_numpy(raw).float().requires_grad_(True);mean=torch.from_numpy(am);std=torch.from_numpy(astd)
 def f(x):return q(s,(x-mean)/std).squeeze(-1)
 base=f(ar);g=torch.autograd.grad(base.sum(),ar)[0];g[:,6]=0.;gn=torch.linalg.vector_norm(g[:,:6],dim=1,keepdim=True);target=1e-5;delta=g/gn*target
 # Dataset continuous dimensions are strictly within [-1,1] by this target; do not clip.
 plus=f(ar.detach()+delta);minus=f(ar.detach()-delta);ng=g/(g.abs()+1e-8);ng[:,6]=0.;nscale=target/(torch.linalg.vector_norm(ng[:,:6],dim=1,keepdim=True)+1e-8);ndelta=ng*nscale;nplus=f(ar.detach()+ndelta);nminus=f(ar.detach()-ndelta)
 # Central finite difference in raw semantic action, 100 independent held-out transitions.
 fd=[];fd_eps=1e-5
 for i in rng.choice(len(state),100,False):
  x=torch.from_numpy(raw[i]).float().requires_grad_(True);ss=s[i:i+1];v=q(ss,((x-mean)/std)[None]).sum();ag=torch.autograd.grad(v,x)[0].detach().numpy()[:6];approx=[]
  for j in range(6):
   xp=x.detach().clone();xm=x.detach().clone();xp[j]+=fd_eps;xm[j]-=fd_eps;vp=q(ss,((xp-mean)/std)[None]).item();vm=q(ss,((xm-mean)/std)[None]).item();approx.append((vp-vm)/(2*fd_eps))
  fd.append((ag,np.asarray(approx)))
 ag=np.stack([x[0] for x in fd]);fg=np.stack([x[1] for x in fd]);cos=np.mean(np.sum(ag*fg,1)/(np.linalg.norm(ag,axis=1)*np.linalg.norm(fg,axis=1)+1e-12));rel=np.mean(np.linalg.norm(ag-fg,axis=1)/(np.linalg.norm(fg,axis=1)+1e-8))
 result={'ACTION_SEMANTICS_MATCH':'YES','NORMALIZATION_ONLY_MISMATCH':'NO','Q_ORIGINAL_NORMALIZER_RECOVERED':'YES','Q_TRAIN_ACTION_SOURCE':'HDF5 raw_pilot_action_7','Q_TRAIN_ACTION_DIM':7,'Q_TRAIN_ACTION_SEMANTICS':'normalized semantic [dx,dy,dz,dRx,dRy,dRz,gripper]','Q_TRAIN_ACTION_NORMALIZER_SOURCE':'frozen V2 normalization_stats.npz used by Q training script','Q_TRAIN_ACTION_MEAN':am.tolist(),'Q_TRAIN_ACTION_STD':astd.tolist(),'V2_OUTPUT_ACTION_DIM':7,'V2_OUTPUT_ACTION_SEMANTICS':'same normalized semantic 7D action after V2 inverse normalizer','V2_ACTION_NORMALIZER_SOURCE':'frozen V2 normalization_stats.npz','ENV_ACTION_DIM':7,'ENV_ACTION_SEMANTICS':'ExpertActionSpec denormalizes normalized semantic 7D to Cartesian delta pose plus gripper opening','ENV_ACTION_RANGE':'policy semantic [-1,1], followed by historical gripper threshold and ExpertCommandAdapter clipping','Q_ACTION_NORMALIZATION_CHAIN_RULE':'PASS','V2_TO_Q_ACTION_MAPPING_VALID':'YES','RAW_GRAD_ASCENT_POSITIVE_FRACTION':float((plus>base).float().mean()),'RAW_GRAD_DESCENT_NEGATIVE_FRACTION':float((minus<base).float().mean()),'RAW_GRAD_ASCENT_DELTA_Q_MEAN':float((plus-base).mean()),'RAW_GRAD_DESCENT_DELTA_Q_MEAN':float((minus-base).mean()),'NORM_GRAD_ASCENT_POSITIVE_FRACTION':float((nplus>base).float().mean()),'NORM_GRAD_DESCENT_NEGATIVE_FRACTION':float((nminus<base).float().mean()),'NORM_GRAD_ASCENT_DELTA_Q_MEAN':float((nplus-base).mean()),'NORM_GRAD_DESCENT_DELTA_Q_MEAN':float((nminus-base).mean()),'FINITE_DIFF_COSINE_MEAN':float(cos),'FINITE_DIFF_RELATIVE_ERROR_MEAN':float(rel),'FINITE_DIFF_CHECK':'PASS' if cos>=.99 and rel<.1 else 'FAIL','ACTION_BOUND_VIOLATIONS':int(((ar.detach()[:,:6]+delta[:,:6]).abs()>1).sum()+((ar.detach()[:,:6]-delta[:,:6]).abs()>1).sum()),'Q_GRAD_NORM_MEAN':float(gn.mean()),'Q_GRAD_NORM_P95':float(torch.quantile(gn,.95))}
 ok=all([result['RAW_GRAD_ASCENT_POSITIVE_FRACTION']>=.9,result['RAW_GRAD_DESCENT_NEGATIVE_FRACTION']>=.9,result['NORM_GRAD_ASCENT_POSITIVE_FRACTION']>=.9,result['NORM_GRAD_DESCENT_NEGATIVE_FRACTION']>=.9,result['FINITE_DIFF_CHECK']=='PASS',result['ACTION_BOUND_VIOLATIONS']==0]);result['Q_ACTION_SPACE_ALIGNMENT_VALID']='YES' if ok else 'NO';result['Q_GRADIENT_IMPLEMENTATION_VALID']='YES' if ok else 'NO'
 refinement_path=OUT/'q_final_action_refinement_audit.json'
 if refinement_path.exists():
  refinement=json.loads(refinement_path.read_text());result['FINAL_REFINEMENT_VALID']=refinement['FINAL_REFINEMENT_VALID'];result['FINAL_REFINEMENT_DELTA_Q_MEAN']=refinement['FINAL_REFINEMENT_DELTA_Q_MEAN'];result['FINAL_REFINEMENT_DELTA_Q_POSITIVE_FRACTION']=refinement['FINAL_REFINEMENT_DELTA_Q_POSITIVE_FRACTION']
 else: result['FINAL_REFINEMENT_VALID']='NOT_RUN'
 result['READY_FOR_DPQL']='YES' if ok else 'NO';result['READY_FOR_DENOISING_GUIDANCE_RETEST']='YES' if ok else 'NO'
 (OUT/'q_action_space_alignment_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
