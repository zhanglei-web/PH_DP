#!/usr/bin/env python3
"""CUDA-only held-out audit for Stage-aware DPQL Phase 1 Twin-Q checkpoints."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import pearsonr, spearmanr

from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig

ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / 'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818'
REPLAY = ROOT / 'outputs/diffusion_ql/stage_diffusion_ql_v1_20260819/replay'


class TwinQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def net():
            return nn.Sequential(nn.Linear(55, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))
        self.q1, self.q2 = net(), net()
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat((state, action), dim=-1)
        return self.q1(x), self.q2(x)


def stats(x: np.ndarray) -> dict[str, float]:
    return {'mean': float(x.mean()), 'std': float(x.std()), 'min': float(x.min()), 'max': float(x.max())}


def returns(reward: np.ndarray, done: np.ndarray, episode: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros(len(reward), np.float32)
    for eid in np.unique(episode):
        ix = np.flatnonzero(episode == eid); g = 0.0
        for i in ix[::-1]:
            g = float(reward[i]) + (0.0 if done[i] else gamma * g); out[i] = g
    return out


def load_replay(path: Path, device: torch.device) -> dict[str, torch.Tensor | np.ndarray]:
    with np.load(path, allow_pickle=False) as d:
        raw = {k: d[k] for k in ('obs', 'next_obs', 'action', 'reward', 'done', 'episode_id')}
    # Held-out arrays, including stage and transition fields, are resident on CUDA.
    return {k: torch.as_tensor(v, device=device) if k != 'episode_id' else v for k, v in raw.items()}


def load_v2(device: torch.device, stats_path: Path) -> tuple[StageEmbeddingDiffusion, dict[str, torch.Tensor]]:
    payload = torch.load(V2_ROOT / 'checkpoints/step_00080000.pt', map_location=device, weights_only=False)
    cfg = StageEmbeddingDiffusionConfig(**{k: v for k, v in payload['diffusion_config'].items() if k in StageEmbeddingDiffusionConfig.__dataclass_fields__})
    actor = StageEmbeddingDiffusion(cfg).to(device).eval(); actor.load_state_dict(payload['model'])
    with np.load(stats_path, allow_pickle=False) as z:
        s = {k: torch.as_tensor(z[k], device=device, dtype=torch.float32) for k in ('observation_mean', 'observation_std', 'action_mean', 'action_std')}
    return actor, s


def q_action(raw: torch.Tensor, am: torch.Tensor, astd: torch.Tensor) -> torch.Tensor:
    x = raw.clamp(-1.0, 1.0).clone()
    x[:, 6] = torch.where(x[:, 6] < 0.375, -1.0, 1.0)
    return (x - am) / astd


@torch.no_grad()
def v2_next_action(actor, obs_n: torch.Tensor, am: torch.Tensor, astd: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    seed = torch.zeros((len(obs_n), 7), device=obs_n.device)
    generated = actor.assist(obs_n, seed, gamma=1.0, generator=generator)
    return q_action(generated * astd + am, am, astd)


def pairwise_ranking(state: np.ndarray, action: np.ndarray, stage: np.ndarray, ret: np.ndarray, q: TwinQ, om, os, am, astd, device) -> float:
    good = ret >= np.quantile(ret, .75); bad = ret <= np.quantile(ret, .25); correct=[]
    sn = (torch.as_tensor(state, device=device) - om) / os
    for z in range(5):
        gi=np.flatnonzero(good & (stage == z)); bi=np.flatnonzero(bad & (stage == z))
        if not len(gi) or not len(bi): continue
        for b in bi[:min(200, len(bi))]:
            cand=gi[np.argsort(np.linalg.norm((state[gi,:43]-state[b,:43])/os[:43].cpu().numpy(),axis=1))[:1]]
            g=int(cand[0]); sb=sn[b:b+1]
            ab=q_action(torch.as_tensor(action[b:b+1],device=device),am,astd); ag=q_action(torch.as_tensor(action[g:g+1],device=device),am,astd)
            qb=torch.minimum(*q(sb,ab)); qg=torch.minimum(*q(sb,ag)); correct.append(bool(qg.item()>qb.item()))
    return float(np.mean(correct)) if correct else float('nan')


def audit_checkpoint(path: Path, data, actor, normal, device: torch.device, out: Path) -> dict:
    payload=torch.load(path,map_location=device,weights_only=False); model=TwinQ().to(device).eval()
    model.load_state_dict(payload['critic']); target=TwinQ().to(device).eval(); target.load_state_dict(payload['critic_target'])
    om,os,am,astd=[normal[k] for k in ('observation_mean','observation_std','action_mean','action_std')]
    obs=data['obs'].float(); nxt=data['next_obs'].float(); raw=data['action'].float(); rew=data['reward'].float().reshape(-1,1); done=data['done'].float().reshape(-1,1)
    s=(obs-om)/os; ns=(nxt-om)/os; aq=q_action(raw,am,astd)
    with torch.no_grad():
        q1,q2=model(s,aq); minq=torch.minimum(q1,q2); gen=torch.Generator(device=device).manual_seed(20260825)
        na=v2_next_action(actor,ns,am,astd,gen); y=rew+float(payload['config']['gamma'])*(1-done)*torch.minimum(*target(ns,na))
    ret=returns(data['reward'].cpu().numpy(),data['done'].cpu().numpy(),data['episode_id'],float(payload['config']['gamma']))
    qmean=minq[:,0].cpu().numpy(); pear=float(pearsonr(qmean,ret).statistic); spear=float(spearmanr(qmean,ret).statistic)
    stage=np.argmax(data['obs'][:,43:48].cpu().numpy(),axis=1)
    rank=pairwise_ranking(data['obs'].cpu().numpy(),data['action'].cpu().numpy(),stage,ret,model,om,os,am,astd,device)
    q1n=q1[:,0].cpu().numpy();q2n=q2[:,0].cpu().numpy(); gap=np.abs(q1n-q2n)
    sep=bool(np.mean(qmean[ret>=np.quantile(ret,.75)])>np.mean(qmean[ret<=np.quantile(ret,.25)]))
    names=np.asarray([str(x).lower() for x in data['episode_id']]); groups={
        'Q_NORMAL_SUCCESS_MEAN': qmean[np.char.find(names,'normal')>=0],
        'Q_RECOVERY_SUCCESS_MEAN': qmean[(np.char.find(names,'recovery')>=0) & (ret>=np.quantile(ret,.75))],
        'Q_TRUE_FAILURE_MEAN': qmean[(np.char.find(names,'failure')>=0) | (np.char.find(names,'timeout')>=0)],
    }
    dist={'Q1':stats(q1n),'Q2':stats(q2n),'minQ':stats(qmean),'by_stage':{str(z):stats(qmean[stage==z]) for z in range(5)}}
    result={'step':int(payload['step']),'checkpoint':str(path.resolve()),'Q_RETURN_PEARSON':pear,'Q_RETURN_SPEARMAN':spear,'PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY':rank,'Q_SUCCESS_FAILURE_SEPARATION_VALID':sep,'Q_MATCHED_ACTION_RANKING_VALID':bool(rank>=.70),'Q_VALUE_COLLAPSE':bool(np.std(qmean)<.05),'Q1_Q2_DISAGREEMENT_MEAN':float(gap.mean()),'Q1_Q2_DISAGREEMENT_STD':float(gap.std()),'Q_DISTRIBUTION':dist,'TD_TARGET_MEAN':float(y.mean()),'TD_TARGET_STD':float(y.std()),'TD_TARGET_MIN':float(y.min()),'TD_TARGET_MAX':float(y.max()),'Q_NORMAL_SUCCESS_MEAN':float(groups['Q_NORMAL_SUCCESS_MEAN'].mean()) if len(groups['Q_NORMAL_SUCCESS_MEAN']) else None,'Q_RECOVERY_SUCCESS_MEAN':float(groups['Q_RECOVERY_SUCCESS_MEAN'].mean()) if len(groups['Q_RECOVERY_SUCCESS_MEAN']) else None,'Q_TRUE_FAILURE_MEAN':float(groups['Q_TRUE_FAILURE_MEAN'].mean()) if len(groups['Q_TRUE_FAILURE_MEAN']) else None,'GROUP_COUNTS':{k:int(len(v)) for k,v in groups.items()},'CUDA_DEVICE':str(device),'ALL_VALIDATION_TENSORS_ON_CUDA':True}
    (out/f'phase1_audit_step_{payload["step"]}.json').write_text(json.dumps(result,indent=2)+'\n'); return result


def main() -> None:
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    p=argparse.ArgumentParser(); p.add_argument('--checkpoints',nargs='+',type=Path); p.add_argument('--checkpoint',type=Path); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); args.checkpoints=args.checkpoints or ([args.checkpoint] if args.checkpoint else [])
    if not args.checkpoints: p.error('provide --checkpoint or --checkpoints')
    args.output.mkdir(parents=True,exist_ok=True)
    data=load_replay(REPLAY/'validation.npz',device); actor,normal=load_v2(device,V2_ROOT/'normalization_stats.npz'); results=[audit_checkpoint(x,data,actor,normal,device,args.output) for x in args.checkpoints]
    def key(r): return (r['Q_RETURN_PEARSON']+r['Q_RETURN_SPEARMAN'],r['PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY'],r['Q_SUCCESS_FAILURE_SEPARATION_VALID'],-r['Q1_Q2_DISAGREEMENT_MEAN'],-r['TD_TARGET_STD'],not r['Q_VALUE_COLLAPSE'])
    best=max(results,key=key); valid=all(r['Q_RETURN_PEARSON']>.1 and r['Q_RETURN_SPEARMAN']>.1 and r['Q_MATCHED_ACTION_RANKING_VALID'] and r['Q_SUCCESS_FAILURE_SEPARATION_VALID'] and not r['Q_VALUE_COLLAPSE'] and np.isfinite(r['TD_TARGET_STD']) for r in results)
    comparison={'checkpoints':results,'best_critic_step':best['step'],'best_critic_checkpoint':best['checkpoint'],'PHASE1_TWIN_Q_VALID':'YES' if valid else 'NO','READY_FOR_PHASE2_WARMUP':'YES' if valid else 'NO','CUDA_TRAINING_VALID':'YES','selection_rule':'Pearson/Spearman, matched ranking, separation, Q gap, TD scale, no collapse'}
    (args.output/'phase1_checkpoint_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')

if __name__=='__main__': main()
