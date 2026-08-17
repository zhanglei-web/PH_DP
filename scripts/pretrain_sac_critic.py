#!/usr/bin/env python3
"""Critic-only MC-return pretraining and held-out local-ranking evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch

from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.critic_pretraining import (
    CATEGORIES, PHASES, CriticArrays, CriticPretrainConfig, build_arrays,
    evaluate, predict, train_critic,
)


MANIFEST=Path("manifests/rule_expert_v1_formal.json")
REWARD_RUN=Path("outputs/reward_validation/sac_reward_v1_regression_20260812T150245Z")
ACTOR_ARTIFACT=Path("outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt")
OLD_AUDIT=Path("outputs/sac_diagnostics/sac_local_policy_audit_20260813T180000Z")


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: np.ndarray) -> dict[str,float]:
    x=np.asarray(values,np.float64)
    return {"count":len(x),"mean":float(x.mean()),"median":float(np.median(x)),
            "std":float(x.std()),"min":float(x.min()),"max":float(x.max()),
            "p05":float(np.percentile(x,5)),"p95":float(np.percentile(x,95))}


def return_distribution(arrays: dict[str,CriticArrays]) -> dict[str,Any]:
    out={}
    for split,data in arrays.items():
        out[split]={"transitions":len(data),"phase_counts":dict(Counter(data.phase)),"category":{}}
        for category in CATEGORIES:
            mask=data.category==category
            if mask.any():
                episode_returns=[]
                for eid in np.unique(data.episode_id[mask]):
                    emask=data.episode_id==eid; episode_returns.append(float(data.mc_return[emask][0, 0]))
                out[split]["category"][category]={"transitions":int(mask.sum()),
                    "episodes":len(episode_returns),"episode_g0":stats(np.asarray(episode_returns)),
                    "transition_g_t":stats(data.mc_return[mask])}
        out[split]["phase_g_t"]={p:stats(data.mc_return[data.phase==p]) for p in PHASES}
    return out


@torch.no_grad()
def td_diagnostic(critic,arrays,mean,std,actor,gamma=.995):
    values=[]
    for start in range(0,len(arrays),4096):
        stop=start+4096;obs=(torch.from_numpy(arrays.observation[start:stop])-mean)/std
        nxt=(torch.from_numpy(arrays.next_observation[start:stop])-mean)/std
        act=torch.from_numpy(arrays.action[start:stop]);r=torch.from_numpy(arrays.reward[start:stop])
        term=torch.from_numpy(arrays.terminated[start:stop]).float()
        q1,q2=critic(obs,act);na=actor.deterministic_action(nxt);n1,n2=critic(nxt,na)
        target=r+gamma*(1-term)*torch.minimum(n1,n2);values.append(((q1+q2)/2-target).numpy())
    residual=np.concatenate(values);return {"mean":float(residual.mean()),"std":float(residual.std()),
        "mae":float(abs(residual).mean()),"rmse":float(np.sqrt((residual**2).mean()))}


def ranking_for_critic(critic,rows,mean,std):
    q_rows=[]
    for start in range(0,len(rows),2048):
        chunk=rows[start:start+2048]
        state=torch.tensor([r["policy_state"] for r in chunk],dtype=torch.float32)
        state=(state-mean)/std
        a0=torch.tensor([r["reference_action"] for r in chunk]);a1=torch.tensor([r["candidate_action"] for r in chunk])
        with torch.no_grad():
            q10,q20=critic(state,a0);q11,q21=critic(state,a1)
        dq=(torch.minimum(q11,q21)-torch.minimum(q10,q20)).squeeze(-1).numpy()
        for row,value in zip(chunk,dq,strict=True): q_rows.append({**row,"new_delta_q":float(value)})
    summary={}
    dimensions=[("overall",q_rows)]
    dimensions += [(source,[r for r in q_rows if r["source"]==source]) for source in sorted({r["source"] for r in q_rows})]
    phase_names=("PRE_GRASP","GRASP","TRANSPORT","PLACE_AND_RETREAT")
    dimensions += [(phase,[r for r in q_rows if r["phase"]==phase]) for phase in phase_names]
    for name,selected in dimensions:
        entry={"samples":len(selected)}
        for h in (1,5,10,20):
            positive=[r for r in selected if r["new_delta_q"]>0]
            entry[f"false_improvement_h{h}"]=float(np.mean([r[f"delta_g_h{h}"]<0 for r in positive])) if positive else None
            entry[f"sign_agreement_h{h}"]=float(np.mean([np.sign(r["new_delta_q"])==np.sign(r[f"delta_g_h{h}"]) for r in selected]))
            correlations=[];groups={}
            for row in selected:groups.setdefault((row["source"],row["phase"],row["state_index"]),[]).append(row)
            for group in groups.values():
                x=[r["new_delta_q"] for r in group];y=[r[f"delta_g_h{h}"] for r in group]
                if np.std(x)>0 and np.std(y)>0:
                    rho=spearmanr(x,y).statistic
                    if np.isfinite(rho):correlations.append(rho)
            entry[f"mean_per_state_spearman_h{h}"]=float(np.mean(correlations)) if correlations else None
            truly_better=[r for r in selected if r[f"delta_g_h{h}"]<0] # reference a* better than candidate
            entry[f"expert_preference_accuracy_h{h}"]=float(np.mean([r["new_delta_q"]<0 for r in truly_better])) if truly_better else None
        summary[name]=entry
    return summary,q_rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));parser.add_argument("--max-epochs",type=int,default=100)
    args=parser.parse_args();run=Path("outputs/sac_critic")/f"sac_critic_pretrain_v1_{args.run_id}";run.mkdir(parents=True)
    config=CriticPretrainConfig(max_epochs=args.max_epochs)
    actor_payload=torch.load(ACTOR_ARTIFACT,map_location="cpu",weights_only=False);actor=SACConstrainedGaussianActor();actor.load_state_dict(actor_payload["actor_state_dict"]);actor.eval();actor.requires_grad_(False)
    actor_before=hashlib.sha256(b"".join(v.numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    mean=torch.as_tensor(actor_payload["observation_mean"],dtype=torch.float32);std=torch.as_tensor(actor_payload["observation_std"],dtype=torch.float32)
    arrays,audit=build_arrays(MANIFEST,REWARD_RUN);distribution=return_distribution(arrays)
    (run/"dataset_split.json").write_text(json.dumps(audit,indent=2)+"\n");(run/"return_distribution.json").write_text(json.dumps(distribution,indent=2)+"\n")
    (run/"config.json").write_text(json.dumps({"objective":"twin MC-return regression","config":asdict(config),"manifest":str(MANIFEST.resolve()),"reward_run":str(REWARD_RUN.resolve()),"actor_reference":str(ACTOR_ARTIFACT.resolve()),"actor_sha256":sha(ACTOR_ARTIFACT),"old_audit":str(OLD_AUDIT.resolve())},indent=2)+"\n")
    variants={"success_only":arrays["train"].subset(arrays["train"].category=="nominal_success"),"mixed":arrays["train"]}
    validation={"success_only":arrays["validation"].subset(arrays["validation"].category=="nominal_success"),"mixed":arrays["validation"]}
    critics={};train_info={};metrics={}
    for offset,name in enumerate(("success_only","mixed")):
        variant_config=CriticPretrainConfig(**{**asdict(config),"seed":config.seed+offset})
        critic,info,_=train_critic(variants[name],validation[name],mean,std,variant_config,run/f"training_history_{name}.jsonl")
        critics[name]=critic;train_info[name]=info
        metrics[name]={"validation":evaluate(critic,arrays["validation"],mean,std),"test":evaluate(critic,arrays["test"],mean,std),"td_diagnostic_test":td_diagnostic(critic,arrays["test"],mean,std,actor)}
    rows=json.loads((OLD_AUDIT/"counterfactual_results.json").read_text())
    rankings={};ranking_rows={}
    for name,critic in critics.items():rankings[name],ranking_rows[name]=ranking_for_critic(critic,rows,mean,std)
    old=json.loads((OLD_AUDIT/"phase_q_reliability.json").read_text())
    comparison={"old_critic":old,"success_only":rankings["success_only"],"mixed":rankings["mixed"]}
    (run/"critic_value_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (run/"counterfactual_ranking.json").write_text(json.dumps(comparison,indent=2)+"\n")
    phase_keys=("PRE_GRASP","GRASP","TRANSPORT","PLACE_AND_RETREAT")
    old_phase={"PRE_GRASP":"P1","GRASP":"P2","TRANSPORT":"P3","PLACE_AND_RETREAT":"P4"}
    phase_comparison={}
    for name,values in comparison.items():
        phase_comparison[name]={phase:values.get(phase,values.get(old_phase[phase])) for phase in phase_keys}
    (run/"phase_ranking.json").write_text(json.dumps(phase_comparison,indent=2)+"\n")
    public_train={name:{key:value for key,value in info.items()
                        if key!="optimizer_state_dict"} for name,info in train_info.items()}
    (run/"ablation_success_only_vs_mixed.json").write_text(json.dumps({"training":public_train,"test":{k:metrics[k]["test"] for k in metrics},"ranking":{k:rankings[k] for k in rankings}},indent=2)+"\n")
    mixed=critics["mixed"];target=type(mixed)();target.load_state_dict(mixed.state_dict());target.requires_grad_(False)
    torch.save({"format_version":"sac_critic_pretrain_v1_mc","critic_state_dict":mixed.state_dict(),"target_critic_state_dict":target.state_dict(),"optimizer_state_dict":train_info["mixed"]["optimizer_state_dict"],"epoch":train_info["mixed"]["best_epoch"],"validation_metrics":metrics["mixed"]["validation"],"config":asdict(config),"dataset_split":audit,"observation_mean":mean,"observation_std":std,"actor_reference":str(ACTOR_ARTIFACT.resolve()),"actor_sha256":sha(ACTOR_ARTIFACT)},run/"critic_pretrained_best.pt")
    actor_after=hashlib.sha256(b"".join(v.numpy().tobytes() for v in actor.state_dict().values())).hexdigest()
    if actor_before!=actor_after:raise RuntimeError("frozen Actor changed")
    headline={"run":str(run.resolve()),"train_transitions":{k:len(v) for k,v in variants.items()},"test_metrics":{k:metrics[k]["test"]["overall"] for k in metrics},"ranking_h20":{k:v["overall"]["mean_per_state_spearman_h20"] for k,v in rankings.items()},"false_improvement_h20":{k:v["overall"]["false_improvement_h20"] for k,v in rankings.items()},"actor_frozen":True,"online_training":False}
    (run/"summary.json").write_text(json.dumps(headline,indent=2)+"\n");print(json.dumps(headline,indent=2))

if __name__=="__main__":main()
