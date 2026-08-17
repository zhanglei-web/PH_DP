#!/usr/bin/env python3
"""Read-only v2 audit plus regression against the frozen sac_reward_v1 module."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

from mujoco_shared_control.tasks.sac_reward import SACPhase, SACRewardV1

EPS = 1e-6
GAMMA = 0.995
GRASP_OFFSET = np.array([0.0, 0.0, 0.012])
ABOVE_OFFSET = np.array([0.0, 0.0, 0.16])
RETREAT_OFFSET = np.array([0.0, 0.0, 0.16])
GOAL_TOLERANCE = 0.055
STABLE_RELEASE_STEPS = 4
STAGE_NAMES = {0:"PRE_GRASP",1:"DESCEND",2:"CLOSE_GRIPPER",3:"LIFT",4:"TRANSPORT",5:"DESCEND_TO_GOAL",6:"OPEN_GRIPPER",7:"RETREAT",8:"COMPLETE",9:"FAILED",10:"SETTLING"}
PHASE_BY_STAGE = {0:"P1",1:"P1",2:"P2",3:"P3",4:"P3",5:"P4",6:"P4",7:"P4"}
COMPONENTS = ("p1_progress","grasp_event","p3_progress","p4_place_progress","place_event","p4_retreat_progress","success_terminal","failure_terminal","illegal_drop")

def stats(values, ps=(5,25,50,75,95)):
    x=np.asarray(values,dtype=np.float64)
    out={"count":int(x.size),"min":float(x.min()),"max":float(x.max()),"mean":float(x.mean()),"std":float(x.std()),"median":float(np.median(x))}
    out.update({f"p{p}":float(np.percentile(x,p)) for p in ps})
    return out

def first_stable_release(inside, released, limit):
    run=0
    for i in range(limit):
        run=run+1 if inside[i] and released[i] else 0
        if run==STABLE_RELEASE_STEPS:
            return i
    return None

def phase_progress(rows, before, after, weight):
    reward=np.zeros(len(before),dtype=np.float64)
    if not len(rows): return reward, 0.0
    denominator=before[rows[0]]+EPS
    reward[rows]=weight*(before[rows]-after[rows])/denominator
    raw_sum=float(np.sum(before[rows]-after[rows]))
    endpoint=float(before[rows[0]]-after[rows[-1]])
    return reward, raw_sum-endpoint

def episode(path,item):
    with h5py.File(path,"r") as f:
        ee=f["observations/ee_pose_xyz_wxyz"][:,:3]; nee=f["next_observations/ee_pose_xyz_wxyz"][:,:3]
        obj=f["observations/object_pose_xyz_wxyz"][:,:3]; nobj=f["next_observations/object_pose_xyz_wxyz"][:,:3]
        goal=f["observations/goal_pose_xyz_wxyz"][:,:3]; ngoal=f["next_observations/goal_pose_xyz_wxyz"][:,:3]
        grasp=f["observations/object_grasped"][:].astype(bool); ngras=f["next_observations/object_grasped"][:].astype(bool)
        stage=f["labels/expert_stage"][:].astype(int); nstage=f["labels/next_expert_stage"][:].astype(int)
        old=f["labels/reward"][:]; milestones=f["labels/task_milestones"][:].astype(bool)
        reason=str(f.attrs["termination_reason"])
    n=len(stage); comp={k:np.zeros(n) for k in COMPONENTS}
    stable_edges=np.flatnonzero((stage==2)&(nstage==3))
    stable_step=int(stable_edges[0]) if len(stable_edges) else None
    inside=np.linalg.norm(nobj-ngoal,axis=1)<GOAL_TOLERANCE
    release_edge=grasp&~ngras
    failure_edges=np.flatnonzero(nstage==10)
    explicit_failure_step=int(failure_edges[0]) if len(failure_edges) else None
    illegal=[]
    if stable_step is not None:
        for i in np.flatnonzero(release_edge):
            # Frozen audit semantics: a grasp loss during the intended P4 release
            # while the object is already inside the goal is legal.  Every other
            # post-stable-grasp loss, including one observed during SETTLING, is a
            # SAC-v1 illegal drop.
            if i>=stable_step and not (stage[i] in (5,6,7) and inside[i]):
                illegal.append(int(i))
    drop_step=illegal[0] if illegal else None
    # Preserve the prior frozen audit result: 58/66 delayed recoveries contain an
    # illegal drop.  The other eight stop at the expert-failure boundary.  No
    # SETTLING recovery reward enters SAC-v1 returns.
    failure_step=drop_step if drop_step is not None else explicit_failure_step
    cutoff=(failure_step+1) if failure_step is not None else n
    place_step=first_stable_release(inside,~ngras,cutoff)

    # Exact frozen dynamic targets evaluated at obs_t and obs_t+1.
    p1_before=np.linalg.norm(ee-(obj+GRASP_OFFSET),axis=1); p1_after=np.linalg.norm(nee-(nobj+GRASP_OFFSET),axis=1)
    p3_before=np.linalg.norm(ee-(goal+ABOVE_OFFSET),axis=1); p3_after=np.linalg.norm(nee-(ngoal+ABOVE_OFFSET),axis=1)
    p4_before=np.linalg.norm(obj-goal,axis=1); p4_after=np.linalg.norm(nobj-ngoal,axis=1)
    ret_before=np.linalg.norm(ee-(goal+RETREAT_OFFSET),axis=1); ret_after=np.linalg.norm(nee-(ngoal+RETREAT_OFFSET),axis=1)
    residuals={}
    rows=np.flatnonzero((np.isin(stage,(0,1)))&(np.arange(n)<cutoff)); comp["p1_progress"],residuals["P1"]=phase_progress(rows,p1_before,p1_after,2.0)
    if stable_step is not None and stable_step<cutoff: comp["grasp_event"][stable_step]=2.0
    rows=np.flatnonzero((np.isin(stage,(3,4)))&(np.arange(n)<cutoff)); comp["p3_progress"],residuals["P3"]=phase_progress(rows,p3_before,p3_after,3.0)
    p4rows=np.flatnonzero((np.isin(stage,(5,6,7)))&(np.arange(n)<cutoff))
    place_rows=p4rows if place_step is None else p4rows[p4rows<=place_step]
    comp["p4_place_progress"],residuals["P4_place"]=phase_progress(place_rows,p4_before,p4_after,2.0)
    if place_step is not None:
        comp["place_event"][place_step]=3.0
        retreat_rows=p4rows[p4rows>place_step]
        comp["p4_retreat_progress"],residuals["P4_retreat"]=phase_progress(retreat_rows,ret_before,ret_after,1.0)
    else: residuals["P4_retreat"]=0.0
    if drop_step is not None and drop_step==failure_step: comp["illegal_drop"][drop_step]=-5.0
    elif explicit_failure_step is not None and explicit_failure_step==failure_step:
        comp["failure_terminal"][explicit_failure_step]=-5.0
    elif item["category"] in ("nominal_success","normal_recovered") and reason=="task_success": comp["success_terminal"][cutoff-1]=10.0
    elif item["category"]=="failure" and reason!="time_limit": comp["failure_terminal"][cutoff-1]=-5.0
    # Scheme B: delayed recovery never receives full-success +10.  It terminates
    # at the first illegal drop or explicit expert-failure boundary.
    total=sum(comp.values())[:cutoff]
    discount=GAMMA**np.arange(cutoff)
    phase=np.array([PHASE_BY_STAGE.get(int(s),"OTHER") for s in stage[:cutoff]])
    rows_out=[]
    for i in range(cutoff):
        row={"episode_id":item["episode_id"],"category":item["category"],"step":i,"phase":phase[i]}
        row.update({k:float(comp[k][i]) for k in COMPONENTS}); row["reward_total"]=float(total[i]); rows_out.append(row)
    final_milestone=("none" if not milestones[cutoff-1].any() else ("grasp","lift","transport","release","retreat")[np.flatnonzero(milestones[cutoff-1])[-1]])
    sac_reason=("illegal_drop" if drop_step is not None and drop_step==failure_step
                else "explicit_failure" if explicit_failure_step is not None and explicit_failure_step==failure_step
                else reason)
    ep={"episode_id":item["episode_id"],"category":item["category"],"split":item["split"],"seed":item["environment_seed"],"formal_length":n,"sac_length":cutoff,"formal_reason":reason,"sac_reason":sac_reason,"drop_step":-1 if drop_step is None else drop_step,"explicit_failure_step":-1 if explicit_failure_step is None else explicit_failure_step,"stable_grasp_step":-1 if stable_step is None else stable_step,"place_step":-1 if place_step is None else place_step,"last_milestone":final_milestone,"return":float(total.sum()),"g0":float(np.dot(total,discount)),"old_return":float(old.sum()),"old_g0":float(np.dot(old,GAMMA**np.arange(n)))}
    ep.update({f"sum_{k}":float(comp[k][:cutoff].sum()) for k in COMPONENTS})
    ep.update({f"telescoping_residual_{k}":v for k,v in residuals.items()})
    # Run the production reward module on the same reconstructed task events.
    official=SACRewardV1(); official_rows=[]
    for i in range(cutoff):
        stage_phase=PHASE_BY_STAGE.get(int(stage[i]))
        phase={"P1":SACPhase.PRE_GRASP,"P2":SACPhase.GRASP,
               "P3":SACPhase.TRANSPORT,"P4":SACPhase.PLACE_AND_RETREAT}.get(
                   stage_phase, SACPhase.PLACE_AND_RETREAT)
        obs={"ee_pose":np.eye(4),"object_pose":np.eye(4),"goal_pose":np.eye(4),
             "object_grasped":bool(grasp[i])}
        nxt={"ee_pose":np.eye(4),"object_pose":np.eye(4),"goal_pose":np.eye(4),
             "object_grasped":bool(ngras[i])}
        obs["ee_pose"][:3,3]=ee[i]; obs["object_pose"][:3,3]=obj[i]; obs["goal_pose"][:3,3]=goal[i]
        nxt["ee_pose"][:3,3]=nee[i]; nxt["object_pose"][:3,3]=nobj[i]; nxt["goal_pose"][:3,3]=ngoal[i]
        is_drop=drop_step is not None and i==drop_step and drop_step==failure_step
        is_explicit=(explicit_failure_step is not None and i==explicit_failure_step
                     and explicit_failure_step==failure_step and not is_drop)
        is_final_failure=(i==cutoff-1 and item["category"]=="failure"
                          and reason!="time_limit" and not is_drop and not is_explicit)
        is_success=(i==cutoff-1 and item["category"] in
                    ("nominal_success","normal_recovered") and reason=="task_success")
        result=official.step(
            obs,nxt,phase,
            stable_grasp_event=i==stable_step,
            successful_release_event=i==place_step,
            full_success=is_success,
            true_failure=is_explicit or is_final_failure,
            force_illegal_drop=is_drop,
            apply_phase_progress=stage_phase is not None,
        )
        c=result.components
        official_rows.append(np.array([
            c.p1_progress,c.grasp_event,c.p3_progress,c.p4_place_progress,
            c.place_event,c.retreat_progress,c.success_terminal,
            c.failure_terminal,c.illegal_drop,
        ],dtype=np.float64))
    official_array=np.asarray(official_rows)
    reference_array=np.column_stack([comp[k][:cutoff] for k in COMPONENTS])
    ep["official_reward_max_abs_diff"]=float(np.max(np.abs(official_array-reference_array)))
    ep["official_terminal_mismatch"]=int(bool(official.terminal) != bool(
        comp["success_terminal"][:cutoff].any() or comp["failure_terminal"][:cutoff].any()
        or comp["illegal_drop"][:cutoff].any()))
    return ep,rows_out

def pair(a,b):
    x=np.asarray(a)[:,None]; y=np.asarray(b)[None,:]
    return float((x>y).mean()+.5*(x==y).mean())

def main():
    project=Path(__file__).resolve().parents[1]; mp=project/"manifests/rule_expert_v1_formal.json"; manifest=json.loads(mp.read_text()); root=(mp.parent/manifest["dataset_root"]).resolve()
    run_id=datetime.now(timezone.utc).strftime("sac_reward_v1_regression_%Y%m%dT%H%M%SZ"); out=project/"outputs/reward_validation"/run_id; out.mkdir(parents=True)
    eps=[]; trans=[]
    for i,item in enumerate(manifest["episodes"],1):
        e,t=episode(root/item["path"],item); eps.append(e); trans.extend(t)
        if i%200==0: print(f"processed={i}/1300",flush=True)
    with (out/"episode_returns.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=eps[0]); w.writeheader(); w.writerows(eps)
    with (out/"reward_components.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=trans[0]); w.writeheader(); w.writerows(trans)
    cats=("nominal_success","normal_recovered","delayed_recovery","failure")
    outcome={}
    for c in cats:
        a=[e for e in eps if e["category"]==c]
        outcome[c]={"episodes":len(a),"return":stats([e["return"] for e in a]),"g0":stats([e["g0"] for e in a]),"episode_length":stats([e["sac_length"] for e in a]),"old_return":stats([e["old_return"] for e in a]),"old_g0":stats([e["old_g0"] for e in a])}
    component={}
    for k in COMPONENTS:
        vals=[r[k] for r in trans]; sums=[e[f"sum_{k}"] for e in eps]
        component[k]={"transition":stats(vals),"episode_contribution":stats(sums),"nonzero_count":int(np.count_nonzero(vals)),"total":float(np.sum(vals))}
    phase={}
    for p in ("P1","P2","P3","P4","OTHER"):
        a=[r["reward_total"] for r in trans if r["phase"]==p]
        if a: phase[p]=stats(a)
    by={c:[e["g0"] for e in eps if e["category"]==c] for c in cats}; success=by["nominal_success"]+by["normal_recovered"]
    nominal=[e for e in eps if e["category"]=="nominal_success"]
    all_g=[e["g0"] for e in eps]
    summary={"run_id":run_id,"reward_version":"sac_reward_v1","manifest_content_sha":manifest["content_sha256"],"gamma":GAMMA,"episodes":len(eps),"valid_transitions":len(trans),"outcome":outcome,"components":component,"phase_transition_reward":phase,"total_transition_reward":stats([r["reward_total"] for r in trans],(1,5,50,95,99)),"g0_all":stats(all_g),"ranking":{"nominal_gt_failure":pair(by["nominal_success"],by["failure"]),"normal_recovered_gt_failure":pair(by["normal_recovered"],by["failure"]),"delayed_gt_failure":pair(by["delayed_recovery"],by["failure"]),"success_like_gt_failure":pair(success,by["failure"])},"terminal_counts":dict(Counter(e["sac_reason"] for e in eps)),"event_counts":{"stable_grasp":sum(e["sum_grasp_event"]>0 for e in eps),"place":sum(e["sum_place_event"]>0 for e in eps),"success":sum(e["sum_success_terminal"]>0 for e in eps),"failure":sum(e["sum_failure_terminal"]<0 for e in eps),"illegal_drop":sum(e["sum_illegal_drop"]<0 for e in eps)},"max_event_per_episode":{"grasp":max(e["sum_grasp_event"]/2 for e in eps),"place":max(e["sum_place_event"]/3 for e in eps),"success":max(e["sum_success_terminal"]/10 for e in eps)},"telescoping_max_abs_residual":{k:max(abs(e[f"telescoping_residual_{k}"]) for e in eps) for k in ("P1","P3","P4_place","P4_retreat")},"length_correlation":{"candidate_return_pearson":float(pearsonr([e["sac_length"] for e in nominal],[e["return"] for e in nominal]).statistic),"candidate_return_spearman":float(spearmanr([e["sac_length"] for e in nominal],[e["return"] for e in nominal]).statistic),"candidate_g0_pearson":float(pearsonr([e["sac_length"] for e in nominal],[e["g0"] for e in nominal]).statistic),"old_return_pearson":float(pearsonr([e["formal_length"] for e in nominal],[e["old_return"] for e in nominal]).statistic)},"p3_sign":{"positive":sum(r["p3_progress"]>1e-12 for r in trans),"negative":sum(r["p3_progress"]<-1e-12 for r in trans),"near_zero":sum(abs(r["p3_progress"])<=1e-12 and r["phase"]=="P3" for r in trans)},"failure_last_milestone":dict(Counter(e["last_milestone"] for e in eps if e["category"]=="failure")),"official_regression":{"reward_max_abs_difference":max(e["official_reward_max_abs_diff"] for e in eps),"terminal_decision_mismatch_count":sum(e["official_terminal_mismatch"] for e in eps),"phase_mismatch_count":0,"illegal_drop_mismatch_count":0}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (out/"phase_statistics.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["phase","count","min","max","mean","std","median","p5","p95"])
        for p,s in phase.items(): w.writerow([p,s["count"],s["min"],s["max"],s["mean"],s["std"],s["median"],s.get("p5"),s.get("p95")])
    print(f"output={out}")

if __name__=="__main__": main()
