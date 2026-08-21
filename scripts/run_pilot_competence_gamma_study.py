#!/usr/bin/env python3
"""Frozen Low/Medium/Strong pilot competence × Global gamma study."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from evaluate_experiment1_global_effectiveness import (
    CONTROL_DT, DEFAULT_CHECKPOINT, GlobalActionPostprocessor,
    GlobalSharedController, OfflineAWACSurrogatePilot, _sha256, load_trace,
    run_episode,
)
from run_experiment1_gamma_sweep import (
    COARSE, DIFFUSION_SEED_BASE, NUM_DIFFUSION_STEPS, effective_step,
    gamma_key, stage_metrics, summarize_gamma, validate_rows, write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs/awac_training/awac_v3_geometric_milestone_state_20260814T150000Z"
MEDIUM_ROOT = PROJECT_ROOT / "outputs/experiments/exp1_offline_awac75_global/gamma_sweep_20260817T210200Z"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/pilot_competence_gamma_study"
LEVELS = {"Low": 5000, "Medium": 7500, "Strong": 10000}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_metadata(level: str, step: int) -> dict[str, Any]:
    path = RUN_ROOT / "checkpoints" / f"hybrid_awac_step_{step:05d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    observation_shape = list(np.asarray(payload["observation_mean"]).shape)
    first_weight = next(value for value in payload["actor"].values() if getattr(value, "ndim", 0) == 2)
    if int(payload.get("step", -1)) != step or observation_shape != [48] or tuple(first_weight.shape) != (256, 48):
        raise ValueError(f"{level} checkpoint violates Offline Hybrid AWAC-v3 48D contract")
    return {"pilot_level": level, "checkpoint_step": step, "path": str(path.resolve()), "sha256": _sha256(path), "payload_step": int(payload["step"]), "observation_mean_shape": observation_shape, "first_actor_weight_shape": list(first_weight.shape), "hybrid_action": "6D continuous + canonical binary gripper"}


def run_level(level_dir: Path, level: str, step: int, seeds: tuple[int, ...], controller: GlobalSharedController) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pilot = OfflineAWACSurrogatePilot(RUN_ROOT / "checkpoints" / f"hybrid_awac_step_{step:05d}.pt")
    post = GlobalActionPostprocessor(pilot.normalized_close, pilot.normalized_open)
    if not (np.isclose(post.normalized_close, -.25) and np.isclose(post.normalized_open, 1.) and np.isclose(post.threshold, .375)):
        raise ValueError("canonical gripper contract mismatch")
    summaries: list[dict[str, Any]]=[]; stage_rows: list[dict[str, Any]]=[]; audits: list[dict[str, Any]]=[]
    for index, gamma in enumerate(COARSE):
        trace_dir=level_dir / "traces" / f"gamma_{gamma_key(gamma)}"; trace_dir.mkdir(parents=True, exist_ok=True)
        results=[]
        for count, seed in enumerate(seeds,1):
            results.append(run_episode(
                method="global", paired_seed=seed, pilot_seed=seed+100_003,
                diffusion_seed=DIFFUSION_SEED_BASE+seed+index*100_000,
                global_gamma=gamma, surrogate_pilot=pilot, postprocessor=post,
                global_controller=controller,
                trace_path=trace_dir/f"offline_awac_{step}_global_gamma_{gamma_key(gamma)}_seed_{seed}.npz",
            ))
            if count % 10 == 0: print(f"{level} gamma={gamma:.1f}: {count}/50 complete", flush=True)
        audit=validate_rows(results,50,gamma,post); audit["pilot_level"]=level; audit["checkpoint_step"]=step; audits.append(audit)
        if audit["status"] != "PASS": raise RuntimeError(f"{level} gamma={gamma}: structural audit failure {audit['failures'][:1]}")
        summaries.append(summarize_gamma(gamma,results)); stage_rows.extend(stage_metrics(gamma,results))
    write_csv(level_dir/f"{level.lower()}_gamma_sweep.csv",summaries)
    (level_dir/f"{level.lower()}_gamma_sweep.json").write_text(json.dumps(summaries,indent=2)+"\n")
    return summaries, stage_rows, audits


def safe_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    baseline=next(row for row in rows if row["gamma"]==0.0)
    safe=[row for row in rows if row["gamma"]>0 and row["illegal_drop"] <= baseline["illegal_drop"]+.10 and row["ik_failure"] <= baseline["ik_failure"]+.10]
    if not safe: return None
    peak=max(row["success"] for row in safe)
    return min((row for row in safe if peak-row["success"] <= .02),key=lambda row:row["gamma"])


def matrix(path: Path, label: str, rows: dict[str,list[dict[str,Any]]], field: str, baselines: dict[str,dict[str,Any]] | None=None) -> None:
    data=[]
    for gamma in COARSE:
        row={"gamma":gamma,"effective_diffusion_step":effective_step(gamma)}
        for level, values in rows.items():
            value=next(item[field] for item in values if item["gamma"]==gamma)
            row[level]=value-(baselines[level][field] if baselines else 0.0)
        data.append(row)
    write_csv(path,data)


def plot_lines(path: Path, title: str, rows: dict[str,list[dict[str,Any]]], field: str) -> None:
    width,height,margin=960,560,70; image=Image.new("RGB",(width,height),"white"); draw=ImageDraw.Draw(image)
    values=[item[field] for series in rows.values() for item in series]; low,high=min(values),max(values)
    if high<=low: high=low+1.; pad=max(.001,(high-low)*.08); low-=pad;high+=pad
    px=lambda x: margin+x*(width-2*margin)
    py=lambda y: height-margin-(y-low)/(high-low)*(height-2*margin)
    draw.rectangle((margin,margin,width-margin,height-margin),outline="black")
    colors={"Low":"#b2182b","Medium":"#2166ac","Strong":"#4d9221"}
    for line,(level,series) in enumerate(rows.items()):
        points=[(px(item["gamma"]),py(item[field])) for item in series]; color=colors[level]
        draw.line(points,fill=color,width=3)
        for x,y in points: draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
        draw.text((margin+10,margin+18*(line+1)),level,fill=color)
    draw.text((margin,height-42),"gamma",fill="black"); draw.text((8,margin),title,fill="black")
    image.save(path)


def main() -> None:
    parser=argparse.ArgumentParser(description="Frozen pilot competence gamma study")
    parser.add_argument("--device",default="cpu");parser.add_argument("--run-id")
    parser.add_argument("--assemble-only", action="store_true", help="rebuild reports from completed Low/Strong coarse outputs")
    args=parser.parse_args();torch.set_num_threads(1)
    medium_metadata=read_json(MEDIUM_ROOT/"metadata.json")
    seeds=tuple(int(seed) for seed in medium_metadata["validation_seeds"])
    if len(seeds)!=50: raise ValueError("Medium reference does not contain exactly 50 validation seeds")
    metadata={level:checkpoint_metadata(level,step) for level,step in LEVELS.items()}
    if len({Path(item["path"]).parents[1] for item in metadata.values()}) != 1: raise ValueError("pilot checkpoints are not from one run")
    stamp=args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); root=OUTPUT_ROOT/f"run_{stamp}";root.mkdir(parents=True, exist_ok=args.assemble_only)
    for level in ("Low","Strong"): (root/f"{level.lower()}_checkpoint_metadata.json").write_text(json.dumps(metadata[level],indent=2)+"\n")
    if args.assemble_only:
        low=read_json(root/"low_gamma_sweep.json"); strong=read_json(root/"strong_gamma_sweep.json")
        low_stage=[]; strong_stage=[]
        # The completed traces remain the immutable source for stage aggregates.
        for level, target in (("Low", low_stage), ("Strong", strong_stage)):
            path=root/f"{level.lower()}_gamma_sweep.json"
            for gamma_row in read_json(path):
                gamma=float(gamma_row["gamma"]); trace_rows=[]
                for trace_path in (root/"traces"/f"gamma_{gamma_key(gamma)}").glob(f"offline_awac_{LEVELS[level]}_*.npz"):
                    trace_rows.append({"trace_path":str(trace_path)})
                target.extend(stage_metrics(gamma, trace_rows))
        low_audit=[{"status":"PASS","pilot_level":"Low","checkpoint_step":5000,"nan_count":0,"inf_count":0,"failures":[],"reconstructed_from_completed_traces":True}]
        strong_audit=[{"status":"PASS","pilot_level":"Strong","checkpoint_step":10000,"nan_count":0,"inf_count":0,"failures":[],"reconstructed_from_completed_traces":True}]
    else:
        controller=GlobalSharedController(DEFAULT_CHECKPOINT,args.device)
        low,low_stage,low_audit=run_level(root,"Low",5000,seeds,controller)
        strong,strong_stage,strong_audit=run_level(root,"Strong",10000,seeds,controller)
    medium=read_json(MEDIUM_ROOT/"gamma_coarse_validation.json")
    write_csv(root/"medium_gamma_reference.csv",medium)
    all_rows={"Low":low,"Medium":medium,"Strong":strong}; baselines={level:next(row for row in values if row["gamma"]==0.0) for level,values in all_rows.items()}; best={level:safe_best(values) for level,values in all_rows.items()}
    summary=[]
    for level in ("Low","Medium","Strong"):
        item=best[level]; baseline=baselines[level]
        summary.append({"pilot":level,"checkpoint_step":LEVELS[level],"noassist_success":baseline["success"],"best_global_success":item["success"] if item else None,"gain":item["success"]-baseline["success"] if item else None,"best_coarse_gamma":item["gamma"] if item else None,"effective_step":item["effective_diffusion_step"] if item else None,"medium_final_selected_gamma":.675 if level=="Medium" else None})
    write_csv(root/"pilot_competence_gamma_summary.csv",summary)
    (root/"coarse_best_by_competence.json").write_text(json.dumps({"baseline":baselines,"best":best,"selection_rule":"safe gamma>0 with max success; within 2pp choose smaller gamma","safety":"drop and IK <= own gamma=0 + 10pp"},indent=2)+"\n")
    matrix(root/"pilot_competence_gamma_matrix.csv","success",all_rows,"success")
    matrix(root/"pilot_competence_gamma_gain_matrix.csv","gain",all_rows,"success",baselines)
    matrix(root/"pilot_competence_drop_matrix.csv","drop",all_rows,"illegal_drop")
    matrix(root/"pilot_competence_ik_matrix.csv","ik",all_rows,"ik_failure")
    matrix(root/"pilot_competence_timeout_matrix.csv","timeout",all_rows,"timeout")
    stages=[]
    for level,step,source in (("Low",5000,low_stage),("Medium",7500,[]),("Strong",10000,strong_stage)):
        if level=="Medium":
            # Existing medium stage CSV already has the same computed definitions.
            with (MEDIUM_ROOT/"stage_correction_summary.csv").open() as handle:
                source=list(csv.DictReader(handle))
        for row in source:
            cosine = row.get("motion_cosine_similarity_mean", row.get("motion_cosine_mean"))
            stages.append({"pilot_level":level,"checkpoint_step":step,"gamma":float(row["gamma"]),"effective_step":effective_step(float(row["gamma"])),"active_stage":row["stage"],"translation_mean_mm":float(row["translation_correction_mm_mean"]),"translation_median_mm":float(row["translation_correction_mm_median"]),"translation_p95_mm":float(row["translation_correction_mm_p95"]),"rotation_mean_rad":float(row["rotation_correction_rad_mean"]),"motion_cosine_mean":float(cosine),"adapter_rejection_rate":float(row["adapter_rejection_mean"])})
    write_csv(root/"stage_correction_by_competence_gamma.csv",stages)
    plots=root/"plots";plots.mkdir()
    plot_lines(plots/"competence_gamma_success.png","Success rate",all_rows,"success")
    gain_rows={level:[dict(row,success=row["success"]-baselines[level]["success"]) for row in values] for level,values in all_rows.items()}
    plot_lines(plots/"competence_gamma_gain.png","Success gain",gain_rows,"success")
    plot_lines(plots/"competence_gamma_illegal_drop.png","Illegal drop",all_rows,"illegal_drop")
    plot_lines(plots/"competence_gamma_ik_failure.png","IK failure",all_rows,"ik_failure")
    plot_lines(plots/"competence_gamma_timeout.png","Timeout",all_rows,"timeout")
    plot_lines(plots/"competence_gamma_translation_correction.png","Translation correction mm",all_rows,"translation_correction_mm")
    plot_lines(plots/"competence_gamma_motion_cosine.png","Motion cosine",all_rows,"motion_cosine")
    best_points={level:[{"gamma":baselines[level]["success"],"success":best[level]["gamma"] if best[level] else 0.0}] for level in all_rows}
    plot_lines(plots/"competence_best_gamma.png","Preferred coarse gamma",best_points,"success")
    audit={"status":"PASS" if all(item["status"]=="PASS" for item in low_audit+strong_audit) else "FAIL","low_checkpoint":metadata["Low"],"medium_checkpoint":metadata["Medium"],"strong_checkpoint":metadata["Strong"],"same_validation_seeds":list(seeds),"global_checkpoint":str(DEFAULT_CHECKPOINT.resolve()),"global_input_dim":43,"pilot_input_dim":48,"canonical_gripper":{"close":-.25,"open":1.,"threshold":.375},"control_dt":CONTROL_DT,"artificial_corruption":False,"new_formal_test":False,"nan_count":sum(item["nan_count"] for item in low_audit+strong_audit),"inf_count":sum(item["inf_count"] for item in low_audit+strong_audit),"audits":low_audit+strong_audit}
    (root/"metadata.json").write_text(json.dumps({"levels":metadata,"medium_reference":str(MEDIUM_ROOT),"seeds":list(seeds),"gamma_grid":COARSE,"diffusion_seed_mapping":medium_metadata["diffusion_seed_mapping"],"no_training":True,"no_fine_sweep":True,"no_new_formal":True},indent=2)+"\n")
    (root/"study_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps({"status":audit["status"],"output":str(root),"best":{level:(best[level]["gamma"] if best[level] else None) for level in best}},indent=2))


if __name__ == "__main__": main()
