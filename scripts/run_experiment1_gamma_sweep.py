#!/usr/bin/env python3
"""Pre-registered Global-assistance strength validation for Experiment 1.

This orchestrates frozen E1 components only.  It does not train, alter a
checkpoint, or alter the environment/action contracts.  Gamma is varied only
at the existing GlobalSharedController.assist call site.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from evaluate_experiment1_global_effectiveness import (
    ACTION_DIM,
    CONTROL_DT,
    DEFAULT_CHECKPOINT,
    DEFAULT_SURROGATE_CHECKPOINT,
    OBSERVATION_DIM,
    OfflineAWACSurrogatePilot,
    GlobalActionPostprocessor,
    GlobalSharedController,
    _bootstrap_ci,
    _mcnemar_exact,
    _sha256,
    aggregate_stage_corrections,
    load_trace,
    run_episode,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/exp1_offline_awac75_global"
COARSE = tuple(round(i / 10.0, 1) for i in range(11))
VALIDATION_SEEDS = tuple(range(2_400_000, 2_400_050))
FINAL_SEEDS = tuple(range(2_500_000, 2_500_300))
NUM_DIFFUSION_STEPS = 50
DIFFUSION_SEED_BASE = 7_000_000


def effective_step(gamma: float) -> int:
    return int((NUM_DIFFUSION_STEPS - 1) * float(gamma))


def gamma_key(gamma: float) -> str:
    return f"{gamma:.3f}".rstrip("0").rstrip(".") or "0"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows]))


def _mean_trace_field(rows: list[dict[str, Any]], field: str) -> float:
    values = [np.asarray(load_trace(row["trace_path"])[field], np.float64) for row in rows]
    values = [value for value in values if len(value)]
    return float(np.concatenate(values).mean()) if values else 0.0


def _motion_intervention(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Return 6-D pilot-to-assisted motion difference statistics."""
    values: list[np.ndarray] = []
    for row in rows:
        trace = load_trace(row["trace_path"])
        values.append(np.linalg.norm(
            np.asarray(trace["assisted_action_7"][:, :6] - trace["raw_pilot_action_7"][:, :6], np.float64), axis=1
        ))
    value = np.concatenate(values) if values else np.empty(0)
    return (
        float(value.mean()) if len(value) else 0.0,
        float(np.median(value)) if len(value) else 0.0,
        float(np.quantile(value, .95)) if len(value) else 0.0,
    )


def summarize_gamma(gamma: float, rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = np.asarray([row["episode_steps"] for row in rows], np.float64)
    total_steps = max(1, int(steps.sum()))
    intervention_mean, intervention_median, intervention_p95 = _motion_intervention(rows)
    return {
        "gamma": float(gamma),
        "effective_diffusion_step": effective_step(gamma),
        "N": len(rows),
        "success": _rate(rows, "success"),
        "grasp": _rate(rows, "grasp"),
        "lift": _rate(rows, "lift"),
        "transport": _rate(rows, "transport"),
        "place": _rate(rows, "place"),
        "retreat": _rate(rows, "retreat"),
        "illegal_drop": _rate(rows, "illegal_drop"),
        "ik_failure": _rate(rows, "ik_failure"),
        "timeout": _rate(rows, "timeout"),
        "mean_steps": float(steps.mean()),
        "median_steps": float(np.median(steps)),
        "translation_correction_mm": 1000.0 * _mean_trace_field(rows, "translation_correction_m"),
        "rotation_correction_rad": _mean_trace_field(rows, "rotation_correction_rad"),
        "motion_cosine": _mean_trace_field(rows, "motion_cosine_similarity"),
        "pilot_assist_motion_delta_normalized": intervention_mean,
        "pilot_assist_motion_delta_normalized_median": intervention_median,
        "pilot_assist_motion_delta_normalized_p95": intervention_p95,
        "policy_clip_fraction": float(sum(row["policy_clip_steps"] for row in rows) / total_steps),
        "adapter_rejection_rate": float(sum(row["adapter_rejection_count"] for row in rows) / total_steps),
        "fallback_rate": float(sum(row["fallback_count"] for row in rows) / total_steps),
        "gripper_change_by_assist_rate": _mean_trace_field(rows, "gripper_changed_by_assist"),
    }


def stage_metrics(gamma: float, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    stages = ("APPROACH", "GRASP_LIFT", "TRANSPORT", "PLACE_RELEASE", "RETREAT")
    for index, name in enumerate(stages):
        values: dict[str, list[np.ndarray]] = {
            "translation_correction_mm": [], "rotation_correction_rad": [],
            "motion_cosine": [], "intervention_norm": [], "policy_clip": [],
            "adapter_rejection": [], "fallback": [], "gripper_changed": [],
        }
        for row in rows:
            trace = load_trace(row["trace_path"])
            mask = trace["active_stage"] == index
            if not mask.any():
                continue
            values["translation_correction_mm"].append(np.asarray(trace["translation_correction_m"][mask], np.float64) * 1000.0)
            values["rotation_correction_rad"].append(np.asarray(trace["rotation_correction_rad"][mask], np.float64))
            values["motion_cosine"].append(np.asarray(trace["motion_cosine_similarity"][mask], np.float64))
            delta = np.asarray(trace["assisted_action_7"][:, :6] - trace["raw_pilot_action_7"][:, :6], np.float64)
            values["intervention_norm"].append(np.linalg.norm(delta[mask], axis=1))
            values["policy_clip"].append(np.asarray(trace["action_clipped"][mask], np.float64))
            values["adapter_rejection"].append((~np.asarray(trace["adapter_accepted"][mask], bool)).astype(float))
            values["fallback"].append(np.asarray(trace["fallback_used"][mask], np.float64))
            values["gripper_changed"].append(np.asarray(trace["gripper_changed_by_assist"][mask], np.float64))
        row: dict[str, Any] = {"gamma": gamma, "effective_diffusion_step": effective_step(gamma), "stage": name}
        for metric, chunks in values.items():
            array = np.concatenate(chunks) if chunks else np.empty(0)
            row["frames"] = int(len(array))
            row[f"{metric}_mean"] = float(array.mean()) if len(array) else 0.0
            row[f"{metric}_median"] = float(np.median(array)) if len(array) else 0.0
            row[f"{metric}_p95"] = float(np.quantile(array, .95)) if len(array) else 0.0
        output.append(row)
    return output


def validate_rows(rows: list[dict[str, Any]], expected: int, gamma: float, postprocessor: GlobalActionPostprocessor) -> dict[str, Any]:
    failures: list[str] = []
    if len(rows) != expected:
        failures.append(f"rollout count {len(rows)} != {expected}")
    for row in rows:
        if float(row["gamma"]) != float(gamma):
            failures.append(f"gamma mismatch {row['episode_id']}")
        trace = load_trace(row["trace_path"])
        if trace["state_43"].shape[1:] != (OBSERVATION_DIM,) or trace["raw_pilot_action_7"].shape[1:] != (ACTION_DIM,):
            failures.append(f"contract mismatch {row['episode_id']}")
        if any(not np.isfinite(value).all() for value in trace.values() if value.dtype.kind in "fc"):
            failures.append(f"nonfinite trace {row['episode_id']}")
        if not np.array_equal(trace["clean_pilot_action_7"], trace["raw_pilot_action_7"]):
            failures.append(f"raw_clean mismatch {row['episode_id']}")
        for field in ("raw_pilot_action_7", "postprocessed_action_7", "executed_action_7"):
            if not np.isin(trace[field][:, 6], [postprocessor.normalized_close, postprocessor.normalized_open]).all():
                failures.append(f"noncanonical gripper {field} {row['episode_id']}")
    return {"status": "PASS" if not failures else "FAIL", "gamma": gamma, "expected_rollouts": expected, "nan_count": int(sum(row["nan_count"] for row in rows)), "inf_count": int(sum(row["inf_count"] for row in rows)), "failures": failures[:100]}


def run_gamma(output: Path, gamma: float, gamma_index: int, seeds: tuple[int, ...], pilot: OfflineAWACSurrogatePilot, postprocessor: GlobalActionPostprocessor, controller: GlobalSharedController) -> list[dict[str, Any]]:
    trace_root = output / "traces" / f"gamma_{gamma_key(gamma)}"
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        row = run_episode(
            method="global", paired_seed=seed, pilot_seed=seed + 100_003,
            diffusion_seed=DIFFUSION_SEED_BASE + seed + gamma_index * 100_000,
            global_gamma=gamma, surrogate_pilot=pilot, postprocessor=postprocessor,
            global_controller=controller,
            trace_path=trace_root / f"offline_awac_7p5k_global_gamma_{gamma_key(gamma)}_seed_{seed}.npz",
        )
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"gamma={gamma:.3f}: {len(rows)}/{len(seeds)} complete", flush=True)
    return rows


def safety(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return row["illegal_drop"] <= baseline["illegal_drop"] + .10 and row["ik_failure"] <= baseline["ik_failure"] + .10


def select(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row["gamma"] > 0.0 and safety(row, baseline)]
    if not eligible:
        return None
    best_success = max(row["success"] for row in eligible)
    return min((row for row in eligible if best_success - row["success"] <= .02), key=lambda row: row["gamma"])


def make_plots(output: Path, rows: list[dict[str, Any]], baseline: dict[str, Any]) -> None:
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    rows = sorted(rows, key=lambda r: r["gamma"])
    x = [r["gamma"] for r in rows]
    def draw(filename: str, series: list[tuple[str, str]], ylabel: str, baseline_line: bool = False) -> None:
        # Pillow avoids adding a plotting dependency solely for four raw,
        # unsmoothed validation curves.
        width, height, margin = 960, 560, 70
        image = Image.new("RGB", (width, height), "white"); canvas = ImageDraw.Draw(image)
        values = [value for _, key in series for value in (r[key] for r in rows)]
        if baseline_line: values.append(baseline["success"])
        low, high = min(values), max(values)
        if high <= low: high = low + 1.0
        pad = max((high - low) * .08, .001); low -= pad; high += pad
        px = lambda value: margin + (value - min(x)) / max(max(x) - min(x), 1e-9) * (width - 2 * margin)
        py = lambda value: height - margin - (value - low) / (high - low) * (height - 2 * margin)
        canvas.rectangle((margin, margin, width-margin, height-margin), outline="black")
        colors = ("#2166ac", "#b2182b", "#4d9221")
        for color, (label, key) in zip(colors, series):
            points = [(px(row["gamma"]), py(row[key])) for row in rows]
            canvas.line(points, fill=color, width=3)
            for point in points: canvas.ellipse((point[0]-4, point[1]-4, point[0]+4, point[1]+4), fill=color)
            canvas.text((margin + 12, margin + 18 * (list(series).index((label, key)) + 1)), label, fill=color)
        if baseline_line:
            y = py(baseline["success"]); canvas.line((margin, y, width-margin, y), fill="black", width=1)
        canvas.text((margin, height-42), "gamma", fill="black"); canvas.text((8, margin), ylabel, fill="black")
        image.save(plots / filename)
    draw("gamma_performance.png", [("Success", "success")], "rate", True)
    draw("gamma_failures.png", [("Illegal Drop", "illegal_drop"), ("IK Failure", "ik_failure"), ("Timeout", "timeout")], "rate")
    draw("gamma_correction.png", [("Translation correction", "translation_correction_mm")], "mm / control step")
    draw("gamma_alignment.png", [("Motion cosine", "motion_cosine")], "cosine similarity")


def paired_final(rows: list[dict[str, Any]]) -> dict[str, Any]:
    no = {int(r["paired_seed"]): r for r in rows if r["method"] == "noassist"}
    gl = {int(r["paired_seed"]): r for r in rows if r["method"] == "global"}
    seeds = sorted(set(no) & set(gl))
    a=np.asarray([no[s]["success"] for s in seeds], bool); b=np.asarray([gl[s]["success"] for s in seeds], bool)
    diff=b.astype(float)-a.astype(float)
    steps=np.asarray([gl[s]["episode_steps"]-no[s]["episode_steps"] for s in seeds], float)
    return {"N":len(seeds), "success_difference_global_minus_noassist":float(diff.mean()), "success_difference_95_bootstrap_ci":_bootstrap_ci(diff, 2026081901), "mcnemar_exact":_mcnemar_exact(a,b), "episode_steps_difference_mean_global_minus_noassist":float(steps.mean()), "episode_steps_difference_median_global_minus_noassist":float(np.median(steps)), "episode_steps_difference_95_bootstrap_ci":_bootstrap_ci(steps,2026081902)}


def main() -> None:
    parser=__import__('argparse').ArgumentParser(description="Frozen E1 Global gamma sweep")
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--surrogate-checkpoint',type=Path,default=DEFAULT_SURROGATE_CHECKPOINT)
    parser.add_argument('--device',default='cpu'); parser.add_argument('--run-id')
    args=parser.parse_args(); torch.set_num_threads(1)
    checkpoint=args.checkpoint.expanduser().resolve(); surrogate_path=args.surrogate_checkpoint.expanduser().resolve()
    payload=torch.load(surrogate_path,map_location='cpu',weights_only=False)
    if int(payload.get('step',-1)) != 7500 or np.asarray(payload['observation_mean']).shape != (48,): raise SystemExit('frozen 48D 7.5k pilot required')
    if _sha256(surrogate_path) != '3895e3a9354ed1f0afdb4070d038e5b19587e99b5cb8b2ebb5e6e7d392ebf5f7': raise SystemExit('surrogate SHA256 mismatch')
    stamp=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root=OUTPUT_ROOT/f'gamma_sweep_{stamp}'; root.mkdir(parents=True)
    controller=GlobalSharedController(checkpoint,args.device); pilot=OfflineAWACSurrogatePilot(surrogate_path)
    post=GlobalActionPostprocessor(pilot.normalized_close,pilot.normalized_open)
    if not (np.isclose(post.normalized_close,-.25) and np.isclose(post.normalized_open,1.) and np.isclose(post.threshold,.375)): raise SystemExit('canonical gripper mismatch')
    metadata={'experiment':'exp1_gamma_sweep','checkpoint':str(checkpoint),'global_sha256':_sha256(checkpoint),'surrogate_checkpoint':str(surrogate_path),'surrogate_sha256':_sha256(surrogate_path),'coarse_gammas':COARSE,'validation_seeds':list(VALIDATION_SEEDS),'final_seeds':list(FINAL_SEEDS),'diffusion_seed_mapping':'7000000 + environment_seed + gamma_index * 100000','effective_step_formula':'int((50 - 1) * gamma)','control_dt':CONTROL_DT,'frozen':True,'global_input_dim':43,'global_action_dim':7,'pilot_input_dim':48,'artificial_corruption':False}
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    all_rows=[]; stage_rows=[]; summaries=[]; audits=[]
    for index,gamma in enumerate(COARSE):
        rows=run_gamma(root,gamma,index,VALIDATION_SEEDS,pilot,post,controller); all_rows.extend(rows)
        audit=validate_rows(rows,50,gamma,post); audits.append(audit)
        if audit['status']!='PASS': (root/'validation_audit.json').write_text(json.dumps(audits,indent=2)+'\n'); raise SystemExit(f'structural audit failure gamma={gamma}: {audit["failures"][:1]}')
        summaries.append(summarize_gamma(gamma,rows)); stage_rows.extend(stage_metrics(gamma,rows))
    baseline=next(r for r in summaries if r['gamma']==0.0)
    coarse_best=select(summaries,baseline)
    fine=[]
    if coarse_best is not None and coarse_best['success'] > baseline['success']:
        center=float(coarse_best['gamma']); candidates=[]
        for value in np.arange(max(.001,center-.10), min(1.,center+.10)+1e-9, .025):
            value=round(float(value),3)
            if value not in COARSE and value not in candidates: candidates.append(value)
        for offset,gamma in enumerate(candidates,start=len(COARSE)):
            rows=run_gamma(root,gamma,offset,VALIDATION_SEEDS,pilot,post,controller); all_rows.extend(rows)
            audit=validate_rows(rows,50,gamma,post); audits.append(audit)
            if audit['status']!='PASS': (root/'validation_audit.json').write_text(json.dumps(audits,indent=2)+'\n'); raise SystemExit(f'structural audit failure fine gamma={gamma}: {audit["failures"][:1]}')
            item=summarize_gamma(gamma,rows); summaries.append(item); fine.append(item); stage_rows.extend(stage_metrics(gamma,rows))
    selected=select(summaries,baseline)
    write_csv(root/'gamma_coarse_validation.csv',[r for r in summaries if r['gamma'] in COARSE])
    (root/'gamma_coarse_validation.json').write_text(json.dumps([r for r in summaries if r['gamma'] in COARSE],indent=2)+'\n')
    write_csv(root/'gamma_fine_validation.csv',fine)
    (root/'gamma_fine_validation.json').write_text(json.dumps(fine,indent=2)+'\n')
    write_csv(root/'stage_correction_summary.csv',stage_rows)
    (root/'validation_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    make_plots(root,[r for r in summaries if r['gamma'] in COARSE],baseline)
    selection={'status':'PASS' if selected is not None and selected['success']>baseline['success'] else 'FAIL','baseline':baseline,'coarse_best':coarse_best,'fine_candidates':fine,'selected_gamma':selected,'safety_rule':{'illegal_drop_max':baseline['illegal_drop']+.10,'ik_failure_max':baseline['ik_failure']+.10},'selection_rule':'highest success among safe gamma>0; within 2pp choose smaller gamma'}
    (root/'gamma_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    if selection['status']=='FAIL': print(json.dumps({'validation':'FAIL','output':str(root)},indent=2)); return
    # Confirmatory test: gamma is frozen before these new 300 paired seeds run.
    gamma=float(selected['gamma']); index=1000+int(round(gamma*1000)); final_rows=[]
    (root / 'final_traces').mkdir(exist_ok=True)
    for seed in FINAL_SEEDS:
        for method, run_gamma_value, diffusion_seed in (('noassist',0.,DIFFUSION_SEED_BASE+seed),('global',gamma,DIFFUSION_SEED_BASE+seed+index*100_000)):
            final_rows.append(run_episode(method=method,paired_seed=seed,pilot_seed=seed+100_003,diffusion_seed=diffusion_seed,global_gamma=run_gamma_value,surrogate_pilot=pilot,postprocessor=post,global_controller=controller if method=='global' else None,trace_path=root/'final_traces'/f'offline_awac_7p5k_{method}_gamma_{gamma_key(run_gamma_value)}_seed_{seed}.npz'))
        if len(final_rows)%20==0: print(f'final gamma={gamma:.3f}: {len(final_rows)//2}/300 pairs complete',flush=True)
    final_audit=validate_rows([r for r in final_rows if r['method']=='global'],300,gamma,post)
    final_audit['noassist']=validate_rows([r for r in final_rows if r['method']=='noassist'],300,0.,post)
    final_audit['status']='PASS' if final_audit['status']=='PASS' and final_audit['noassist']['status']=='PASS' else 'FAIL'
    write_csv(root/'final_episode_results.csv',final_rows)
    final_summary={'noassist':summarize_gamma(0.,[r for r in final_rows if r['method']=='noassist']),'global':summarize_gamma(gamma,[r for r in final_rows if r['method']=='global']),'paired_statistics':paired_final(final_rows)}
    (root/'final_summary.json').write_text(json.dumps(final_summary,indent=2)+'\n')
    (root/'final_audit.json').write_text(json.dumps(final_audit,indent=2)+'\n')
    print(json.dumps({'validation':selection['status'],'selected_gamma':gamma,'final_audit':final_audit['status'],'output':str(root)},indent=2))


if __name__ == '__main__': main()
