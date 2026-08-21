#!/usr/bin/env python3
"""CUDA-only, read-only offline validation of MC-Q action refinement."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_stage_mc_twin_q import TwinQ, load, qaction
from audit_q3b2_policy_shift_metrics import load_v2, v2_actions, postprocess, nearest_dist, MC, REPLAY, V2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/value_guided_refinement/phase1_offline"
ETAS = (0.0, 0.001, 0.005, 0.01)
EPSILONS = (0.01, 0.02, 0.05)

def stats(x: torch.Tensor) -> dict:
    x = x.detach().float()
    return {"mean": float(x.mean()), "std": float(x.std()),
            "p95": float(torch.quantile(x, .95)), "max": float(x.max())}

def write_summary(path: Path, result: dict) -> None:
    lines = ["# Phase 1: Value-guided Action Refinement Offline Validation", "",
             f"- CUDA_VALID: {result['CUDA_VALID']}",
             f"- V2_ON_CUDA: {result['V2_ON_CUDA']}",
             f"- Q1_ON_CUDA: {result['Q1_ON_CUDA']}",
             f"- Q2_ON_CUDA: {result['Q2_ON_CUDA']}",
             f"- VALUE_REFINEMENT_EFFECTIVE: {result['VALUE_REFINEMENT_EFFECTIVE']}",
             f"- READY_FOR_PHASE2_ACTOR_FINE_TUNING: {result['READY_FOR_PHASE2_ACTOR_FINE_TUNING']}", "",
             "| eta | epsilon | Delta Q mean | positive fraction | displacement mean | support NN mean | clipping rate |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["results"]:
        lines.append("| {eta:.3f} | {epsilon:.3f} | {dq:.6f} | {pos:.4f} | {disp:.6f} | {nn:.6f} | {clip:.4f} |".format(
            eta=r["eta"], epsilon=r["epsilon"], dq=r["DELTA_Q_MEAN"],
            pos=r["POSITIVE_DELTA_Q_FRACTION"], disp=r["ACTION_DISPLACEMENT_MEAN"],
            nn=r["REFINED_SUPPORT_NN"]["mean"], clip=r["CLIPPING_RATE"]))
    path.write_text("\n".join(lines) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    args.output.mkdir(parents=True, exist_ok=True)

    test, train = load("test", dev), load("train", dev)
    n = min(args.count, len(test["obs"]))
    chosen = np.sort(np.random.default_rng(20260901).choice(len(test["obs"]), n, replace=False))
    ix = torch.as_tensor(chosen, device=dev)
    with np.load(REPLAY / "normalization_stats.npz") as z:
        om, os, am, astd = [torch.as_tensor(z[k], device=dev) for k in
                            ("observation_mean", "observation_std", "action_mean", "action_std")]
    with np.load(V2 / "normalization_stats.npz") as z:
        vom, vos, vam, vas = [torch.as_tensor(z[k], device=dev) for k in
                              ("observation_mean", "observation_std", "action_mean", "action_std")]
    v2 = load_v2(dev)
    checkpoint = torch.load(MC / "checkpoints/mc_step_00005000.pt", map_location=dev, weights_only=False)
    critic = TwinQ().to(dev).eval(); critic.load_state_dict(checkpoint["critic"])
    critic.requires_grad_(False)
    # V2 output is inverse-normalized into semantic action space, then formal evaluator semantics apply.
    baseline = postprocess(v2_actions(v2, test["obs"][ix].float(), vom, vos, vam, vas, dev))
    state = (test["obs"][ix].float() - om) / os
    support = qaction(train["action"].float(), am, astd)
    with torch.no_grad():
        q0 = torch.minimum(*critic(state, qaction(baseline, am, astd)))[:, 0]
    baseline_nn = nearest_dist(qaction(baseline, am, astd), support)

    # Differentiate only through the six continuous semantic action channels; gripper remains canonical g0.
    seed_action = baseline.clone().detach().requires_grad_(True)
    q_seed = torch.minimum(*critic(state, qaction(seed_action, am, astd)))[:, 0]
    grad = torch.autograd.grad(q_seed.sum(), seed_action, only_inputs=True)[0]
    grad[:, 6] = 0.0
    results = []
    for eta in ETAS:
        raw_proposal = baseline + eta * grad
        raw_proposal[:, 6] = baseline[:, 6]
        raw_delta = raw_proposal[:, :6] - baseline[:, :6]
        for epsilon in EPSILONS:
            norm = torch.linalg.vector_norm(raw_delta, dim=1, keepdim=True)
            projected_delta = raw_delta * torch.clamp(epsilon / norm.clamp_min(1e-12), max=1.0)
            candidate = baseline.clone()
            candidate[:, :6] += projected_delta
            # Record pre-postprocess bounds. Formal semantic postprocess is explicit, never silent.
            proposal_bounds = (candidate[:, :6].abs() > 1.0)
            canonical = postprocess(candidate)
            displacement = torch.linalg.vector_norm(canonical[:, :6] - baseline[:, :6], dim=1)
            with torch.no_grad():
                q_new = torch.minimum(*critic(state, qaction(canonical, am, astd)))[:, 0]
            delta_q = q_new - q0
            refined_nn = nearest_dist(qaction(canonical, am, astd), support)
            clipping_count = int((norm[:, 0] > epsilon).sum())
            bound_violations = int(proposal_bounds.sum())
            # Canonical action must match global evaluator execution semantics exactly.
            mapping_valid = bool(torch.all(canonical[:, :6].abs() <= 1.0) and
                                 torch.all((canonical[:, 6] == -1.0) | (canonical[:, 6] == 1.0)))
            results.append({
                "eta": eta, "epsilon": epsilon,
                "DELTA_Q_MEAN": float(delta_q.mean()), "DELTA_Q_STD": float(delta_q.std()),
                "POSITIVE_DELTA_Q_FRACTION": float((delta_q > 0).float().mean()),
                "ACTION_DISPLACEMENT_MEAN": float(displacement.mean()),
                "ACTION_DISPLACEMENT_STD": float(displacement.std()),
                "ACTION_DISPLACEMENT_P95": float(torch.quantile(displacement, .95)),
                "BASELINE_SUPPORT_NN": stats(baseline_nn), "REFINED_SUPPORT_NN": stats(refined_nn),
                "SUPPORT_NN_MEAN_INCREASE": float(refined_nn.mean() - baseline_nn.mean()),
                "CLIPPING_COUNT": clipping_count, "CLIPPING_RATE": clipping_count / n,
                "PROPOSAL_BOUND_VIOLATIONS": bound_violations,
                "BOUND_VIOLATIONS": int(((canonical[:, :6].abs() > 1.0)).sum()),
                "SILENT_CLIPPING": False, "GRIPPER_CHANGED": False,
                "ACTION_MAPPING_VALID": mapping_valid, "ACTION_SCALE_VALID": bool(torch.isfinite(canonical).all()),
            })
    # "Small" is intentionally explicit: no more than 0.05 average normalized NN increase.
    valid = [r for r in results if r["eta"] > 0 and r["DELTA_Q_MEAN"] > 0 and
             r["POSITIVE_DELTA_Q_FRACTION"] > .7 and r["SUPPORT_NN_MEAN_INCREASE"] <= .05 and
             r["BOUND_VIOLATIONS"] == 0 and r["ACTION_DISPLACEMENT_MEAN"] <= r["epsilon"] and
             r["ACTION_MAPPING_VALID"] and r["ACTION_SCALE_VALID"]]
    result = {
        "CUDA_VALID": True, "V2_ON_CUDA": next(v2.parameters()).device.type == "cuda",
        "Q1_ON_CUDA": next(critic.q1.parameters()).device.type == "cuda",
        "Q2_ON_CUDA": next(critic.q2.parameters()).device.type == "cuda",
        "MC_CRITIC_CHECKPOINT": str((MC / "checkpoints/mc_step_00005000.pt").resolve()),
        "TEST_SPLIT": "frozen held-out test", "sample_count": n, "train_support_count": len(support),
        "ACTION_SEMANTICS": "V2 inverse-normalized semantic action; continuous6 clip[-1,1]; gripper threshold 0.375",
        "BASELINE_Q": stats(q0), "BASELINE_SUPPORT_NN": stats(baseline_nn),
        "results": results, "VALID_CONFIGURATIONS": [{"eta": r["eta"], "epsilon": r["epsilon"]} for r in valid],
        "VALUE_REFINEMENT_EFFECTIVE": "YES" if valid else "NO",
        "READY_FOR_PHASE2_ACTOR_FINE_TUNING": "YES" if valid else "NO",
        "DPQL_ACTOR": "NOT_RUN", "CLOSED_LOOP": "NOT_RUN", "TRAINING_PERFORMED": False,
    }
    (args.output / "refinement_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "audit.json").write_text(json.dumps({k: result[k] for k in result if k != "results"}, indent=2) + "\n")
    write_summary(args.output / "summary.md", result)
    print(json.dumps({"VALUE_REFINEMENT_EFFECTIVE": result["VALUE_REFINEMENT_EFFECTIVE"],
                      "VALID_CONFIGURATIONS": result["VALID_CONFIGURATIONS"]}, indent=2))

if __name__ == "__main__":
    main()
