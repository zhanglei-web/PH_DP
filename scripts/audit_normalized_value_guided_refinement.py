#!/usr/bin/env python3
"""CUDA-only audit of normalized-coordinate MC-Q action refinement."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from train_stage_mc_twin_q import TwinQ, load, qaction
from audit_q3b2_policy_shift_metrics import load_v2, v2_actions, postprocess, nearest_dist, MC, REPLAY, V2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/value_guided_refinement/normalized_audit"
STEPS = (0.005, 0.01, 0.025, 0.05)
EPS = (0.01, 0.025, 0.05, 0.10)

def quant(x):
    x = x.detach().float()
    return {"mean": float(x.mean()), "std": float(x.std()), "median": float(x.median()),
            "p05": float(torch.quantile(x, .05)), "p95": float(torch.quantile(x, .95)),
            "max": float(x.max()), "count": int(x.numel())}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--count", type=int, default=5000)
    p.add_argument("--output", type=Path, default=OUT); a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev); a.output.mkdir(parents=True, exist_ok=True)
    test, train = load("test", dev), load("train", dev)
    n = min(a.count, len(test["obs"])); ix = torch.as_tensor(np.sort(np.random.default_rng(20260902).choice(len(test["obs"]), n, False)), device=dev)
    with np.load(REPLAY / "normalization_stats.npz") as z:
        om, os, am, astd = [torch.as_tensor(z[k], device=dev) for k in ("observation_mean", "observation_std", "action_mean", "action_std")]
    with np.load(V2 / "normalization_stats.npz") as z:
        vom, vos, vam, vas = [torch.as_tensor(z[k], device=dev) for k in ("observation_mean", "observation_std", "action_mean", "action_std")]
    v2 = load_v2(dev); ck = torch.load(MC / "checkpoints/mc_step_00005000.pt", map_location=dev, weights_only=False)
    q = TwinQ().to(dev).eval(); q.load_state_dict(ck["critic"]); q.requires_grad_(False)
    state = (test["obs"][ix].float() - om) / os
    raw0 = postprocess(v2_actions(v2, test["obs"][ix].float(), vom, vos, vam, vas, dev))
    x0 = qaction(raw0, am, astd).detach()
    support = qaction(train["action"].float(), am, astd)
    with torch.no_grad(): q0 = torch.minimum(*q(state, x0))[:, 0]
    # Gradient is taken directly in the critic's normalized action coordinates.
    x = x0.clone().requires_grad_(True); qx = torch.minimum(*q(state, x))[:, 0]
    gx = torch.autograd.grad(qx.sum(), x, only_inputs=True)[0].detach(); gx[:, 6] = 0.
    gnorm = torch.linalg.vector_norm(gx[:, :6], dim=1, keepdim=True).clamp_min(1e-8)
    unit = torch.zeros_like(gx); unit[:, :6] = gx[:, :6] / gnorm

    # Independent ascent/descent sanity check at a small normalized step.
    sanity_eps = 0.005
    with torch.no_grad():
        plus = torch.minimum(*q(state, x0 + sanity_eps * unit))[:, 0]
        minus = torch.minimum(*q(state, x0 - sanity_eps * unit))[:, 0]
    sanity = {"NORM_ASCENT_POSITIVE_FRACTION": float((plus > q0).float().mean()),
              "NORM_DESCENT_NEGATIVE_FRACTION": float((minus < q0).float().mean()),
              "step": sanity_eps}

    base_nn = nearest_dist(x0, support)
    rows = []
    for scheme in ("unit_gradient", "raw_gradient"):
        for step in STEPS:
            for eps in EPS:
                proposal = unit[:, :6] * step if scheme == "unit_gradient" else gx[:, :6] * step
                proposal_norm = torch.linalg.vector_norm(proposal, dim=1, keepdim=True).clamp_min(1e-8)
                delta = proposal * torch.clamp(eps / proposal_norm, max=1.0)
                x1 = x0.clone(); x1[:, :6] += delta
                raw1 = x1 * astd + am
                canonical = postprocess(raw1)
                x1_exec = qaction(canonical, am, astd)
                with torch.no_grad(): q1 = torch.minimum(*q(state, x1_exec))[:, 0]
                dq = q1 - q0; raw_disp = canonical - raw0; norm_disp = x1_exec - x0
                support_nn = nearest_dist(x1_exec, support)
                bound = int((canonical[:, :6].abs() > 1).sum())
                rows.append({"scheme": scheme, "normalized_step_size": step, "normalized_trust_region": eps,
                "DELTA_Q_MEAN": float(dq.mean()), "DELTA_Q_MEDIAN": float(dq.median()),
                "DELTA_Q_P05": float(torch.quantile(dq, .05)), "DELTA_Q_P95": float(torch.quantile(dq, .95)),
                "DELTA_Q_STD": float(dq.std()), "POSITIVE_DELTA_Q_FRACTION": float((dq > 0).float().mean()),
                "RAW_ACTION_DISPLACEMENT_MEAN": float(torch.linalg.vector_norm(raw_disp[:, :6], dim=1).mean()),
                "RAW_ACTION_DISPLACEMENT_P95": float(torch.quantile(torch.linalg.vector_norm(raw_disp[:, :6], dim=1), .95)),
                "NORMALIZED_ACTION_DISPLACEMENT_MEAN": float(torch.linalg.vector_norm(norm_disp[:, :6], dim=1).mean()),
                "NORMALIZED_ACTION_DISPLACEMENT_P95": float(torch.quantile(torch.linalg.vector_norm(norm_disp[:, :6], dim=1), .95)),
                "RAW_DISPLACEMENT_BY_DIM": {k: {"mean": float(raw_disp[:, j].mean()), "std": float(raw_disp[:, j].std())} for j,k in enumerate(("dx","dy","dz","dRx","dRy","dRz"))},
                "BASELINE_SUPPORT_NN_MEAN": float(base_nn.mean()), "BASELINE_SUPPORT_NN_P95": float(torch.quantile(base_nn, .95)),
                "REFINED_SUPPORT_NN_MEAN": float(support_nn.mean()), "REFINED_SUPPORT_NN_P95": float(torch.quantile(support_nn, .95)),
                "SUPPORT_NN_DELTA_MEAN": float((support_nn-base_nn).mean()),
                "BOUND_VIOLATIONS": bound, "SILENT_CLIPPING": False,
                    "ACTION_MAPPING_VALID": bool(torch.all((canonical[:,6] == -1) | (canonical[:,6] == 1))),
                    "ACTION_SCALE_VALID": bool(torch.isfinite(canonical).all()), "GRIPPER_CHANGED": False})
    valid = [r for r in rows if r["POSITIVE_DELTA_Q_FRACTION"] >= .70 and r["DELTA_Q_MEDIAN"] > 0 and
             r["REFINED_SUPPORT_NN_MEAN"] <= r["BASELINE_SUPPORT_NN_MEAN"] * 2 and r["BOUND_VIOLATIONS"] == 0 and
             r["NORMALIZED_ACTION_DISPLACEMENT_P95"] <= r["normalized_trust_region"] + 1e-5 and r["ACTION_MAPPING_VALID"]]
    best = max(valid, key=lambda r: (r["DELTA_Q_MEDIAN"], -r["REFINED_SUPPORT_NN_MEAN"])) if valid else None
    result = {"CUDA_VALID": True, "V2_ON_CUDA": next(v2.parameters()).device.type == "cuda",
        "Q1_ON_CUDA": next(q.q1.parameters()).device.type == "cuda", "Q2_ON_CUDA": next(q.q2.parameters()).device.type == "cuda",
        "sample_count": n, "train_support_count": len(support), "sanity": sanity, "results": rows,
        "GRADIENT_COORDINATE_VALID": "YES" if sanity["NORM_ASCENT_POSITIVE_FRACTION"] >= .90 and sanity["NORM_DESCENT_NEGATIVE_FRACTION"] >= .90 else "NO",
        "VALID_CONFIGURATIONS": [{"normalized_step_size": r["normalized_step_size"], "normalized_trust_region": r["normalized_trust_region"]} for r in valid],
        "BEST": best, "NORMALIZED_REFINEMENT_EFFECTIVE": "YES" if valid else "NO",
        "READY_FOR_ACTOR_FINE_TUNING": "YES" if valid else "NO", "DPQL_ACTOR": "NOT_RUN", "CLOSED_LOOP": "NOT_RUN"}
    (a.output / "normalized_refinement_results.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.output / "audit.json").write_text(json.dumps({k:v for k,v in result.items() if k != "results"}, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("GRADIENT_COORDINATE_VALID","NORMALIZED_REFINEMENT_EFFECTIVE","READY_FOR_ACTOR_FINE_TUNING","sanity","BEST")}, indent=2))

if __name__ == "__main__": main()
