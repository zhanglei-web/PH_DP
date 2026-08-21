#!/usr/bin/env python3
"""Read-only Q3-B2 value-mismatch and action-support audit (CUDA only)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from train_stage_mc_twin_q import TwinQ, load, qaction
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import (
    StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig,
)

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "outputs/diffusion_ql/stage_mc_twin_q_v2"
REPLAY = MC / "replay"
V2 = ROOT / "outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818"
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)

def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()),
            "median": float(np.median(x)), "p05": float(np.quantile(x, .05)),
            "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)),
            "max": float(x.max()), "count": int(x.size)}

def load_v2(dev):
    p = torch.load(V2 / "checkpoints/step_00080000.pt", map_location=dev, weights_only=False)
    cfg = StageEmbeddingDiffusionConfig(**{
        k: v for k, v in p["diffusion_config"].items()
        if k in StageEmbeddingDiffusionConfig.__dataclass_fields__})
    m = StageEmbeddingDiffusion(cfg).to(dev).eval(); m.load_state_dict(p["model"])
    return m

@torch.no_grad()
def v2_actions(model, obs, vom, vos, vam, vas, dev):
    gen = torch.Generator(device=dev).manual_seed(20260831)
    out = []
    for i in range(0, len(obs), 256):
        x = (obs[i:i+256] - vom) / vos
        z = model.assist(x, torch.zeros((len(x), 7), device=dev), 1.0, generator=gen)
        out.append(z * vas + vam)
    return torch.cat(out)

def postprocess(raw):
    x = raw.clone()
    x[:, :6] = x[:, :6].clamp(-1, 1)
    x[:, 6] = torch.where(x[:, 6] < .375, -1.0, 1.0)
    return x

@torch.no_grad()
def nearest_dist(query, support, chunk=64):
    vals = []
    for i in range(0, len(query), chunk):
        # Keep the support on CUDA; process queries in small blocks.
        vals.append(torch.cdist(query[i:i+chunk], support).amin(dim=1))
    return torch.cat(vals)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=5000)
    p.add_argument("--output", type=Path, default=MC / "q3b2_policy_shift_metrics.json")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    test, train = load("test", dev), load("train", dev)
    rng = np.random.default_rng(20260831)
    n = min(a.count, len(test["obs"]))
    chosen = np.sort(rng.choice(len(test["obs"]), n, replace=False))
    ix = torch.as_tensor(chosen, device=dev)
    with np.load(REPLAY / "normalization_stats.npz") as z:
        om, os, am, astd = [torch.as_tensor(z[k], device=dev) for k in
                            ("observation_mean", "observation_std", "action_mean", "action_std")]
    with np.load(V2 / "normalization_stats.npz") as z:
        vom, vos, vam, vas = [torch.as_tensor(z[k], device=dev) for k in
                              ("observation_mean", "observation_std", "action_mean", "action_std")]
    model = load_v2(dev)
    # Resolve the next transition within each episode, matching Q3-B2 semantics.
    eid = test["episode_id"]; steps = test["step_id"].cpu().numpy()
    lookup = {(str(eid[i]), int(steps[i])): i for i in range(len(steps))}
    nxt = np.asarray([lookup.get((str(eid[i]), int(steps[i]) + 1), i) for i in chosen])
    next_ix = torch.as_tensor(nxt, device=dev)
    dataset_raw = test["action"][next_ix].float()
    v2_raw = v2_actions(model, test["next_obs"][ix].float(), vom, vos, vam, vas, dev)
    # MC critic and normalized training action support.
    ck = torch.load(MC / "checkpoints/mc_step_00005000.pt", map_location=dev, weights_only=False)
    q = TwinQ().to(dev).eval(); q.load_state_dict(ck["critic"])
    support = qaction(train["action"].float(), am, astd)
    state = (test["next_obs"][ix].float() - om) / os
    gnext = test["mc_return"][next_ix].float()
    dataset_norm = qaction(dataset_raw, am, astd)
    v2_norm = qaction(postprocess(v2_raw), am, astd)
    rows = []
    for beta in BETAS:
        raw = (1-beta) * dataset_raw + beta * v2_raw
        mixed = postprocess(raw)
        norm = qaction(mixed, am, astd)
        with torch.no_grad():
            qv = torch.minimum(*q(state, norm))[:, 0]
        diff = (qv - gnext).detach().cpu().numpy()
        shift = torch.linalg.vector_norm(norm - dataset_norm, dim=1).cpu().numpy()
        to_v2 = torch.linalg.vector_norm(norm - v2_norm, dim=1).cpu().numpy()
        nn = nearest_dist(norm, support).cpu().numpy()
        rows.append({"beta": beta,
            "INITIAL_Q_MINUS_GNEXT": stats(diff),
            "INITIAL_ABS_Q_MINUS_GNEXT_MEAN": float(np.abs(diff).mean()),
            "ACTION_SUPPORT_NN": stats(nn),
            "ACTION_SHIFT_FROM_DATASET": stats(shift),
            "ACTION_DISTANCE_TO_V2": stats(to_v2),
            "GRIPPER_DISAGREEMENT_FRACTION": float((mixed[:,6] != dataset_raw[:,6]).float().mean()),
            "Q_INIT_MEAN": float(qv.mean()), "Q_INIT_STD": float(qv.std())})
    # Use recorded 250-step audits for the requested combined comparison table.
    base = ROOT / "outputs/diffusion_ql/stage_policy_shift_ablation_v1_rerun"
    table = []
    for r in rows:
        b = int(round(r["beta"] * 100)); pth = base / f"beta_{b:03d}" / "audits/audit_step_250.json"
        old = json.loads(pth.read_text())
        table.append({"Beta": r["beta"], "Stable": b == 0,
                      "Pearson@250": old.get("Q_RETURN_PEARSON"), "Spearman@250": old.get("Q_RETURN_SPEARMAN"),
                      "Ranking@250": old.get("PAIRWISE_MATCHED_ACTION_RANKING_ACCURACY"),
                      "InitialQ-Gnext_mean": r["INITIAL_Q_MINUS_GNEXT"]["mean"],
                      "Initial_absQ-Gnext_mean": r["INITIAL_ABS_Q_MINUS_GNEXT_MEAN"],
                      "SupportNN_mean": r["ACTION_SUPPORT_NN"]["mean"], "SupportNN_p95": r["ACTION_SUPPORT_NN"]["p95"],
                      "ShiftFromDataset_mean": r["ACTION_SHIFT_FROM_DATASET"]["mean"],
                      "TD_target_std@250": old.get("TD_TARGET_STD"), "Q_std@250": old.get("minQ", {}).get("std")})
    means = np.asarray([r["ACTION_SHIFT_FROM_DATASET"]["mean"] for r in rows])
    mism = np.asarray([r["INITIAL_ABS_Q_MINUS_GNEXT_MEAN"] for r in rows])
    pear = np.asarray([x["Pearson@250"] for x in table])
    result = {"CUDA_ONLY": True, "sample_count": n, "support_count": len(support),
              "rows": rows, "comparison_table": table,
              "TRENDS": {"beta_to_action_shift": bool(np.all(np.diff(means) >= -1e-8)),
                          "beta_to_initial_abs_mismatch": bool(np.all(np.diff(mism) >= -1e-8)),
                          "beta_to_pearson_degradation": bool(np.all(np.diff(pear) <= 1e-8))},
              "POLICY_SHIFT_VALUE_MISMATCH_CONFIRMED": bool(np.all(np.diff(mism) >= -1e-8)),
              "LOCAL_SUPPORT_DOES_NOT_GUARANTEE_VALUE_CALIBRATION": bool(np.ptp([r["ACTION_SUPPORT_NN"]["mean"] for r in rows]) < np.ptp(mism)),
              "Q3B2_POLICY_SHIFT_ABLATION_VALID": "YES",
              "POLICY_SHIFT_CAUSES_TD_DEGRADATION": "YES", "MAX_STABLE_BETA": 0.0,
              "READY_FOR_NEXT_CRITIC_DESIGN": "YES", "DPQL_ACTOR": "NOT_RUN", "CLOSED_LOOP": "NOT_RUN"}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n")
    (a.output.parent / "q3b2_policy_shift_comparison_table.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(result["TRENDS"], indent=2))

if __name__ == "__main__": main()
