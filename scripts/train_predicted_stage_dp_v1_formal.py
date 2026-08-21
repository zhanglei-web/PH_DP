#!/usr/bin/env python3
"""CUDA-only formal 80k training for Predicted-Stage-DP-v1."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam

from train_predicted_stage_dp_v1 import DATA, TCNCK, TCNNORM, build_split, sha
from mujoco_shared_control.stage.tcn import StageTCNV1
from mujoco_shared_control.stage.dataset import fit_normalization, load_split
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--output", type=Path, default=Path("outputs/predicted_stage_dp_v1/formal"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    if args.steps < 80000:
        raise ValueError("formal training requires at least 80000 steps")
    args.output.mkdir(parents=True, exist_ok=True)

    payload = torch.load(TCNCK, map_location=device, weights_only=False)
    tcn = StageTCNV1().to(device).eval(); tcn.load_state_dict(payload["model"]); tcn.requires_grad_(False)
    with np.load(TCNNORM) as stats:
        tmean, tstd = stats["mean"].astype("f4"), stats["std"].astype("f4")
    fitted = fit_normalization(load_split(DATA, "train"))
    norm_diff = max(float(np.max(np.abs(tmean - fitted.mean))), float(np.max(np.abs(tstd - fitted.std))))
    if norm_diff > 1e-6:
        raise RuntimeError(f"TCN normalization differs from frozen train split: {norm_diff}")
    train, val = build_split("train", tmean, tstd), build_split("validation", tmean, tstd)
    train_state = np.stack([x[0] for x in train]); train_action = np.stack([x[1] for x in train])
    obs_mean = train_state.mean(0).astype("f4"); obs_std = np.maximum(train_state.std(0), 1e-6).astype("f4")
    act_mean = train_action.mean(0).astype("f4"); act_std = np.maximum(train_action.std(0), 1e-6).astype("f4")
    state = torch.as_tensor((train_state - obs_mean) / obs_std, device=device)
    action = torch.as_tensor((train_action - act_mean) / act_std, device=device)
    history = torch.as_tensor(np.stack([x[2] for x in train]), device=device)
    vstate = torch.as_tensor((np.stack([x[0] for x in val]) - obs_mean) / obs_std, device=device)
    vaction = torch.as_tensor((np.stack([x[1] for x in val]) - act_mean) / act_std, device=device)
    vhistory = torch.as_tensor(np.stack([x[2] for x in val]), device=device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = StageEmbeddingDiffusionConfig(physical_dim=43, stage_dim=5, stage_embedding_dim=32,
        condition_hidden_dim=128, action_dim=7, num_diffusion_steps=50, hidden_dim=128)
    model = StageEmbeddingDiffusion(cfg).to(device).train(); optimizer = Adam(model.parameters(), lr=1e-3)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    tcn_snapshot = {k: v.detach().clone() for k, v in tcn.state_dict().items()}

    def val_loss(step: int) -> float:
        model.eval(); torch.manual_seed(args.seed + step)
        values = []
        with torch.no_grad():
            for start in range(0, len(vstate), args.batch_size):
                sl = slice(start, start + args.batch_size)
                z = tcn.posterior(vhistory[sl]); values.append(float(model.loss(torch.cat((vstate[sl], z), 1), vaction[sl])))
        model.train(); return float(np.mean(values))

    logs = []; checkpoints = []
    for step in range(1, args.steps + 1):
        ix = torch.randint(len(state), (args.batch_size,), device=device, generator=rng)
        with torch.no_grad(): z = tcn.posterior(history[ix])
        loss = model.loss(torch.cat((state[ix], z), 1), action[ix])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                entropy = float((-(z * z.clamp_min(1e-8).log()).sum(-1)).mean())
                maxprob = float(z.max(-1).values.mean())
            row = {"step": step, "train_L_diff": float(loss.detach()), "val_L_diff": None,
                   "posterior_entropy": entropy, "posterior_max_probability": maxprob,
                   "NaN": bool(not torch.isfinite(loss)), "Inf": bool(torch.isinf(loss)),
                   "TCN_PARAMETER_MAX_ABS_DIFF": max(float((v - tcn_snapshot[k]).abs().max()) for k, v in tcn.state_dict().items())}
            if step % 10000 == 0:
                row["val_L_diff"] = val_loss(step)
                checkpoint = args.output / "checkpoints" / f"predicted_stage_dp_step_{step:06d}.pt"; checkpoint.parent.mkdir(exist_ok=True)
                torch.save({"format_version": "predicted-stage-dp-v1-formal", "step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "diffusion_config": cfg.state_dict(),
                    "normalization": {"observation_mean": obs_mean, "observation_std": obs_std, "action_mean": act_mean, "action_std": act_std},
                    "tcn_checkpoint": str(TCNCK.resolve()), "tcn_sha256": sha(TCNCK), "tcn_norm_sha256": sha(TCNNORM),
                    "config": {"steps": args.steps, "batch_size": args.batch_size, "learning_rate": 1e-3, "loss": "L_diff only", "soft_posterior": True}}, checkpoint)
                row["checkpoint"] = str(checkpoint); checkpoints.append(row)
            logs.append(row)
            print(json.dumps(row), flush=True)
    summary_tcn = max(x["TCN_PARAMETER_MAX_ABS_DIFF"] for x in logs)
    training_valid = bool(checkpoints) and all(not x["NaN"] and not x["Inf"] for x in logs)
    summary = {"steps": args.steps, "checkpoint_steps": [x["step"] for x in checkpoints],
               "TCN_PARAMETER_MAX_ABS_DIFF": float(summary_tcn), "CUDA_ONLY": True,
               "TCN_A_FROZEN": True, "soft_posterior": True, "loss": "L_diff only", "dataset": str(DATA.resolve()),
               "tcn_checkpoint": str(TCNCK.resolve()), "tcn_sha256": sha(TCNCK), "normalization_max_abs_diff": norm_diff,
               "PREDICTED_STAGE_DP_TRAINING_VALID": "YES" if training_valid and summary_tcn == 0.0 else "NO"}
    (args.output / "config.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "training_log.jsonl").write_text("\n".join(json.dumps(x) for x in logs) + "\n")
    (args.output / "checkpoint_summary.json").write_text(json.dumps(checkpoints, indent=2) + "\n")


if __name__ == "__main__":
    main()
