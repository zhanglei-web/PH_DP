#!/usr/bin/env python3
"""Final-action Q refinement audit; no environment or closed-loop execution."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from audit_stage_q_gradient import load
from train_stage_value_q_recovery import QNet
from mujoco_shared_control.rss2023.oracle_stage_embedding_model import (
    StageEmbeddingDiffusion, StageEmbeddingDiffusionConfig,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stage_value_guidance/v2_stage_q_recovery_value_v2'
CKPT = ROOT / 'outputs/oracle_stage_diffusion/oracle_stage_embedding_v2_20260818/checkpoints/step_00080000.pt'


def main() -> None:
    state, _raw, om, os, am, astd = load()
    rng = np.random.default_rng(20260823)
    ix = rng.choice(len(state), 1000, replace=False)
    state = state[ix]
    stage = np.argmax(state[:, 43:48], axis=1).astype(np.int64)

    payload = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = StageEmbeddingDiffusionConfig(**{
        k: v for k, v in payload['diffusion_config'].items()
        if k in StageEmbeddingDiffusionConfig.__dataclass_fields__
    })
    v2 = StageEmbeddingDiffusion(cfg).eval()
    v2.load_state_dict(payload['model'])
    obs = torch.from_numpy(((state - om) / os).astype('f4'))
    human = torch.zeros((len(state), 7), dtype=torch.float32)
    generated = []
    # Batch generation is mathematically the same assist() pipeline as the evaluator.
    for start in range(0, len(state), 100):
        end = min(start + 100, len(state))
        gen = torch.Generator(device='cpu').manual_seed(2026082300 + start)
        with torch.inference_mode():
            internal = v2.assist(obs[start:end], human[start:end], gamma=1.0, generator=gen)
        generated.append((internal.numpy() * astd + am).astype('f4'))
    raw_generated = np.concatenate(generated, axis=0)

    # This is the explicit historical evaluator postprocess in semantic space.
    clipped = np.clip(raw_generated, -1.0, 1.0)
    env_action = clipped.copy()
    env_action[:, 6] = np.where(env_action[:, 6] < 0.375, -1.0, 1.0)
    clip_count = int(np.count_nonzero(np.abs(raw_generated[:, :6]) > 1.0))
    gripper_binary_count = int(np.count_nonzero(raw_generated[:, 6] != env_action[:, 6]))

    q = QNet()
    q.load_state_dict(torch.load(OUT / 'q_checkpoint_valid.pt', map_location='cpu', weights_only=False)['model'])
    q.eval()
    s = torch.from_numpy(((state - om) / os).astype('f4'))
    ar = torch.from_numpy(env_action).float().requires_grad_(True)
    mean, std = torch.from_numpy(am), torch.from_numpy(astd)

    def score(action: torch.Tensor) -> torch.Tensor:
        return q(s, (action - mean) / std).squeeze(-1)

    base = score(ar)
    grad = torch.autograd.grad(base.sum(), ar)[0]
    grad[:, 6] = 0.0
    norm = torch.linalg.vector_norm(grad[:, :6], dim=1, keepdim=True)
    delta = grad / (norm + 1e-12) * 1e-5
    refined = ar.detach() + delta
    refined[:, 6] = ar.detach()[:, 6]
    refined_q = score(refined)
    delta_q = refined_q - base.detach()

    result = {
        'sample_count': 1000,
        'checkpoint': str(CKPT),
        'action_semantics': '[dx,dy,dz,dRx,dRy,dRz,gripper] normalized semantic 7D',
        'v2_to_q_mapping': 'identity after explicit evaluator clip and gripper threshold',
        'raw_v2_out_of_continuous_bounds_count': clip_count,
        'explicit_gripper_threshold_changes': gripper_binary_count,
        'silent_clipping': False,
        'FINAL_REFINEMENT_DELTA_Q_MEAN': float(delta_q.mean()),
        'FINAL_REFINEMENT_DELTA_Q_POSITIVE_FRACTION': float((delta_q > 0).float().mean()),
        'FINAL_REFINEMENT_ACTION_DISPLACEMENT_MEAN': float(torch.linalg.vector_norm(delta[:, :6], dim=1).mean()),
        'FINAL_REFINEMENT_ACTION_DISPLACEMENT_P95': float(torch.quantile(torch.linalg.vector_norm(delta[:, :6], dim=1), .95)),
        'FINAL_REFINEMENT_VALID': 'YES' if float((delta_q > 0).float().mean()) >= .90 else 'NO',
    }
    (OUT / 'q_final_action_refinement_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
