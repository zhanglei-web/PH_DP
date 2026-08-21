from __future__ import annotations

import numpy as np
import torch

from mujoco_shared_control.experts.recovery_bc import RecoveryBCPolicy


def test_recovery_bc_policy_has_43d_input_and_canonical_gripper() -> None:
    policy = RecoveryBCPolicy()
    motion, logits = policy(torch.zeros(3, 43))
    assert motion.shape == (3, 6)
    assert logits.shape == (3,)
    action = policy.action(np.zeros(43, np.float32), np.zeros(43, np.float32), np.ones(43, np.float32))
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert action[6] in (-0.25, 1.0)
