from __future__ import annotations

import numpy as np
import torch

from mujoco_shared_control.experts.temporal_recovery_bc import UnifiedStageAwareTemporalBC


def test_temporal_policy_has_expected_output_and_canonical_gripper() -> None:
    policy = UnifiedStageAwareTemporalBC()
    motion, logits = policy(torch.zeros(3, 20, 48))
    assert motion.shape == (3, 6)
    assert logits.shape == (3,)
    action = policy.action(np.zeros((20, 48), np.float32))
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert action[6] in (-0.25, 1.0)


def test_causal_features_are_invariant_to_future_suffix() -> None:
    torch.manual_seed(7)
    policy = UnifiedStageAwareTemporalBC().eval()
    prefix = torch.randn(2, 11, 48)
    future = torch.randn(2, 9, 48)
    full = torch.cat((prefix, future), dim=1)
    with torch.no_grad():
        prefix_features = policy.temporal_features(prefix)[:, -1]
        full_features_at_prefix_end = policy.temporal_features(full)[:, 10]
    torch.testing.assert_close(prefix_features, full_features_at_prefix_end, rtol=0, atol=0)
