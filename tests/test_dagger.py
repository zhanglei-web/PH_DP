from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from mujoco_shared_control.actor_bc.dagger import (
    DAggerConfig, action_is_admissible, load_round, module_checksum,
    round_mixture_counts, train_round,
)
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor


def test_round_mixture_keeps_exactly_half_d0() -> None:
    assert round_mixture_counts(256, 1) == [128, 128]
    assert round_mixture_counts(256, 2) == [128, 64, 64]
    assert round_mixture_counts(256, 3) == [128, 43, 43, 42]


def test_native_action_constraints() -> None:
    assert action_is_admissible(np.array([.6, .8, 0, 0, 0, 1, -1]))
    assert not action_is_admissible(np.array([1, 1, 0, 0, 0, 0, 0]))


def test_frozen_log_std_and_checkpoint_reload(tmp_path: Path) -> None:
    torch.manual_seed(3)
    actor = SACConstrainedGaussianActor()
    with torch.no_grad():
        actor.log_std_head.weight.zero_(); actor.log_std_head.bias.fill_(-3)
    original_log_std = deepcopy(actor.log_std_head.state_dict())
    rng = np.random.default_rng(4)
    d0 = (rng.normal(size=(32, 42)).astype(np.float32),
          rng.uniform(-.2, .2, size=(32, 7)).astype(np.float32))
    d1 = (rng.normal(size=(32, 42)).astype(np.float32),
          rng.uniform(-.2, .2, size=(32, 7)).astype(np.float32))
    config = DAggerConfig(batch_size=8, optimizer_steps=2)
    result = train_round(actor, np.zeros(42, np.float32), np.ones(42, np.float32),
                         d0, [d1], tmp_path, config)
    assert result["mixture_counts"] == [4, 4]
    for key, value in original_log_std.items():
        torch.testing.assert_close(actor.log_std_head.state_dict()[key], value, rtol=0, atol=0)
    payload = torch.load(tmp_path / "actor_step_00000.pt", weights_only=False)
    restored = SACConstrainedGaussianActor(); restored.load_state_dict(payload["actor_state_dict"])
    assert module_checksum(restored) != ""


def test_round_loader_rejects_oracle_label_outside_native_action_space(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, state=np.zeros((1, 42), np.float32),
             rule_action=np.array([[1, 1, 0, 0, 0, 0, 0]], np.float32))
    with pytest.raises(ValueError, match="invalid DAgger"):
        load_round(path)


def test_protocol_is_deterministic_and_has_no_rl_fields() -> None:
    config = DAggerConfig()
    assert config.reward_version == "sac_reward_v2_candidate"
    assert config.temporal_stride == 4
    assert not hasattr(config, "alpha")
    assert not hasattr(config, "critic_lr")
    assert not hasattr(config, "exploration_noise")


def test_seed_pools_are_disjoint_and_final_is_sealed() -> None:
    primary = set(range(300_000, 300_100)); secondary = set(range(420_000, 420_100))
    final = set(range(500_000, 500_100))
    rounds = [set(range(1_000_000, 1_001_000)), set(range(1_010_000, 1_011_000)),
              set(range(1_020_000, 1_021_000))]
    pools = [primary, secondary, final, *rounds]
    assert all(not pools[i] & pools[j] for i in range(len(pools)) for j in range(i + 1, len(pools)))
