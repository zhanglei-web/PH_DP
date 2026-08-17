from __future__ import annotations

import numpy as np
import torch

from mujoco_shared_control.actor_bc.model import ActorBC, parameter_count
from mujoco_shared_control.actor_bc.evaluate import ActorPredictor
from mujoco_shared_control.experts.interfaces import ExpertActionSpec
from mujoco_shared_control.actor_bc.train import _metrics, _normalization


def test_actor_bc_has_frozen_shape_and_parameter_count() -> None:
    model = ActorBC()
    assert model(torch.zeros(3, 42)).shape == (3, 7)
    assert parameter_count(model) == 144_391


def test_train_only_normalization_preserves_constant_dimensions() -> None:
    values = np.ones((4, 42), dtype=np.float32)
    values[:, 0] = np.arange(4)
    mean, scale, raw_std = _normalization(values, 1e-6)
    assert raw_std[1] == 0.0
    assert scale[1] == 1.0
    np.testing.assert_allclose(((values - mean) / scale)[:, 1], 0.0)


def test_metrics_use_fixed_gripper_midpoint_and_physical_xyz() -> None:
    target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25]])
    prediction = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.374]])
    metrics = _metrics(prediction, target)
    assert metrics["gripper_accuracy"] == 1.0
    assert abs(metrics["xyz_error_m"] - 0.01) < 1e-8


def test_action_spec_round_trip_keeps_frozen_bc_action_semantics() -> None:
    spec = ExpertActionSpec()
    normalized = np.array([1, -1, 0.5, 1, -1, 0.5, -0.25], dtype=np.float64)
    command = spec.denormalize(normalized)
    np.testing.assert_allclose(command, [0.025, -0.025, 0.0125, 0.1, -0.1, 0.05, 0.03])
    np.testing.assert_allclose(spec.normalize(command), normalized)
