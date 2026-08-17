from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.actor_bc.model import ActorBC
from mujoco_shared_control.sac.actor import (
    SACGaussianActor,
    freeze_for_mean_calibration,
    initialize_from_bc,
)


CHECKPOINT = Path("outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt")
CALIBRATED = Path("outputs/sac_actor/sac_actor_v1_20260812T160000Z/actor_initialized.pt")
FULL_DISTILLED = Path(
    "outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/actor_initialized.pt"
)


def test_one_calibration_step_changes_only_mean_head_and_moves_toward_bc() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    bc = ActorBC(); bc.load_state_dict(checkpoint["model_state_dict"]); bc.eval()
    actor = SACGaussianActor(); initialize_from_bc(actor, CHECKPOINT)
    before = deepcopy(actor.state_dict())
    freeze_for_mean_calibration(actor)
    state = torch.randn(64, 42)
    with torch.no_grad():
        bc_action = torch.clamp(bc(state), -1, 1)
        target = torch.atanh(torch.clamp(bc_action, -1+1e-6, 1-1e-6))
        loss_before = torch.nn.functional.mse_loss(actor.mean_head(actor.trunk(state)), target)
    prediction = actor.mean_head(actor.trunk(state).detach())
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    with torch.no_grad():
        for parameter in actor.mean_head.parameters():
            parameter.add_(parameter.grad, alpha=-1e-3)
    after = actor.state_dict()
    for name in after:
        if name.startswith("mean_head."):
            assert not torch.equal(before[name], after[name])
        else:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    with torch.no_grad():
        loss_after = torch.nn.functional.mse_loss(actor.mean_head(actor.trunk(state)), target)
    assert loss_after < loss_before
    mean, log_std, _ = actor.distribution_stats(state)
    action = actor.deterministic_action(state)
    assert torch.isfinite(mean).all() and torch.isfinite(log_std).all() and torch.isfinite(action).all()
    assert torch.all(log_std == -3.0)
    assert np.isclose(0.5 * (0.999 + 1.0) * .08, .07996)


def test_saved_calibration_preserves_exact_bc_trunk_and_frozen_log_std() -> None:
    bc = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"]
    calibrated = torch.load(CALIBRATED, map_location="cpu", weights_only=False)["actor_state_dict"]
    mapping = {
        "network.0.weight":"trunk.0.weight", "network.0.bias":"trunk.0.bias",
        "network.2.weight":"trunk.2.weight", "network.2.bias":"trunk.2.bias",
        "network.4.weight":"trunk.4.weight", "network.4.bias":"trunk.4.bias",
    }
    for source, target in mapping.items():
        torch.testing.assert_close(bc[source], calibrated[target], rtol=0, atol=0)
    assert torch.count_nonzero(calibrated["log_std_head.weight"]) == 0
    torch.testing.assert_close(
        calibrated["log_std_head.bias"], torch.full((7,), -3.0), rtol=0, atol=0
    )
    assert not torch.equal(bc["network.6.weight"], calibrated["mean_head.weight"])


def test_full_distilled_artifact_is_finite_frozen_std_and_reload_exact() -> None:
    payload = torch.load(FULL_DISTILLED, map_location="cpu", weights_only=False)
    first = SACGaussianActor(); first.load_state_dict(payload["actor_state_dict"]); first.eval()
    second = SACGaussianActor(); second.load_state_dict(payload["actor_state_dict"]); second.eval()
    states = torch.randn(32, 42)
    with torch.inference_mode():
        first_action = first.deterministic_action(states)
        second_action = second.deterministic_action(states)
        mean, log_std, _ = first.distribution_stats(states)
    torch.testing.assert_close(first_action, second_action, rtol=0, atol=0)
    assert torch.isfinite(mean).all() and torch.isfinite(first_action).all()
    assert torch.all(first_action.abs() < 1)
    assert torch.all(log_std == -3)
    assert torch.count_nonzero(payload["actor_state_dict"]["log_std_head.weight"]) == 0


def test_bc_teacher_can_be_fully_frozen() -> None:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    teacher = ActorBC(); teacher.load_state_dict(checkpoint["model_state_dict"]); teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    assert not teacher.training
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
