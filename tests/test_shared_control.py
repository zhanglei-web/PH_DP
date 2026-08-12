from __future__ import annotations

import numpy as np
import pytest

from mujoco_shared_control.shared_control import (
    CommandSpace,
    HumanPassthroughPolicy,
    ModelInput,
    ModelInputSpec,
    ModelOutput,
    load_policy,
)


def _input(*, human_valid: bool = True, active: bool = True) -> ModelInput:
    spec = ModelInputSpec(
        history_length=4,
        use_rgb=True,
        use_depth=True,
        image_history_length=1,
    )
    human = np.zeros((spec.history_length, 8), dtype=np.float32)
    human[-1] = [0.4, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0, 0.06]
    return ModelInput(
        timestamp=1.25,
        step_index=25,
        task_name="pick_place",
        spec=spec,
        state_history=np.zeros((spec.history_length, 26), dtype=np.float32),
        human_action_history=human,
        executed_action_history=np.zeros((4, 8), dtype=np.float32),
        history_timestamps=np.arange(4, dtype=np.float64) * 0.05,
        human_action_timestamps=np.arange(4, dtype=np.float64) * 0.05,
        human_action_age_ms=np.zeros(4, dtype=np.float32),
        history_valid=np.ones(4, dtype=np.bool_),
        human_action_valid=np.array([False, False, False, human_valid]),
        human_control_active=np.array([False, False, False, active]),
        rgb={"front": np.zeros((1, 12, 16, 3), dtype=np.uint8)},
        depth={"front": np.ones((1, 12, 16), dtype=np.float32)},
        image_timestamps={"front": np.array([1.25])},
        image_valid={"front": np.array([True])},
    )


def test_human_passthrough_uses_canonical_cartesian_command() -> None:
    output = HumanPassthroughPolicy().predict(_input())

    assert output.valid
    assert output.control_active
    assert output.command_space == CommandSpace.CARTESIAN_POSE
    np.testing.assert_allclose(
        output.command, [0.4, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0, 0.06]
    )


def test_human_passthrough_marks_missing_human_command_invalid() -> None:
    output = HumanPassthroughPolicy().predict(_input(human_valid=False))
    assert not output.valid
    assert np.isnan(output.command).all()


def test_model_output_normalizes_cartesian_quaternion() -> None:
    output = ModelOutput(
        timestamp=0.0,
        command=np.array([0.4, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.08]),
    )
    np.testing.assert_allclose(output.command[3:7], [1.0, 0.0, 0.0, 0.0])


def test_model_input_rejects_missing_requested_depth() -> None:
    with pytest.raises(ValueError, match="depth cameras"):
        data = _input()
        ModelInput(
            timestamp=data.timestamp,
            step_index=data.step_index,
            task_name=data.task_name,
            spec=data.spec,
            state_history=data.state_history,
            human_action_history=data.human_action_history,
            executed_action_history=data.executed_action_history,
            history_timestamps=data.history_timestamps,
            human_action_timestamps=data.human_action_timestamps,
            human_action_age_ms=data.human_action_age_ms,
            history_valid=data.history_valid,
            human_action_valid=data.human_action_valid,
            human_control_active=data.human_control_active,
            rgb=data.rgb,
            depth={},
        )


def test_builtin_policy_loader() -> None:
    assert isinstance(load_policy("human_passthrough"), HumanPassthroughPolicy)
