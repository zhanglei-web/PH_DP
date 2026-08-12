"""Built-in policies and dynamic plugin loading."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from mujoco_shared_control.shared_control.interfaces import (
    CommandSpace,
    ModelInput,
    ModelInputSpec,
    ModelOutput,
    SharedPolicy,
)


class HumanPassthroughPolicy:
    """Reference plugin: preserve the existing human Cartesian command."""

    input_spec = ModelInputSpec(
        history_length=1,
        use_state_26=False,
        use_human_action=True,
        use_executed_action=False,
        use_rgb=False,
        use_depth=False,
    )

    def __init__(self, **_: Any) -> None:
        self.task_name = ""

    def reset(self, task_name: str) -> None:
        self.task_name = task_name

    def predict(self, model_input: ModelInput) -> ModelOutput:
        valid = bool(model_input.human_action_valid[-1])
        command = model_input.latest_human_action.astype(np.float64, copy=True)
        if not valid:
            command[:] = np.nan
        return ModelOutput(
            timestamp=model_input.timestamp,
            command=command,
            command_space=CommandSpace.CARTESIAN_POSE,
            valid=valid,
            control_active=bool(model_input.human_control_active[-1]),
            confidence=1.0 if valid else 0.0,
            policy_name=type(self).__name__,
            diagnostics={"mode": "human_passthrough"},
        )


BUILTIN_POLICIES = {
    "human_passthrough": HumanPassthroughPolicy,
}


def load_policy(specification: str, config: dict[str, Any] | None = None) -> SharedPolicy:
    """Load a built-in alias or ``module:ClassName`` policy plugin."""
    config = dict(config or {})
    if specification in BUILTIN_POLICIES:
        policy_type = BUILTIN_POLICIES[specification]
    else:
        module_name, separator, class_name = specification.partition(":")
        if not separator or not module_name or not class_name:
            raise ValueError(
                "policy_plugin must be a built-in alias or 'module:ClassName'"
            )
        module = importlib.import_module(module_name)
        policy_type = getattr(module, class_name)
    policy = policy_type(**config)
    if not isinstance(policy, SharedPolicy):
        raise TypeError(
            f"{specification!r} must implement reset(task_name) and predict(input)"
        )
    return policy
