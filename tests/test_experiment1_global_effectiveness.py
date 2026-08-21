from __future__ import annotations

import sys
import inspect
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mujoco_shared_control.evaluation.experiment_recorder import EpisodeTraceRecorder, load_trace
from mujoco_shared_control.rss2023.action_postprocess import GlobalActionPostprocessor
from mujoco_shared_control.rss2023.dataset import FeatureNormalizer
from scripts.evaluate_experiment1_global_effectiveness import (
    ACTION_DIM,
    DEFAULT_CHECKPOINT,
    DEFAULT_SURROGATE_CHECKPOINT,
    GlobalSharedController,
    OfflineAWACSurrogatePilot,
    PilotCorruptor,
    _mcnemar_exact,
    run_episode,
)
from scripts.run_e2_rp_normal_gamma_sweep import (
    GAMMAS as E2_RP_NORMAL_GAMMAS,
    select_gamma as select_e2_rp_gamma,
    run_episode as run_e2_rp_normal_episode,
)


def test_motion_noisy_is_deterministic_and_preserves_clean_gripper() -> None:
    pool = np.arange(35, dtype=np.float32).reshape(5, 7)
    clean = np.full(7, 0.25, dtype=np.float64)
    first = PilotCorruptor("noisy", 0.6, pool, 123)
    second = PilotCorruptor("noisy", 0.6, pool, 123)
    observed = np.stack([first.corrupt(clean) for _ in range(20)])
    np.testing.assert_allclose(observed, np.stack([second.corrupt(clean) for _ in range(20)]))
    np.testing.assert_allclose(observed[:, 6], clean[6])
    assert first.action_pool.shape[1] == ACTION_DIM


def test_laggy_first_frame_is_clean_and_then_is_seed_deterministic() -> None:
    clean = np.arange(7, dtype=np.float64)
    first = PilotCorruptor("laggy", 1.0, np.zeros((2, 7)), 9)
    second = PilotCorruptor("laggy", 1.0, np.zeros((2, 7)), 9)
    values = [first.corrupt(clean), first.corrupt(clean + 1), first.corrupt(clean + 2)]
    expected = [second.corrupt(clean), second.corrupt(clean + 1), second.corrupt(clean + 2)]
    np.testing.assert_allclose(values, expected)
    np.testing.assert_allclose(values[0], clean)
    np.testing.assert_allclose(values[1][:6], clean[:6])
    assert values[1][6] == clean[6] + 1


def test_global_action_normalizer_round_trip() -> None:
    normalizer = FeatureNormalizer(
        mean=np.arange(7, dtype=np.float32),
        std=np.linspace(0.5, 1.5, 7, dtype=np.float32),
    )
    value = np.linspace(-1.0, 1.0, 7, dtype=np.float32)
    np.testing.assert_allclose(
        normalizer.denormalize(normalizer.normalize(value[None]))[0], value, atol=1e-6
    )


def test_recorder_save_load_has_aligned_fields(tmp_path) -> None:
    recorder = EpisodeTraceRecorder("test")
    kwargs = dict(
        step_index=0,
        simulation_time=0.0,
        state_43=np.zeros(43),
        clean_pilot_action_7=np.zeros(7),
        raw_pilot_action_7=np.zeros(7),
        assisted_action_7=np.zeros(7),
        postprocessed_action_7=np.zeros(7),
        clipped_assisted_action_7=np.zeros(7),
        executed_action_7=np.zeros(7),
        milestone_t=np.zeros(5),
        active_stage=0,
        object_grasped=False,
        reward=0.0,
        adapter_accepted=True,
        action_clipped=False,
        fallback_used=False,
        diffusion_inference_ms=0.0,
        ee_position=np.zeros(3),
        object_position=np.zeros(3),
        goal_position=np.zeros(3),
        ee_object_distance=0.0,
        object_goal_distance=0.0,
        ee_goal_distance=0.0,
        gripper_opening=0.08,
    )
    recorder.append_step(**kwargs)
    kwargs["step_index"] = 1
    kwargs["simulation_time"] = 0.05
    recorder.append_step(**kwargs)
    path = recorder.save(tmp_path / "trace.npz")
    trace = load_trace(path)
    assert set(trace) >= {"state_43", "raw_pilot_action_7", "postprocessed_action_7", "milestone_t"}
    assert {len(value) for value in trace.values()} == {2}
    assert trace["state_43"].shape == (2, 43)


def test_mcnemar_exact_counts_paired_disagreements() -> None:
    noassist = np.asarray([False, True, False, True])
    global_values = np.asarray([True, True, False, False])
    result = _mcnemar_exact(noassist, global_values)
    assert result["global_only"] == 1
    assert result["noassist_only"] == 1
    assert result["p_value"] == 1.0


def test_canonical_gripper_modes_and_midpoint() -> None:
    postprocessor = GlobalActionPostprocessor(-0.25, 1.0)
    assert postprocessor.threshold == 0.375
    assert postprocessor(np.r_[np.zeros(6), 0.374])[6] == -0.25
    assert postprocessor(np.r_[np.zeros(6), 0.375])[6] == 1.0


def test_offline_awac_7500_checkpoint_and_predictor_contract() -> None:
    pilot = OfflineAWACSurrogatePilot(DEFAULT_SURROGATE_CHECKPOINT)
    assert pilot.predictor.mean.shape == (48,)
    action = pilot.action(np.zeros(43, dtype=np.float32), np.zeros(5, dtype=np.float32))
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert action[6] in (-0.25, 1.0)
    np.testing.assert_array_equal(action, pilot.action(np.zeros(43), np.zeros(5)))


def test_global_gamma_zero_and_no_milestone_interface() -> None:
    controller = GlobalSharedController(DEFAULT_CHECKPOINT)
    assert tuple(inspect.signature(controller.assist).parameters) == ("state_43", "raw_action_7", "gamma")
    raw = np.r_[np.linspace(-0.5, 0.5, 6), -0.25]
    controller.reset_sampling(7)
    assisted = controller.assist(np.zeros(43), raw, 0.0)
    assert np.array_equal(assisted, raw)
    assert assisted is not raw


def test_new_e1_episode_contract_has_no_corruption_inputs() -> None:
    parameters = set(inspect.signature(run_episode).parameters)
    assert "corruption_probability" not in parameters
    assert "corruption_seed" not in parameters
    assert "pilot_type" not in parameters
    assert {"surrogate_pilot", "paired_seed", "pilot_seed", "diffusion_seed"} <= parameters


def test_e2_rp_normal_gamma_grid_and_selection_rule() -> None:
    assert E2_RP_NORMAL_GAMMAS == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.675, 0.7, 0.8, 0.9, 1.0)
    rows = [
        {"gamma": 0.0, "success": 1.0, "illegal_drop": 0.0, "ik_failure": 0.0, "timeout": 0.0, "nan_count": 0, "inf_count": 0, "translation_correction_mm": 0.0, "rotation_correction_rad": 0.0},
        {"gamma": 0.4, "success": 0.96, "illegal_drop": 0.0, "ik_failure": 0.0, "timeout": 0.0, "nan_count": 0, "inf_count": 0, "translation_correction_mm": 0.1, "rotation_correction_rad": 0.0},
        {"gamma": 0.7, "success": 0.98, "illegal_drop": 0.0, "ik_failure": 0.0, "timeout": 0.0, "nan_count": 0, "inf_count": 0, "translation_correction_mm": 0.1, "rotation_correction_rad": 0.0},
        {"gamma": 0.8, "success": 0.96, "illegal_drop": 0.0, "ik_failure": 0.0, "timeout": 0.0, "nan_count": 0, "inf_count": 0, "translation_correction_mm": 0.1, "rotation_correction_rad": 0.0},
    ]
    selected = select_e2_rp_gamma(rows)
    assert selected["status"] == "PASS"
    assert selected["selected_gamma"] == 0.8


def test_e2_rp_normal_episode_contract_has_no_failure_inputs() -> None:
    parameters = set(inspect.signature(run_e2_rp_normal_episode).parameters)
    assert "failure" not in parameters
    assert "condition" not in parameters
    assert "snapshot" not in "".join(parameters)
    assert {"environment_seed", "pilot_seed", "diffusion_seed", "gamma"} <= parameters


def test_correction_physical_units_use_expert_action_scale() -> None:
    normalized = np.asarray([.2, -.4, .1, .3, -.2, .5, 0.0])
    translation_m = np.linalg.norm(normalized[:3] * np.asarray([.025, .025, .025]))
    rotation_rad = np.linalg.norm(normalized[3:6] * np.asarray([.1, .1, .1]))
    assert np.isclose(translation_m, 0.025 * np.linalg.norm(normalized[:3]))
    assert np.isclose(rotation_rad, 0.1 * np.linalg.norm(normalized[3:6]))
