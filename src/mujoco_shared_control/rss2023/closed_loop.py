"""MuJoCo closed-loop evaluation using recorded demonstrations as surrogate pilots."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
import torch

from mujoco_shared_control import PickPlaceEnv
from mujoco_shared_control.rss2023.dataset import (
    EpisodeData,
    build_observation_29,
    load_episode,
)
from mujoco_shared_control.rss2023.evaluation import (
    SURROGATE_PILOTS,
    _checkpoint_paths,
    _training_action_pool,
    align_action_quaternion_sign,
    corrupt_action_sequence,
)
from mujoco_shared_control.rss2023.inference import RSS2023Predictor
from mujoco_shared_control.utils.pose import quaternion_to_matrix


PHYSICS_STEPS_PER_COMMAND = 5
SETTLE_COMMAND_FRAMES = 10


def action_pose_matrix(action: NDArray[np.floating]) -> NDArray[np.float64]:
    command = np.asarray(action, dtype=np.float64)
    if command.shape != (8,) or not np.isfinite(command).all():
        raise ValueError("Cartesian command must have shape (8,) and be finite")
    quaternion = command[3:7]
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-8:
        raise ValueError("Cartesian command quaternion cannot be zero")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_to_matrix(quaternion / norm)
    pose[:3, 3] = command[:3]
    return pose


@dataclass(frozen=True)
class ClosedLoopResult:
    episode: str
    pilot: str
    probability: float
    gamma: float
    seed: int
    success: bool
    success_command_step: int
    command_frames: int
    final_distance_m: float
    minimum_distance_m: float
    ik_failures: int
    fallback_frames: int
    intervention_normalized: float
    inference_mean_ms: float
    wall_time_s: float


def run_closed_loop_episode(
    env: PickPlaceEnv,
    predictor: RSS2023Predictor,
    episode: EpisodeData,
    *,
    pilot: str,
    probability: float,
    gamma: float,
    seed: int,
    random_action_pool: NDArray[np.float64],
    action_std: NDArray[np.float32],
) -> ClosedLoopResult:
    rng = np.random.default_rng(seed)
    human_actions = corrupt_action_sequence(
        episode.action,
        pilot=pilot,
        probability=probability,
        random_action_pool=random_action_pool,
        rng=rng,
    )
    object_xy = episode.observation[0, 15:17].astype(np.float64)
    goal_xy = episode.observation[0, 22:24].astype(np.float64)
    observation, _ = env.reset(
        seed=seed,
        options={
            "randomize_object": False,
            "randomize_goal": False,
            "object_xy": object_xy,
            "goal_xy": goal_xy,
        },
    )
    q_command = env.home_joint_positions
    gripper_command = 0.08
    success = False
    success_step = -1
    ik_failures = 0
    fallback_frames = 0
    inference_elapsed = 0.0
    inference_calls = 0
    interventions: list[float] = []
    _, _, initial_info = env.task.evaluate(observation)
    minimum_distance = float(initial_info["object_goal_distance"])
    started = time.perf_counter()

    for command_index, human_action in enumerate(human_actions):
        command = human_action
        if gamma > 0.0:
            model_observation = build_observation_29(observation)
            inference_started = time.perf_counter()
            command = predictor.predict(
                model_observation,
                human_action,
                gamma=gamma,
                seed=seed * 1_000_003 + command_index,
            )
            inference_elapsed += time.perf_counter() - inference_started
            inference_calls += 1
            interventions.append(
                float(
                    np.linalg.norm(
                        (
                            align_action_quaternion_sign(
                                command[None, :], human_action[None, :]
                            )[0]
                            - human_action
                        )
                        / action_std
                    )
                )
            )
        try:
            result = env.ik_controller.inverse_kinematics(
                action_pose_matrix(command), initial_guess=q_command
            )
            if not result.converged:
                raise ValueError("IK did not converge")
            q_command = result.joint_positions
            gripper_command = float(np.clip(command[7], 0.0, 0.08))
        except ValueError:
            ik_failures += 1
            fallback_frames += 1

        joint_action = np.concatenate((q_command, [gripper_command]))
        for _ in range(PHYSICS_STEPS_PER_COMMAND):
            observation, _, success, _, info = env.step(joint_action)
            minimum_distance = min(
                minimum_distance, float(info["object_goal_distance"])
            )
            if success:
                success_step = command_index
                break
        if success:
            break

    if not success:
        joint_action = np.concatenate((q_command, [gripper_command]))
        for _ in range(SETTLE_COMMAND_FRAMES * PHYSICS_STEPS_PER_COMMAND):
            observation, _, success, _, info = env.step(joint_action)
            minimum_distance = min(
                minimum_distance, float(info["object_goal_distance"])
            )
            if success:
                success_step = len(human_actions)
                break
    _, _, final_info = env.task.evaluate(observation)
    wall_time = time.perf_counter() - started
    return ClosedLoopResult(
        episode=episode.path.name,
        pilot=pilot,
        probability=probability,
        gamma=gamma,
        seed=seed,
        success=bool(success),
        success_command_step=success_step,
        command_frames=len(human_actions),
        final_distance_m=float(final_info["object_goal_distance"]),
        minimum_distance_m=minimum_distance,
        ik_failures=ik_failures,
        fallback_frames=fallback_frames,
        intervention_normalized=float(np.mean(interventions)) if interventions else 0.0,
        inference_mean_ms=(inference_elapsed * 1_000.0 / inference_calls)
        if inference_calls
        else 0.0,
        wall_time_s=wall_time,
    )


def evaluate_closed_loop(
    checkpoint_path: str | Path,
    *,
    split: str,
    pilots: Iterable[str],
    probabilities: Iterable[float],
    gammas: Iterable[float],
    seeds: Iterable[int],
    device_name: str = "auto",
    use_ema: bool = False,
) -> list[ClosedLoopResult]:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    predictor = RSS2023Predictor.from_checkpoint(
        checkpoint_file, device_name=device_name, use_ema=use_ema
    )
    episodes = [load_episode(path) for path in _checkpoint_paths(checkpoint, split)]
    action_pool = _training_action_pool(checkpoint)
    action_std = np.asarray(checkpoint["action_normalizer"]["std"], dtype=np.float32)
    results: list[ClosedLoopResult] = []
    env = PickPlaceEnv(
        control_timestep=0.01,
        max_episode_steps=2_000_000_000,
        enable_camera=False,
    )
    try:
        for episode_index, episode in enumerate(episodes):
            for pilot in tuple(pilots):
                pilot_probabilities = (0.0,) if pilot == "clean" else tuple(probabilities)
                for probability in pilot_probabilities:
                    for gamma in tuple(gammas):
                        for seed in tuple(seeds):
                            result = run_closed_loop_episode(
                                env,
                                predictor,
                                episode,
                                pilot=pilot,
                                probability=float(probability),
                                gamma=float(gamma),
                                seed=int(seed) + episode_index * 100_003,
                                random_action_pool=action_pool,
                                action_std=action_std,
                            )
                            results.append(result)
                            print(
                                f"episode={episode.path.name} pilot={pilot} "
                                f"p={float(probability):.2f} gamma={float(gamma):.2f} "
                                f"seed={seed} success={int(result.success)} "
                                f"distance={result.final_distance_m:.4f} "
                                f"ik_failures={result.ik_failures}"
                            )
    finally:
        env.close()
    return results


def summarize_closed_loop(results: Iterable[ClosedLoopResult]) -> list[dict[str, float | str]]:
    rows = list(results)
    groups = sorted({(row.pilot, row.probability, row.gamma) for row in rows})
    summary: list[dict[str, float | str]] = []
    for pilot, probability, gamma in groups:
        selected = [
            row
            for row in rows
            if row.pilot == pilot
            and row.probability == probability
            and row.gamma == gamma
        ]
        successful_steps = [row.success_command_step for row in selected if row.success]
        summary.append(
            {
                "pilot": pilot,
                "probability": probability,
                "gamma": gamma,
                "rollouts": float(len(selected)),
                "success_rate": float(np.mean([row.success for row in selected])),
                "success_step_mean": float(np.mean(successful_steps))
                if successful_steps
                else -1.0,
                "final_distance_m_mean": float(
                    np.mean([row.final_distance_m for row in selected])
                ),
                "minimum_distance_m_mean": float(
                    np.mean([row.minimum_distance_m for row in selected])
                ),
                "ik_failures_mean": float(
                    np.mean([row.ik_failures for row in selected])
                ),
                "fallback_frames_mean": float(
                    np.mean([row.fallback_frames for row in selected])
                ),
                "intervention_normalized_mean": float(
                    np.mean([row.intervention_normalized for row in selected])
                ),
                "inference_mean_ms": float(
                    np.mean([row.inference_mean_ms for row in selected])
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MuJoCo closed-loop RSS 2023 evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--pilots", nargs="+", choices=SURROGATE_PILOTS, default=["clean", "noisy", "laggy"])
    parser.add_argument("--probabilities", nargs="+", type=float, default=[0.6])
    parser.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = evaluate_closed_loop(
        args.checkpoint,
        split=args.split,
        pilots=args.pilots,
        probabilities=args.probabilities,
        gammas=args.gammas,
        seeds=args.seeds,
        device_name=args.device,
        use_ema=args.weights == "ema",
    )
    output = args.output_dir.expanduser().resolve()
    detail = [
        {field.name: getattr(result, field.name) for field in fields(result)}
        for result in results
    ]
    summary = summarize_closed_loop(results)
    _write_csv(output / "episode_results.csv", detail)
    _write_csv(output / "summary.csv", summary)
    metadata = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "split": args.split,
        "pilots": args.pilots,
        "probabilities": args.probabilities,
        "gammas": args.gammas,
        "seeds": args.seeds,
        "protocol": "recorded Cartesian commands replayed from matched object/goal reset",
        "physics_steps_per_command": PHYSICS_STEPS_PER_COMMAND,
        "weights": args.weights,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote closed-loop evaluation to {output}")


if __name__ == "__main__":
    main()
