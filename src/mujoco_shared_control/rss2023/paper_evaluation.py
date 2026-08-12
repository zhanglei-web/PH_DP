"""Paper pilot definitions and MuJoCo evaluation for the local task mapping."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray
import torch

from mujoco_shared_control import PickPlaceEnv
from mujoco_shared_control.rss2023.closed_loop import action_pose_matrix
from mujoco_shared_control.rss2023.model import DiffusionConfig
from mujoco_shared_control.rss2023.paper_dataset import (
    PAPER_ACTION_DIM,
    PaperEpisode,
    apply_incremental_action,
    build_paper_observation,
    load_paper_episode,
)
from mujoco_shared_control.rss2023.paper_model import PaperRSS2023Diffusion


PAPER_PILOTS = ("expert", "noisy", "laggy", "zero", "random")
PAPER_PROBABILITY = 0.6
PAPER_GAMMAS = (0.0, 0.2, 1.0)
PAPER_SEEDS = tuple(range(30))
PHYSICS_STEPS_PER_ACTION = 5


class PaperPredictor:
    def __init__(self, model: PaperRSS2023Diffusion, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device_name: str = "auto") -> "PaperPredictor":
        device = torch.device(
            "cuda" if device_name == "auto" and torch.cuda.is_available() else
            "cpu" if device_name == "auto" else device_name
        )
        checkpoint = torch.load(Path(path).resolve(), map_location=device, weights_only=False)
        model = PaperRSS2023Diffusion(DiffusionConfig(**checkpoint["diffusion_config"]))
        model.load_state_dict(checkpoint["model"])
        return cls(model, device)

    @torch.no_grad()
    def predict(
        self, observation: NDArray[np.floating], action: NDArray[np.floating],
        *, gamma: float, seed: int,
    ) -> NDArray[np.float64]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.model.assist(obs, act, gamma, generator=generator).cpu().numpy().astype(np.float64)


class Pilot:
    """Definitions copied from the released actor/base.py behavior."""

    def __init__(self, name: str, probability: float, seed: int) -> None:
        if name not in PAPER_PILOTS:
            raise ValueError(f"unknown paper pilot {name}")
        self.name = name
        self.probability = probability
        self.rng = np.random.default_rng(seed)
        self.previous = self.rng.uniform(-1.0, 1.0, PAPER_ACTION_DIM)

    def action(self, expert: NDArray[np.floating]) -> NDArray[np.float64]:
        expert_action = np.asarray(expert, dtype=np.float64)
        if self.name == "expert":
            return expert_action.copy()
        if self.name == "zero":
            return np.zeros(PAPER_ACTION_DIM, dtype=np.float64)
        if self.name == "random":
            return self.rng.uniform(-1.0, 1.0, PAPER_ACTION_DIM)
        if self.name == "noisy":
            if self.rng.random() < self.probability:
                return self.rng.uniform(-1.0, 1.0, PAPER_ACTION_DIM)
            return expert_action.copy()
        if self.rng.random() < self.probability:
            return self.previous.copy()
        self.previous = expert_action.copy()
        return self.previous.copy()


@dataclass(frozen=True)
class PaperRollout:
    seed: int
    episode: str
    pilot: str
    gamma: float
    probability: float
    correct_goal: bool
    wrong_goal: bool
    timeout: bool
    episode_length: int
    final_distance_m: float
    ik_failures: int
    action_difference_sum: float
    inference_mean_ms: float


def run_rollout(
    env: PickPlaceEnv, predictor: PaperPredictor, episode: PaperEpisode,
    pilot: Pilot, *, gamma: float, seed: int,
) -> PaperRollout:
    observation, _ = env.reset(
        seed=seed,
        options={
            "randomize_object": False, "randomize_goal": False,
            "object_xy": episode.initial_object_xy,
            "goal_xy": episode.initial_goal_xy,
        },
    )
    # The recorded expert increments are expressed relative to the first VR
    # target in this episode, analogous to the environment's initialized
    # effector target in the released Block Pushing task.
    target = episode.expert_targets[0].astype(np.float64).copy()
    q_command = env.home_joint_positions.copy()
    gripper_command = float(target[7])
    ik_failures = 0
    difference_sum = 0.0
    inference_seconds = 0.0
    inference_calls = 0
    success = False
    final_info: dict = {}
    length = 0
    for index, expert_action in enumerate(episode.action):
        human_action = pilot.action(expert_action)
        shared_action = human_action
        if gamma > 0.0:
            state = build_paper_observation(observation, target)
            started = time.perf_counter()
            shared_action = predictor.predict(
                state, human_action, gamma=gamma,
                seed=seed * 1_000_003 + index,
            )
            inference_seconds += time.perf_counter() - started
            inference_calls += 1
        difference_sum += float(np.linalg.norm(shared_action - human_action))
        proposed_target = apply_incremental_action(target, shared_action)
        try:
            ik = env.ik_controller.inverse_kinematics(
                action_pose_matrix(proposed_target), initial_guess=q_command
            )
            if not ik.converged:
                raise ValueError("IK did not converge")
            q_command = ik.joint_positions
            gripper_command = float(proposed_target[7])
            target = proposed_target
        except ValueError:
            ik_failures += 1
        joint_action = np.concatenate((q_command, [gripper_command]))
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            observation, _, success, _, final_info = env.step(joint_action)
            if success:
                break
        length = index + 1
        if success:
            break
    if not final_info:
        _, _, final_info = env.task.evaluate(observation)
    return PaperRollout(
        seed=seed, episode=episode.path.name, pilot=pilot.name,
        gamma=gamma, probability=pilot.probability,
        correct_goal=bool(success), wrong_goal=False, timeout=not bool(success),
        episode_length=length,
        final_distance_m=float(final_info["object_goal_distance"]),
        ik_failures=ik_failures, action_difference_sum=difference_sum,
        inference_mean_ms=(1000.0 * inference_seconds / inference_calls) if inference_calls else 0.0,
    )


def summarize(rows: list[PaperRollout]) -> list[dict[str, float | str]]:
    groups = sorted({(row.pilot, row.gamma) for row in rows})
    result: list[dict[str, float | str]] = []
    for pilot, gamma in groups:
        selected = [row for row in rows if row.pilot == pilot and row.gamma == gamma]
        result.append({
            "pilot": pilot, "gamma": gamma, "rollouts": float(len(selected)),
            "correct_goal_rate": float(np.mean([row.correct_goal for row in selected])),
            "wrong_goal_rate": 0.0,
            "timeout_rate": float(np.mean([row.timeout for row in selected])),
            "episode_length_mean": float(np.mean([row.episode_length for row in selected])),
            "final_distance_m_mean": float(np.mean([row.final_distance_m for row in selected])),
            "ik_failures_mean": float(np.mean([row.ik_failures for row in selected])),
            "action_difference_sum_mean": float(np.mean([row.action_difference_sum for row in selected])),
            "inference_mean_ms": float(np.mean([row.inference_mean_ms for row in selected])),
        })
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RSS2023 paper evaluation protocol")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument("--gammas", nargs="+", type=float, default=list(PAPER_GAMMAS))
    parser.add_argument("--pilots", nargs="+", choices=PAPER_PILOTS, default=list(PAPER_PILOTS))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    episodes = tuple(
        load_paper_episode(path) for path in checkpoint["dataset_manifest"]["evaluation_files"]
    )
    if len(episodes) != 10:
        raise ValueError("paper protocol requires exactly 10 evaluation episodes")
    predictor = PaperPredictor.from_checkpoint(args.checkpoint, args.device)
    env = PickPlaceEnv(control_timestep=0.01, max_episode_steps=2_000_000_000, enable_camera=False)
    rows: list[PaperRollout] = []
    try:
        for seed in args.seeds:
            for pilot_name in args.pilots:
                for gamma in args.gammas:
                    # The released actor persists its Laggy state across the
                    # ten episodes in one evaluation run, but each gamma is a
                    # separate run with the same actor seed.
                    pilot = Pilot(pilot_name, PAPER_PROBABILITY, seed=seed)
                    for episode_index, episode in enumerate(episodes):
                        row = run_rollout(
                            env, predictor, episode, pilot, gamma=gamma,
                            seed=seed * 10_000 + episode_index,
                        )
                        rows.append(row)
                    print(
                        f"seed={seed} pilot={pilot_name} gamma={gamma:.1f} "
                        "episodes=10/10", flush=True
                    )
    finally:
        env.close()
    output = args.output_dir.resolve()
    detail = [asdict(row) for row in rows]
    summary = summarize(rows)
    _write_csv(output / "episode_results.csv", detail)
    _write_csv(output / "summary.csv", summary)
    metadata = {
        "paper_reference": "RSS2023 Block Pushing Table II/VI",
        "pilots": args.pilots, "probability": PAPER_PROBABILITY,
        "gammas": args.gammas, "seeds": args.seeds,
        "episodes_per_seed": 10,
        "paper_exact": {
            "pilot_definitions": True, "gamma_and_probability": True,
            "diffusion_training_and_sampling": True,
        },
        "task_mapping": {
            "environment": "local MuJoCo Franka pick-and-place",
            "action": "7D bounded incremental 3D pose/gripper control",
            "wrong_goal": "not applicable: local task has one goal",
            "expert": "held-out recorded demonstration rather than SAC policy",
            "timeout": "held-out demonstration command length",
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote paper-protocol evaluation to {output}")


if __name__ == "__main__":
    main()
