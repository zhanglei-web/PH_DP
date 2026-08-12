"""Offline RSS 2023 surrogate-pilot evaluation on held-out demonstrations."""

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

from mujoco_shared_control.rss2023.dataset import ACTION_DIM, load_episode
from mujoco_shared_control.rss2023.inference import RSS2023Predictor


SURROGATE_PILOTS = ("clean", "noisy", "laggy")


def corrupt_action_sequence(
    expert_actions: NDArray[np.floating],
    *,
    pilot: str,
    probability: float,
    random_action_pool: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Create the Noisy/Laggy surrogate pilots defined in the RSS 2023 paper."""
    expert = np.asarray(expert_actions, dtype=np.float64)
    pool = np.asarray(random_action_pool, dtype=np.float64)
    if expert.ndim != 2 or expert.shape[1] != ACTION_DIM:
        raise ValueError("expert_actions must have shape (N, 8)")
    if pool.ndim != 2 or pool.shape[1] != ACTION_DIM or pool.shape[0] == 0:
        raise ValueError("random_action_pool must have shape (M, 8), M > 0")
    if pilot not in SURROGATE_PILOTS:
        raise ValueError(f"pilot must be one of {SURROGATE_PILOTS}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if pilot == "clean":
        return expert.copy()
    corrupted = expert.copy()
    if pilot == "noisy":
        mask = rng.random(expert.shape[0]) < probability
        indices = rng.integers(0, pool.shape[0], size=int(mask.sum()))
        corrupted[mask] = pool[indices]
        return corrupted
    for index in range(1, expert.shape[0]):
        if rng.random() < probability:
            corrupted[index] = corrupted[index - 1]
    return corrupted


def quaternion_error_degrees(
    predicted: NDArray[np.floating], reference: NDArray[np.floating]
) -> NDArray[np.float64]:
    first = np.asarray(predicted, dtype=np.float64)
    second = np.asarray(reference, dtype=np.float64)
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    dot = np.abs(np.sum(first * second, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def align_action_quaternion_sign(
    actions: NDArray[np.floating], reference: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Choose the quaternion sign closest to a reference before vector metrics."""
    aligned = np.asarray(actions, dtype=np.float64).copy()
    target = np.asarray(reference, dtype=np.float64)
    negative = np.sum(aligned[..., 3:7] * target[..., 3:7], axis=-1) < 0.0
    aligned[negative, 3:7] *= -1.0
    return aligned


@dataclass(frozen=True)
class OfflineEpisodeResult:
    episode: str
    pilot: str
    probability: float
    gamma: float
    seed: int
    frames: int
    position_before_m: float
    position_after_m: float
    orientation_before_deg: float
    orientation_after_deg: float
    gripper_before_m: float
    gripper_after_m: float
    normalized_before: float
    normalized_after: float
    improved_fraction: float
    intervention_normalized: float
    inference_mean_ms: float


def _mean_errors(
    predicted: NDArray[np.float64],
    reference: NDArray[np.float64],
    action_std: NDArray[np.float32],
) -> tuple[float, float, float, NDArray[np.float64]]:
    predicted = align_action_quaternion_sign(predicted, reference)
    position = np.linalg.norm(predicted[:, :3] - reference[:, :3], axis=1)
    orientation = quaternion_error_degrees(predicted[:, 3:7], reference[:, 3:7])
    gripper = np.abs(predicted[:, 7] - reference[:, 7])
    normalized = np.linalg.norm((predicted - reference) / action_std, axis=1)
    return (
        float(position.mean()),
        float(orientation.mean()),
        float(gripper.mean()),
        normalized,
    )


def _checkpoint_paths(
    checkpoint: dict, split: str
) -> tuple[Path, ...]:
    key = f"{split}_files"
    if key not in checkpoint["dataset_manifest"]:
        raise ValueError(f"checkpoint does not contain split {split!r}")
    return tuple(Path(path) for path in checkpoint["dataset_manifest"][key])


def _training_action_pool(checkpoint: dict) -> NDArray[np.float64]:
    paths = _checkpoint_paths(checkpoint, "train")
    return np.concatenate([load_episode(path).action for path in paths], axis=0).astype(
        np.float64
    )


def evaluate_offline(
    checkpoint_path: str | Path,
    *,
    split: str,
    pilots: Iterable[str],
    probabilities: Iterable[float],
    gammas: Iterable[float],
    seeds: Iterable[int],
    device_name: str = "auto",
    use_ema: bool = False,
) -> tuple[list[OfflineEpisodeResult], float]:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    predictor = RSS2023Predictor.from_checkpoint(
        checkpoint_file, device_name=device_name, use_ema=use_ema
    )
    episodes = [load_episode(path) for path in _checkpoint_paths(checkpoint, split)]
    action_pool = _training_action_pool(checkpoint)
    action_std = np.asarray(
        checkpoint["action_normalizer"]["std"], dtype=np.float32
    )
    requested_pilots = tuple(pilots)
    requested_probabilities = tuple(float(value) for value in probabilities)
    requested_gammas = tuple(float(value) for value in gammas)
    requested_seeds = tuple(int(value) for value in seeds)
    results: list[OfflineEpisodeResult] = []

    for episode_index, episode in enumerate(episodes):
        expert = episode.action.astype(np.float64)
        for pilot in requested_pilots:
            pilot_probabilities = (0.0,) if pilot == "clean" else requested_probabilities
            for probability in pilot_probabilities:
                for seed in requested_seeds:
                    rng = np.random.default_rng(seed + episode_index * 100_003)
                    corrupted = corrupt_action_sequence(
                        expert,
                        pilot=pilot,
                        probability=probability,
                        random_action_pool=action_pool,
                        rng=rng,
                    )
                    before = _mean_errors(corrupted, expert, action_std)
                    for gamma_index, gamma in enumerate(requested_gammas):
                        started = time.perf_counter()
                        assisted = predictor.predict_batch(
                            episode.observation,
                            corrupted,
                            gamma=gamma,
                            seed=seed * 10_000 + episode_index * 100 + gamma_index,
                        )
                        elapsed_ms = (time.perf_counter() - started) * 1_000.0
                        after = _mean_errors(assisted, expert, action_std)
                        aligned_assisted = align_action_quaternion_sign(
                            assisted, corrupted
                        )
                        intervention = np.linalg.norm(
                            (aligned_assisted - corrupted) / action_std, axis=1
                        )
                        results.append(
                            OfflineEpisodeResult(
                                episode=episode.path.name,
                                pilot=pilot,
                                probability=probability,
                                gamma=gamma,
                                seed=seed,
                                frames=expert.shape[0],
                                position_before_m=before[0],
                                position_after_m=after[0],
                                orientation_before_deg=before[1],
                                orientation_after_deg=after[1],
                                gripper_before_m=before[2],
                                gripper_after_m=after[2],
                                normalized_before=float(before[3].mean()),
                                normalized_after=float(after[3].mean()),
                                improved_fraction=float((after[3] < before[3]).mean()),
                                intervention_normalized=float(intervention.mean()),
                                inference_mean_ms=elapsed_ms / expert.shape[0],
                            )
                        )

    candidates = {
        gamma: np.mean(
            [
                result.normalized_after
                for result in results
                if result.pilot != "clean" and result.gamma == gamma
            ]
        )
        for gamma in requested_gammas
    }
    usable_candidates = {
        gamma: value for gamma, value in candidates.items() if np.isfinite(value)
    }
    if not usable_candidates:
        raise ValueError(
            "offline gamma selection requires at least one noisy or laggy pilot"
        )
    offline_best_gamma = float(min(usable_candidates, key=usable_candidates.get))
    return results, offline_best_gamma


def summarize_offline(results: Iterable[OfflineEpisodeResult]) -> list[dict[str, float | str]]:
    rows = list(results)
    groups = sorted({(row.pilot, row.probability, row.gamma) for row in rows})
    metrics = (
        "position_before_m",
        "position_after_m",
        "orientation_before_deg",
        "orientation_after_deg",
        "gripper_before_m",
        "gripper_after_m",
        "normalized_before",
        "normalized_after",
        "improved_fraction",
        "intervention_normalized",
        "inference_mean_ms",
    )
    summary: list[dict[str, float | str]] = []
    for pilot, probability, gamma in groups:
        selected = [
            row
            for row in rows
            if row.pilot == pilot
            and row.probability == probability
            and row.gamma == gamma
        ]
        entry: dict[str, float | str] = {
            "pilot": pilot,
            "probability": probability,
            "gamma": gamma,
            "episode_seed_runs": float(len(selected)),
        }
        for metric in metrics:
            values = np.asarray([getattr(row, metric) for row in selected])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std())
        summary.append(entry)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline RSS 2023 action correction evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--pilots", nargs="+", choices=SURROGATE_PILOTS, default=["clean", "noisy", "laggy"])
    parser.add_argument("--probabilities", nargs="+", type=float, default=[0.6])
    parser.add_argument("--gammas", nargs="+", type=float, default=[round(x / 10, 1) for x in range(11)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results, offline_best_gamma = evaluate_offline(
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
    summary = summarize_offline(results)
    _write_csv(output / "episode_results.csv", detail)
    _write_csv(output / "summary.csv", summary)
    metadata = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "split": args.split,
        "pilots": args.pilots,
        "probabilities": args.probabilities,
        "gammas": args.gammas,
        "seeds": args.seeds,
        "offline_best_gamma": offline_best_gamma,
        "selection_warning": (
            "offline action error is not a deployment criterion; validate gamma "
            "with closed-loop task success and clean-pilot safety"
        ),
        "weights": args.weights,
        "random_action_definition": "sampled from training demonstration actions",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"offline_best_gamma={offline_best_gamma:.2f}")
    print(f"wrote offline evaluation to {output}")


if __name__ == "__main__":
    main()
