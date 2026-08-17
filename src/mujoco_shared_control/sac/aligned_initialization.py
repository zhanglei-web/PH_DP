"""Offline aligned Actor-Critic initialization artifact construction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor
from mujoco_shared_control.sac.critic import TwinSACCritic


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_aligned_payload(actor_path: Path, critic_path: Path,
                          manifest_content_sha: str) -> dict[str, Any]:
    actor_source = torch.load(actor_path, map_location="cpu", weights_only=False)
    critic_source = torch.load(critic_path, map_location="cpu", weights_only=False)
    if actor_source["format_version"] != "sac_constrained_actor_v2_full_mean_path_distilled":
        raise ValueError("aligned v1 requires constrained Actor v2")
    if critic_source["format_version"] != "sac_critic_pretrain_v1_mc":
        raise ValueError("aligned v1 requires MC-pretrained Critic v1")
    actor_sha = file_sha256(actor_path)
    critic_sha = file_sha256(critic_path)
    if critic_source["actor_sha256"] != actor_sha:
        raise ValueError("Critic was not pretrained against this Actor reference")
    actor_mean = torch.as_tensor(actor_source["observation_mean"])
    actor_std = torch.as_tensor(actor_source["observation_std"])
    critic_mean = torch.as_tensor(critic_source["observation_mean"])
    critic_std = torch.as_tensor(critic_source["observation_std"])
    if not torch.equal(actor_mean, critic_mean) or not torch.equal(actor_std, critic_std):
        raise ValueError("Actor/Critic observation normalization mismatch")
    critics = TwinSACCritic()
    critics.load_state_dict(critic_source["critic_state_dict"])
    targets = TwinSACCritic()
    targets.load_state_dict(critics.state_dict())
    targets.requires_grad_(False)
    return {
        "format_version":"aligned_actor_critic_v1",
        "actor_state_dict":actor_source["actor_state_dict"],
        "critic_state_dict":critics.state_dict(),
        "target_critic_state_dict":targets.state_dict(),
        "observation_mean": actor_mean, "observation_std": actor_std,
        "observation_spec":{"name":"policy_state_42","dimension":42,"dtype":"float32"},
        "action_spec":actor_source["action_spec"],
        "action_semantics":"native constrained B3 x B3 x [-1,1] policy action",
        "reward_version": "sac_reward_v1", "gamma": 0.995,
        "manifest_content_sha":manifest_content_sha,
        "actor_source":str(actor_path.resolve()),"actor_source_sha256":actor_sha,
        "critic_source":str(critic_path.resolve()),"critic_source_sha256":critic_sha,
        "critic_training_objective":"twin Monte-Carlo return regression",
        "online_state":None,"optimizer_state":None,"replay":None,
    }


def load_aligned(
    path: Path,
) -> tuple[SACConstrainedGaussianActor, TwinSACCritic, TwinSACCritic, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format_version"] != "aligned_actor_critic_v1":
        raise ValueError("unsupported aligned artifact")
    actor = SACConstrainedGaussianActor()
    actor.load_state_dict(payload["actor_state_dict"])
    critics = TwinSACCritic()
    critics.load_state_dict(payload["critic_state_dict"])
    targets = TwinSACCritic()
    targets.load_state_dict(payload["target_critic_state_dict"])
    targets.requires_grad_(False)
    return actor, critics, targets, payload


def load_aligned_v2(
    path: Path,
) -> tuple[SACConstrainedGaussianActor, TwinSACCritic, TwinSACCritic, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format_version"] != "aligned_actor_critic_v2":
        raise ValueError("unsupported aligned v2 artifact")
    if payload["reward_version"] != "sac_reward_v2_candidate":
        raise ValueError("aligned v2 reward mismatch")
    actor = SACConstrainedGaussianActor(); actor.load_state_dict(payload["actor_state_dict"])
    critics = TwinSACCritic(); critics.load_state_dict(payload["critic_state_dict"])
    targets = TwinSACCritic(); targets.load_state_dict(payload["target_critic_state_dict"])
    targets.requires_grad_(False)
    return actor, critics, targets, payload
