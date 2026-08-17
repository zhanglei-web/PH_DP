from pathlib import Path

import json
import torch

from mujoco_shared_control.sac.aligned_initialization import (
    build_aligned_payload,
    load_aligned,
    load_aligned_v2,
)
from mujoco_shared_control.sac.constrained_actor import SACConstrainedGaussianActor


ACTOR=Path("outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt")
CRITIC=Path("outputs/sac_critic/sac_critic_pretrain_v1_20260813T210000Z/critic_pretrained_best.pt")


def test_actor_critic_observation_action_and_target_alignment() -> None:
    manifest=json.load(open("manifests/rule_expert_v1_formal.json"))
    payload=build_aligned_payload(ACTOR,CRITIC,manifest["content_sha256"])
    assert payload["observation_spec"]=={"name":"policy_state_42","dimension":42,"dtype":"float32"}
    actor=SACConstrainedGaussianActor();actor.load_state_dict(payload["actor_state_dict"])
    state=torch.zeros(16,42);normalized=(state-payload["observation_mean"])/payload["observation_std"]
    action=actor.deterministic_action(normalized)
    assert torch.all(torch.linalg.vector_norm(action[:,:3],dim=-1)<=1)
    assert torch.all(torch.linalg.vector_norm(action[:,3:6],dim=-1)<=1)
    assert torch.all(action[:,6].abs()<=1)
    online=payload["critic_state_dict"];target=payload["target_critic_state_dict"]
    assert online.keys()==target.keys()
    for key in online: torch.testing.assert_close(online[key],target[key],rtol=0,atol=0)
    assert payload["replay"] is None and payload["optimizer_state"] is None


def test_aligned_checkpoint_round_trip_and_sources_are_frozen(tmp_path: Path) -> None:
    manifest = json.loads(Path("manifests/rule_expert_v1_formal.json").read_text())
    actor_before = ACTOR.read_bytes()
    critic_before = CRITIC.read_bytes()
    destination = tmp_path / "aligned.pt"
    torch.save(build_aligned_payload(ACTOR, CRITIC, manifest["content_sha256"]), destination)
    actor, critics, targets, payload = load_aligned(destination)
    assert payload["reward_version"] == "sac_reward_v1" and payload["gamma"] == .995
    assert ACTOR.read_bytes() == actor_before and CRITIC.read_bytes() == critic_before
    assert not any(parameter.requires_grad for parameter in targets.parameters())
    for key, value in critics.state_dict().items():
        torch.testing.assert_close(value, targets.state_dict()[key], rtol=0, atol=0)
    state = torch.randn(4, 42)
    normalized = (state - payload["observation_mean"]) / payload["observation_std"]
    with torch.no_grad():
        action = actor.deterministic_action(normalized)
        q1, q2 = critics(normalized, action)
    assert q1.shape == q2.shape == (4, 1)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()


def test_frozen_reward_regression_is_exact() -> None:
    summary = json.loads(Path(
        "outputs/reward_validation/sac_reward_v1_regression_20260812T150245Z/summary.json"
    ).read_text())
    regression = summary["official_regression"]
    assert regression["reward_max_abs_difference"] < 2e-15
    assert regression["terminal_decision_mismatch_count"] == 0
    assert regression["phase_mismatch_count"] == 0
    assert regression["illegal_drop_mismatch_count"] == 0


def test_aligned_v2_checkpoint_reload_target_copy_and_no_online_state() -> None:
    path = Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")
    actor, critics, targets, payload = load_aligned_v2(path)
    assert payload["replay"] is payload["optimizer_state"] is payload["online_state"] is None
    assert payload["reward_version"] == "sac_reward_v2_candidate"
    for key, value in critics.state_dict().items():
        torch.testing.assert_close(value, targets.state_dict()[key], rtol=0, atol=0)
    normalized = torch.zeros(3, 42)
    with torch.no_grad():
        action = actor.deterministic_action(normalized)
        q1, q2 = critics(normalized, action)
    assert q1.shape == q2.shape == (3, 1)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
