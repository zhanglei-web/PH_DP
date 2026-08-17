from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2
from mujoco_shared_control.sac.critic_pretraining import build_arrays_from_semantic_run
from mujoco_shared_control.sac.offline_adv_bc import (
    build_advantage_arrays,
    state_checksum,
    train_step,
    validate_trainable_actor,
)


ALIGNED = Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")


def _assets():
    actor, critic, target, payload = load_aligned_v2(ALIGNED)
    arrays, audit = build_arrays_from_semantic_run(
        Path("manifests/rule_expert_v1_formal.json"),
        Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z"),
        reward_version="sac_reward_v2_candidate",
    )
    return actor, critic, target, payload, arrays, audit


def test_advantage_is_q_data_minus_fixed_initial_actor_and_threshold_zero() -> None:
    actor, critic, _target, payload, arrays, _audit = _assets()
    subset = arrays["validation"].subset(np.arange(len(arrays["validation"])) < 64)
    reference_checksum = state_checksum(actor)
    values = build_advantage_arrays(
        subset, actor, critic, payload["observation_mean"], payload["observation_std"]
    )
    np.testing.assert_allclose(values.advantage, values.q_data - values.q_initial_actor)
    np.testing.assert_array_equal(values.positive, values.advantage > 0)
    assert state_checksum(actor) == reference_checksum


def test_train_step_updates_mean_path_only_and_not_critic_or_target() -> None:
    actor, critic, target, payload, arrays, _audit = _assets()
    critic_before, target_before = state_checksum(critic), state_checksum(target)
    log_std_before = {k: v.clone() for k, v in actor.log_std_head.state_dict().items()}
    trainable = validate_trainable_actor(actor)
    assert trainable and not any(name.startswith("log_std_head") for name in trainable)
    optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4
    )
    data = arrays["train"]
    obs = torch.from_numpy(data.observation[:16]); action = torch.from_numpy(data.action[:16])
    train_step(actor, optimizer, obs, action, obs, action,
               payload["observation_mean"], payload["observation_std"], 1.0)
    assert state_checksum(critic) == critic_before and state_checksum(target) == target_before
    for key, value in log_std_before.items():
        torch.testing.assert_close(value, actor.log_std_head.state_dict()[key], rtol=0, atol=0)


def test_v2_actor_training_eligibility_has_no_split_leakage() -> None:
    _actor, _critic, _target, _payload, arrays, audit = _assets()
    ids = {split: set(audit["episode_ids"][split]) for split in ("train", "validation", "test")}
    assert not ids["train"].intersection(ids["validation"] | ids["test"])
    eligible = np.isin(arrays["train"].category, ["nominal_success", "normal_recovered"])
    assert set(np.unique(arrays["train"].category[eligible])) == {
        "nominal_success", "normal_recovered"}
    assert not np.any(np.isin(arrays["train"].category[eligible], ["failure", "delayed_recovery"]))
