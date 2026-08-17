import numpy as np
import torch

from mujoco_shared_control.sac.critic import TwinSACCritic
from pathlib import Path

from mujoco_shared_control.sac.critic_pretraining import (
    build_arrays,
    build_arrays_from_semantic_run,
    fixed_split,
    monte_carlo_returns,
)


def test_monte_carlo_return_and_true_terminal() -> None:
    reward=np.array([1.,2.,3.]);term=np.array([0,0,1],bool);trunc=np.zeros(3,bool)
    np.testing.assert_allclose(monte_carlo_returns(reward,term,trunc,.5),[2.75,3.5,3.])


def test_truncation_ends_finite_mc_record_without_changing_label() -> None:
    reward=np.array([1.,2.]);term=np.zeros(2,bool);trunc=np.array([0,1],bool)
    np.testing.assert_allclose(monte_carlo_returns(reward,term,trunc,.5),[2.,2.])
    assert not term[-1] and trunc[-1]


def test_episode_seed_split_has_no_leakage() -> None:
    assert fixed_split(100000)=="train" and fixed_split(100800)=="validation"
    assert fixed_split(100900)=="test" and fixed_split(200299)=="test"


def test_twin_critic_checkpoint_reload() -> None:
    model=TwinSACCritic();clone=TwinSACCritic();clone.load_state_dict(model.state_dict())
    x=torch.randn(3,42);a=torch.randn(3,7)
    for lhs,rhs in zip(model(x,a),clone(x,a),strict=True): torch.testing.assert_close(lhs,rhs)


def test_mixed_formal_data_split_and_attempted_action_semantics() -> None:
    arrays,audit=build_arrays(
        Path("manifests/rule_expert_v1_formal.json"),
        Path("outputs/reward_validation/sac_reward_v1_regression_20260812T150245Z"),
    )
    assert audit["episodes"]=={"train":1040,"validation":130,"test":130}
    assert audit["transitions"]=={"train":125942,"validation":15855,"test":15505}
    assert audit["fallback_attempted_action_rows"]==3
    assert audit["adapter_projected_rows"]==0
    ids=[set(audit["episode_ids"][split]) for split in ("train","validation","test")]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    assert set(np.unique(arrays["train"].category))=={
        "nominal_success","normal_recovered","delayed_recovery","failure"
    }


def test_v2_semantic_corpus_reuses_v1_episode_split_and_has_no_retreat() -> None:
    manifest = Path("manifests/rule_expert_v1_formal.json")
    v1, v1_audit = build_arrays(
        manifest, Path("outputs/reward_validation/sac_reward_v1_regression_20260812T150245Z")
    )
    v2, v2_audit = build_arrays_from_semantic_run(
        manifest, Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z"),
        reward_version="sac_reward_v2_candidate",
    )
    for split in ("train", "validation", "test"):
        assert v2_audit["episode_ids"][split] == v1_audit["episode_ids"][split]
        assert not np.any(v2[split].phase == "OTHER")
    assert v2_audit["total_semantic_transitions"] == 135560
    assert v2_audit["adapter_projected_rows"] == 0
    assert sum(len(value) for value in v2.values()) < sum(len(value) for value in v1.values())


def test_v2_mc_target_ends_at_new_success_terminal() -> None:
    arrays, _ = build_arrays_from_semantic_run(
        Path("manifests/rule_expert_v1_formal.json"),
        Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z"),
        reward_version="sac_reward_v2_candidate",
    )
    data = arrays["test"]
    episode_id = next(value for value in np.unique(data.episode_id)
                      if value.endswith("000900"))
    mask = data.episode_id == episode_id
    assert data.terminated[mask][-1, 0] and not data.truncated[mask][-1, 0]
    assert data.reward[mask][-1, 0] > 9.9
    assert data.mc_return[mask][-1, 0] == data.reward[mask][-1, 0]
