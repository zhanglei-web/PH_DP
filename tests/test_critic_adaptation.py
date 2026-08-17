from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from mujoco_shared_control.sac.aligned_initialization import load_aligned_v2
from mujoco_shared_control.sac.critic_adaptation import (
    ActorTransitionArrays, CriticAdaptationConfig, concatenate_batch, critic_update,
    exact_mixed_indices, fixed_policy_td_target, module_checksum,
    polyak_update,
)
from mujoco_shared_control.sac.critic import TwinSACCritic
from mujoco_shared_control.sac.critic_pretraining import build_arrays_from_semantic_run


ALIGNED = Path("outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt")


def test_fixed_policy_target_terminal_and_truncation_without_entropy() -> None:
    actor, critic, target, payload = load_aligned_v2(ALIGNED)
    actor.eval(); actor.requires_grad_(False); target.eval(); target.requires_grad_(False)
    reward = torch.tensor([[2.0], [2.0]])
    next_observation = torch.zeros(2, 42)
    terminated = torch.tensor([[True], [False]])
    value = fixed_policy_td_target(
        actor, target, reward, next_observation, terminated,
        payload["observation_mean"], payload["observation_std"],
    )
    assert value[0].item() == 2.0
    with torch.no_grad():
        normalized = (next_observation-payload["observation_mean"])/payload["observation_std"]
        action = actor.deterministic_action(normalized)
        q1,q2 = target(normalized,action)
        expected = 2.0 + .995*torch.minimum(q1,q2)[1]
    torch.testing.assert_close(value[1], expected)


def test_exact_half_batches_and_critic_only_update() -> None:
    actor, critic, target, payload = load_aligned_v2(ALIGNED)
    actor.eval(); actor.requires_grad_(False)
    actor_before = module_checksum(actor); target_before = module_checksum(target)
    arrays, _ = build_arrays_from_semantic_run(
        Path("manifests/rule_expert_v1_formal.json"),
        Path("outputs/sac_reward/sac_reward_v2_candidate_20260814T001000Z"),
        reward_version="sac_reward_v2_candidate")
    offline = arrays["train"]
    # The batch merger only relies on the six shared transition arrays.
    online = deepcopy(offline.subset(np.arange(len(offline)) < 256))
    config = CriticAdaptationConfig()
    oi, ni = exact_mixed_indices(len(offline), len(online), config, np.random.default_rng(7))
    assert len(oi) == len(ni) == 128
    batch = concatenate_batch(offline, online, oi, ni)
    assert batch["observation"].shape == (256,42)
    critic_before = module_checksum(critic)
    optimizer = torch.optim.Adam(critic.parameters(),lr=config.learning_rate)
    metrics = critic_update(critic,target,actor,optimizer,batch,
                            payload["observation_mean"],payload["observation_std"],config)
    assert np.isfinite(list(metrics.values())).all()
    assert module_checksum(critic) != critic_before
    assert module_checksum(target) != target_before
    assert module_checksum(actor) == actor_before


def test_config_freezes_ratio_gamma_tau() -> None:
    assert CriticAdaptationConfig().half_batch == 128
    for kwargs in ({"offline_fraction": .25},{"gamma": .99},{"tau": .01}):
        try: CriticAdaptationConfig(**kwargs)
        except ValueError: pass
        else: raise AssertionError("invalid frozen configuration accepted")


def test_online_artifact_episode_split_constraints_and_final_seed_isolation() -> None:
    root = Path("outputs/critic_adaptation/critic_only_online_adaptation_v1_20260815T010000Z")
    splits = {name: ActorTransitionArrays.load(root/"online_actor_dataset"/f"{name}.npz")
              for name in ("train","validation","test")}
    ids = {name:set(np.unique(value.episode_id)) for name,value in splits.items()}
    assert not ids["train"] & ids["validation"]
    assert not ids["train"] & ids["test"]
    assert not ids["validation"] & ids["test"]
    assert (len(ids["train"]),len(ids["validation"]),len(ids["test"])) == (800,100,100)
    all_seeds=np.concatenate([value.seed for value in splits.values()])
    assert all_seeds.min()==800000 and all_seeds.max()==800999
    assert not np.any((all_seeds>=500000)&(all_seeds<=500099))
    for value in splits.values():
        assert np.all(np.linalg.norm(value.action[:,:3],axis=1)<=1+1e-6)
        assert np.all(np.linalg.norm(value.action[:,3:6],axis=1)<=1+1e-6)
        assert np.all(np.abs(value.action[:,6])<=1+1e-6)


def test_polyak_direction_and_adapted_checkpoint_exact_reload() -> None:
    online=TwinSACCritic();target=TwinSACCritic()
    for parameter in online.parameters(): parameter.data.fill_(1)
    for parameter in target.parameters(): parameter.data.zero_()
    polyak_update(online,target,.005)
    assert all(torch.allclose(parameter,torch.full_like(parameter,.005))
               for parameter in target.parameters())
    checkpoint=Path("outputs/critic_adaptation/critic_only_online_adaptation_v1_20260815T010000Z/critic_checkpoints/critic_step_10000.pt")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    restored=TwinSACCritic();restored.load_state_dict(payload["critic_state_dict"])
    for name,value in restored.state_dict().items():
        torch.testing.assert_close(value,payload["critic_state_dict"][name],rtol=0,atol=0)
    assert payload["critic_updates"]==10000 and payload["reward_version"]=="sac_reward_v2_candidate"
