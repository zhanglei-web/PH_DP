# Offline Advantage-Weighted BC v1

## Scope

This experiment is **not** online SAC and is not full AWAC. It is a bounded,
Critic-guided, positive-advantage-filtered behavior-cloning experiment. The
Actor is only trained toward actions that already exist in the frozen Rule
Expert corpus. No gradient through `Q(s, actor(s))` is used.

## Frozen inputs

- Actor initialization: Native Constrained Actor v2 from
  `aligned_actor_critic_v2_20260814T013000Z`.
- Critic: frozen Mixed Critic v2, using `min(Q1, Q2)`.
- Observation: frozen `policy_state_42` ordering and normalization.
- Action: native constrained `action_7`; translation and rotation are inside
  their unit L2 balls and gripper is in `[-1, 1]`.
- Reward/task corpus: `sac_reward_v2_candidate` semantic corpus.
- Final test seeds `500000-500099` remain sealed.

## Fixed advantage dataset

Eligible Actor examples are nominal-success and normal-recovered transitions
from the frozen episode-level train/validation/test split. Delayed recovery and
failure examples remain Critic knowledge but are not Actor targets.

For every eligible transition, the initial Actor `pi_0` is frozen and

```text
Q_data  = min(Q1(s, a_data), Q2(s, a_data))
Q_actor = min(Q1(s, pi_0(s)), Q2(s, pi_0(s)))
A_data  = Q_data - Q_actor
```

The positive filter is exactly `A_data > 0`. Advantages are computed once and
are not refreshed as the Actor changes.

## Actor update

Every step samples an anchor batch uniformly from nominal-success train
transitions and an improvement batch uniformly from positive-advantage nominal
or normal-recovered train transitions. Both batches have size 256.

```text
L_actor = MSE(pi(s_anchor), a_expert)
        + MSE(pi(s_improve), a_data)
```

The original BC optimizer provenance is reused: AdamW, learning rate `3e-4`,
weight decay `1e-4`, and gradient clipping `1.0`. Actor trunk and mean head are
trainable; `log_std` is frozen. Q1/Q2, targets, alpha, replay, and environment
transitions are never updated or used for gradients.

Checkpoints are fixed at steps 0, 1k, 2k, 5k, and 10k. Primary selection uses
seeds `300000-300099`; ties favor smaller drift. The best nonzero update is
also evaluated on secondary seeds `420000-420099` with paired outcomes.

## Outcome

The experiment is recorded as a regressed candidate. Primary V2 task success
was `50/100` at step 0 and `45/100` at the best nonzero checkpoint (step 1k),
then fell to `26/100`, `12/100`, and `1/100`. Secondary success changed from
`53/100` to `55/100`, with 29 gains and 27 losses (`p=0.894`), which does not
confirm a reproducible gain. The initial Actor remains the selected checkpoint.

This result shows that a zero-threshold binary Critic filter plus equal-weight
BC does not provide reliable conservative policy improvement in this task.
No threshold, temperature, learning-rate, dataset, or Critic changes were made
after observing the result.
