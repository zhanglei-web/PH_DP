# DAgger Learned Expert v1

## Purpose

This experiment tests whether Dataset Aggregation can reduce the closed-loop
covariate shift of the frozen Native Constrained BC Actor. It is not an SAC,
Critic, AWAC, or reward-learning experiment.

The immutable oracle is `rule_pick_place_v1` at commit
`d5ce43ff70af25491c545ec513d56e9f988c4f6b`. The learner starts from the Actor
inside `aligned_actor_critic_v2_20260814T013000Z`. Reward V2 is used only for
rollout termination and outcome classification.

## Frozen semantics

- Observation: `policy_state_42`, using the Actor artifact's frozen mean/std.
- Action: native constrained normalized action_7 (`B3 × B3 × [-1,1]`).
- Rollout: deterministic learner action only. Oracle actions are never executed.
- Supervision: the Rule Expert is queried sequentially on each learner-visited
  simulator state; only a valid Rule Expert action is a training label.
- Terminal states receive no further labels. Once the frozen Rule Expert enters
  its irrecoverable `FAILED` state, no synthetic label is invented.
- Subsampling: retain every fourth valid query (`timestep % 4 == 0`), without
  reward, Critic, outcome, or phase-based filtering.
- D0: frozen 900 nominal-success training episodes (115,021 transitions); the
  frozen 100-episode/12,817-transition validation split is never optimized.
- Batch composition: 50% D0. The remaining 50% is divided equally among all
  DAgger rounds available at that round.
- Loss: deterministic native-constrained action MSE. Actor trunk and mean head
  train; log-std remains bitwise frozen.
- Optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`, batch 256,
  gradient-norm clipping 1.0, matching the original BC provenance.

The rollout pools are 1,000,000–1,000,999, 1,010,000–1,010,999, and
1,020,000–1,020,999. Primary development is 300,000–300,099 and secondary
development is 420,000–420,099. Final test 500,000–500,099 remains sealed.

## Protective stopping rule

Each trained checkpoint at 1k, 2k, 5k, and 10k is evaluated on the complete
primary pool. The best trained checkpoint is selected by success, then fewer
illegal drops, then lower action drift. It is confirmed on the secondary pool.
If a round loses at least 20 successes in both 100-seed pools, it is an
unambiguous catastrophic regression and subsequent rounds stop.

## Formal Round 1 result

The formal run `dagger_v1_20260815T180000Z` collected 1,000 episodes and stopped
after Round 1. BC0 reproduced 50/100 primary and 53/100 secondary. The best
trained BC1 checkpoint (5k updates) achieved only 29/100 primary and 31/100
secondary. This crossed the protective stop in both pools, so D2/BC2 and
D3/BC3 were not run.

Round 1 itself had 523 successes, 157 illegal drops, 274 IK-failure-limit
terminations, 37 timeouts, and 9 unstable releases. Of 168,116 learner-visited
states, 88,344 had valid oracle queries; deterministic temporal subsampling
retained 22,473 labels. The frozen expert became irrecoverable on 79,772 later
states, and no labels were fabricated for them.

The largest correction mismatch was in P2: mean action L2 difference 0.6803,
median 0.4814, p95 1.4227. By comparison, P1/P3/P4 means were 0.0303, 0.0925,
and 0.1160. Although DAgger supplied genuine correction labels, transition-wise
MSE on this stateful expert's stage-conditioned commands did not preserve the
learner's closed-loop phase coordination. Illegal drops rose from 14 to 29 on
primary and from 15 to 33 on secondary at the selected checkpoint.

Status: `DAGGER_V1_REGRESSED`.
