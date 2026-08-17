# Mixed Critic V2: Reward-Semantics Causal Test

## Hypothesis

- H0: the old P4 ranking failure was not primarily caused by combining retreat
  with place/release reward and terminal semantics.
- H1: removing retreat and ending at stable in-goal release materially improves
  P4 value ranking.

This is a strict single-variable experiment. V2 reuses the V1 episode split,
`policy_state_42`, constrained action_7, observation normalization, Twin-Q
49-256-256-256-1 architecture, SiLU, optimizer, learning rate, batch size,
mixed-training seed, early stopping, checkpoint criterion, frozen Actor, and
held-out audit states. Only reward/terminal semantics and retreat-row removal differ.

## Data and target

The source remains all 1300 formal Rule Expert episodes. The derived V2 corpus has
135,560 transitions and no retreat rows. Episode membership is byte-for-byte equal
to V1's fixed train/validation/test ID lists. Twin Q regresses the full finite MC
return under `sac_reward_v2_candidate` and gamma 0.995.

## Held-out return prediction

| Metric | Critic V1 | Critic V2 | Difference |
|---|---:|---:|---:|
| Transitions | 15,505 | 13,398 | -2,107 |
| MAE | 2.1222 | 2.0733 | -0.0489 |
| RMSE | 4.0355 | 3.7363 | -0.2992 |
| Pearson | 0.6642 | 0.6600 | -0.0042 |
| Spearman | 0.6014 | 0.5792 | -0.0222 |

V2 Q1/Q2 disagreement MAE is 0.4294. Q values and all metrics are finite. Absolute
Q scales are not compared because successful return definitions changed.

## Expert-vs-Actor H=20 audit

The same 100 held-out reconstructed states are used, with 25 states per phase.
Each branch differs only in its first action (recorded Rule Expert versus frozen
deterministic constrained Actor), then uses the same Rule Expert feedback
continuation and V2 reward/terminal protocol.

| Phase | V1 Spearman | V2 Spearman | Delta | V1 sign | V2 sign |
|---|---:|---:|---:|---:|---:|
| Overall | 0.538 | 0.619 | +0.082 | 85% | 88.5% |
| P1 | 0.818 | 0.799 | -0.018 | 100% | 100% |
| P2 | 0.901 | 0.892 | -0.008 | 100% | 100% |
| P3 | 0.312 | 0.311 | -0.001 | 84% | 76% |
| P4 | -0.312 | +0.233 | **+0.545** | 56% | **76.2%** |

P1/P2 remain strong. P3 rank correlation is effectively unchanged, while its sign
agreement falls eight points and is explicitly retained as a caveat. Overall and
P4 improve; overall improvement is not used to hide phase results.

## P4 substage audit

| Substage | Samples | H=20 Spearman | Sign agreement | Expert better / Actor better |
|---|---:|---:|---:|---:|
| P4a Place | 21 | 0.238 | 76.2% | 4 / 17 |
| P4b Release/Stabilize | 4 | N/A | N/A | 0 / 0 (4 ties) |

No P4c/retreat sample exists in V2. P4a is positively ranked. P4b cannot be judged
from four all-tie samples; this is an identifiability limitation, not evidence of
success or failure. No Markov augmentation or new counterfactual data is added.

## Aligned artifact

- Critic: `outputs/sac_critic/sac_critic_pretrain_v2_20260814T010000Z/critic_pretrained_v2_best.pt`
- Aligned: `outputs/sac_aligned/aligned_actor_critic_v2_20260814T013000Z/aligned_actor_critic_v2.pt`
- Actor source SHA is identical to aligned v1; the Actor was not updated.
- Target critics are exact Q1/Q2 copies.
- No online Replay, optimizer resume, Actor training, AWAC, RLPD, or online SAC.

## Scientific conclusion

P4 changes from negative to positive correlation and gains 20.2 percentage points
in sign agreement without harming P1/P2 or materially changing P3 Spearman. This
supports H1: the old mixed retreat semantics were a major cause of P4 ranking
failure. It does not establish that Release/Stabilize is independently solved,
because that substage has insufficient non-tied samples.

`MIXED_CRITIC_V2_P4_IMPROVED`
