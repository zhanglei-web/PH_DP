# SAC Reward V2 Candidate Specification

## Motivation

The frozen v1 P4 combines three different control objectives:

```text
Place -> Release -> Retreat
```

Aligned-Critic diagnostics were strong in P1/P2 but unreliable across this
heterogeneous P4. V2 changes only the terminal part of P4:

```text
P1 Pre-Grasp -> P2 Grasp -> P3 Transport -> P4 Place & Release -> SUCCESS
```

Retreat is optional post-success rule-based cleanup. It is outside the RL episode,
Replay, reward, return, Critic data, and success condition.

## Version isolation

- Historical `sac_reward_v1` and `SACPickPlaceProtocol` are unchanged.
- Candidate version: `sac_reward_v2_candidate`.
- Implementation: `tasks/sac_reward_v2.py`, dispatched independently by
  `PickPlaceEnv(reward_version="sac_reward_v2_candidate")`.

## Unchanged P1--P3

All thresholds, initial-distance normalization, phase transitions, signed-progress
rules, and weights are inherited unchanged from v1:

- P1 EE-to-grasp signed progress, weight 2.0.
- P2 stable grasp after 8 consecutive grasped steps, one-shot +2.
- P3 EE-to-above-goal signed progress, weight 3.0.

Offline regression over the formal corpus gives maximum component differences of
`8.40e-16`, `0`, and `1.31e-15`, respectively.

## P4 Place & Release

Before success:

```text
d_t = ||p_object,t - p_goal,t||_2
r_P4 = 2 * (d_prev - d_now) / (d_P4,initial + 1e-6)
```

The v1 goal tolerance (0.055 m) and stable-release protocol (four consecutive
steps inside goal while released) are reused exactly. On the fourth step:

```text
inside_goal && released && stable -> +10, terminated=True
```

The former legal-release +3 is disabled. Retreat progress is disabled. The
success transition may also contain its signed P4 progress for that transition,
so total reward is `10 + r_P4`; this is not a duplicate event bonus.

Illegal drop and true failure remain -5 true terminals. A pure time limit remains
`truncated=True`, `terminated=False`, without a -5 penalty. Gamma remains 0.995.

## Formal corpus validation

Source: `formal_rule_v1_20260812T050822Z`, 1300 episodes. Original HDF5 and formal
manifest are read-only. The derived semantic manifest records each V2 cutoff and
does not duplicate or mutate HDF5.

Compatibility:

| Outcome | Episodes | V2 success found |
|---|---:|---:|
| Nominal success | 1000 | 1000 |
| Normal recovered | 78 | 78 |
| Delayed recovery | 66 | 0 |
| Failure | 156 | 0 |

The delayed-recovery result preserves the established SAC semantics: these paths
terminate at illegal drop or explicit failure before legacy SETTLING recovery.

Transition counts:

```text
V1 semantic transitions: 157302
V2 semantic transitions: 135560
V1 P4 transitions:         42796
V2 P4 transitions:         21054
V1 retreat transitions:    16352
V2 retreat transitions:        0
```

Discounted start returns (gamma 0.995):

| Outcome | V1 mean | V2 mean | V2 std | V2 min / max |
|---|---:|---:|---:|---:|
| Nominal success | 14.2475 | 12.5138 | 0.2134 | 11.8089 / 13.0866 |
| Normal recovered | 13.3412 | 11.7911 | 0.4028 | 10.6388 / 12.6802 |
| Delayed recovery | 2.5534 | 2.5534 | 0.6648 | 1.3836 / 3.5854 |
| Failure | -0.0965 | -0.0965 | 0.9140 | -1.7826 / 3.4601 |

The expected success-return reduction comes from deleting +3 and retreat progress
and terminating earlier. Outcome ordering remains sensible.

Event audit: 1078 successes, at most one +10 per episode, zero +3 events, zero
retreat rewards, zero post-success transitions, and all rewards finite. In the
sampled success neighborhoods, the preceding five rewards lie in
`[-0.0659, 0.1767]`; the success transition total is about 10 with only the signed
P4 progress added. There are no NaN/Inf or duplicate terminal events.

## Artifact

`outputs/sac_reward/sac_reward_v2_candidate_20260813T235000Z/`

This candidate is validated offline only. No Actor, Critic, AWAC, RLPD, or online
SAC training was performed.

`SAC_REWARD_V2_CANDIDATE_VALIDATED`
