# SAC Reward v1 Candidate Offline Validation v2

## Validation status

```text
Candidate: sac_reward_v1_candidate (P3 corrected to EE -> frozen above_goal)
Reference reward: collection_reward_v1
Validation run: sac_reward_v1_candidate_v2_20260812T143907Z
Manifest: manifests/rule_expert_v1_formal.json
Manifest content SHA: 5a3f3cfb76f37a95ceed5155c83158529f76cef6f82ea652b9a66ed050aa7c6b
Formal episodes / transitions: 1300 / 160932
SAC-v1 transitions after true-terminal truncation: 157302
Discount gamma: 0.995
Result: SAC_REWARD_V1_CANDIDATE_VALIDATED
```

This was a read-only recomputation. It did not modify the formal HDF5 files,
manifest, split, Rule Expert, environment reward, Actor BC, or task definition,
and it did not run Critic or SAC training.

## 1. Signal and geometry audit

The formal HDF5 contains both sides of each transition for EE, object and goal
poses, gripper opening, `object_grasped`, task milestones, expert/coarse stages,
termination labels, and the episode reset metadata. The transition convention is
`(obs_t, action_t, reward_t, obs_t+1)`.

The frozen source defines:

```python
above_goal = goal + np.array([0.0, 0.0, self.config.lift_height_m])
```

and `RuleExpertConfig.lift_height_m = 0.16`. Thus the corrected P3 target is
exactly reconstructable at both ends of every transition:

```text
p_above_goal,t = p_goal,t + [0, 0, 0.16] m
d_EA,t = ||p_EE,t - p_above_goal,t||_2
```

No object pre-place pose, empirical EE-object offset, or new geometry target is
used. P1 similarly uses the frozen grasp offset `object + [0,0,0.012]`, P4 place
uses the recorded object/goal positions, and retreat uses
`goal + [0,0,0.16]`.

## 2. Expert-stage to frozen-phase mapping

| Expert stage | Value | Frozen phase | Formal transitions | Frozen behavior |
|---|---:|---|---:|---|
| `PRE_GRASP` | 0 | P1 | 33173 | Move EE to object hover |
| `DESCEND` | 1 | P1 | 17276 | Move EE to grasp position |
| `CLOSE_GRIPPER` | 2 | P2 | 15035 | Close and establish stable grasp |
| `LIFT` | 3 | P3 | 19882 | Lift while retaining grasp |
| `TRANSPORT` | 4 | P3 | 29286 | Move EE to frozen `above_goal` |
| `DESCEND_TO_GOAL` | 5 | P4 | 16769 | Lower toward placement height |
| `OPEN_GRIPPER` | 6 | P4 | 9699 | Release and wait |
| `RETREAT` | 7 | P4 | 16352 | Withdraw EE to safe target |
| `COMPLETE` | 8 | terminal | 0 | Completed retreat |
| `FAILED` | 9 | terminal boundary | 0 | Immediately mapped by collector |
| `SETTLING` | 10 | outside active SAC P1-P4 | 3460 | Frozen recovery observation window |

The requested P3 mapping `LIFT + TRANSPORT` is therefore consistent with both
the source and the recorded enum. `SETTLING` is not treated as a fifth SAC phase.

## 3. Reward reconstruction protocol

For every progress component, the denominator is the distance at the first valid
transition of that phase in that episode:

```text
r_progress,t = weight * (d_before,t - d_after,t) / (d_phase,0 + 1e-6)
```

Progress remains signed. The reconstructed reward is:

| Component | Weight/event value | Activation |
|---|---:|---|
| P1 EE-to-grasp progress | 2.0 | `PRE_GRASP`, `DESCEND` |
| Stable-grasp event | +2.0 once | First `CLOSE_GRIPPER -> LIFT` |
| P3 EE-to-`above_goal` progress | 3.0 | `LIFT`, `TRANSPORT` |
| P4 object-to-goal progress | 2.0 | Before valid stable release |
| Successful place/release | +3.0 once | First four-step released-inside-goal window |
| P4 EE-to-retreat progress | 1.0 | After successful release, before success |
| Full task success | +10.0 once | Frozen full-success protocol |
| Illegal drop / true failure | -5.0 once | First SAC-v1 true-terminal boundary |

Delayed recovery uses Scheme B. A first illegal drop terminates reward
accumulation immediately; otherwise accumulation stops at the explicit Expert
failure boundary. No later `SETTLING` reward and no delayed `+10` are included.

## 4. Return distributions by outcome

All standard deviations use the population definition (`ddof=0`). `R` is
undiscounted and `G0` uses `gamma=0.995`.

### Undiscounted return R

| Outcome | Episodes | Mean | Std | Median | Min | Max | P5 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal success | 1000 | 22.864886012948375 | 0.03141131081793206 | 22.872740653062927 | 22.617060292181137 | 22.912498017885714 | 22.8173553623456 | 22.852283624952307 | 22.885590136275944 | 22.89644358055003 |
| Normal recovered | 78 | 22.75714177304874 | 0.08489661153096312 | 22.766458017128237 | 22.460380536054537 | 22.91319702908695 | 22.63248759578351 | 22.712763570094406 | 22.80507773175602 | 22.872193039032368 |
| Delayed recovery (SAC truncation) | 66 | 2.393860408839913 | 0.9071139246500218 | 2.0067103645639275 | 0.8897821493458622 | 3.794701841633337 | 1.0055898326904158 | 1.7620575773191693 | 3.3001508516321785 | 3.7615073742480476 |
| Failure | 156 | -0.8215983680890884 | 1.1439727899838525 | -0.7892029592742333 | -3.076314232557464 | 3.7275670577266387 | -3.0442378217719357 | -1.030053516163293 | -0.3529288575219329 | 0.8265366678926362 |

### Discounted return G0

| Outcome | Episodes | Mean | Std | Median | Min | Max | P5 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal success | 1000 | 14.247517024794186 | 0.253298194653026 | 14.248268676044555 | 13.423335859649436 | 14.92566434717863 | 13.824583375098886 | 14.080349783377002 | 14.423920606634008 | 14.664347190546922 |
| Normal recovered | 78 | 13.341193811111502 | 0.46068672535483496 | 13.350189334166217 | 12.06811058461341 | 14.438645836963984 | 12.540186123133271 | 13.092885621290204 | 13.655285816500013 | 13.957243873106222 |
| Delayed recovery (SAC truncation) | 66 | 2.5534079378051264 | 0.6648301081860718 | 2.377828020107466 | 1.3836316440291192 | 3.585440682712899 | 1.5224082181652396 | 2.0583579855331813 | 3.2061083792140312 | 3.52693942273531 |
| Failure | 156 | -0.09654676197702884 | 0.9139605263811457 | -0.1226942015491416 | -1.782572635788985 | 3.4601297072963524 | -1.6963192093768953 | -0.49946820352261667 | 0.3654368164981252 | 1.381894863236846 |

### SAC-v1 episode length

| Outcome | Episodes | Mean | Std | Median | Min | Max | P5 | P25 | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nominal success | 1000 | 127.838 | 4.652070076858259 | 128 | 116 | 142 | 120 | 124 | 131 | 136 |
| Normal recovered | 78 | 144.30769230769232 | 8.428139211711688 | 144 | 125 | 166 | 132 | 138.25 | 148.75 | 159.3 |
| Delayed recovery (truncated) | 66 | 106.48484848484848 | 17.71169997166094 | 105 | 80 | 197 | 87.25 | 95.25 | 113 | 133.5 |
| Failure (truncated) | 156 | 71.66666666666667 | 14.514066268903406 | 72 | 45 | 120 | 49.75 | 62 | 80 | 97 |

## 5. Reward contribution by component

The table reports mean episode sums by outcome. Event and terminal components
remain separate from phase progress.

| Component | Nominal | Normal recovered | Delayed recovery | Failure |
|---|---:|---:|---:|---:|
| P1 progress | 1.9675235890694558 | 1.9628522357118663 | 1.9631485687985377 | 1.9649441148486615 |
| P2 grasp event | 2.0 | 2.0 | 2.0 | 1.705128205128205 |
| P3 progress | 2.9817407076786857 | 2.9594163966181233 | 2.7883135582082597 | 0.494605653404942 |
| P4 place progress | 1.9386052942459888 | 1.8717655852444834 | 0.642398281833116 | 0.013723658529102999 |
| Place event | 3.0 | 3.0 | 0.0 | 0.0 |
| P4 retreat progress | 0.9770164219542466 | 0.9631075554742673 | 0.0 | 0.0 |
| Full-success terminal | 10.0 | 10.0 | 0.0 | 0.0 |
| Failure terminal | 0.0 | 0.0 | -0.6060606060606061 | -0.7692307692307693 |
| Illegal drop | 0.0 | 0.0 | -4.393939393939394 | -4.230769230769231 |

Across all 1300 episodes, mean component contributions are: P1 `1.96671165`,
P2 event `1.96461538`, P3 `2.67212489`, P4-place `1.63780168`, place event
`2.48769231`, P4-retreat `0.80933755`, success terminal `8.29230769`, and total
failure/drop terminal `-0.85384615`.

For nominal success, dense progress contributes `7.864886012948377` (34.40%),
events contribute `5.0` (21.87%), and the `+10` terminal contributes 43.74% of
the mean `22.864886012948375` return. P3 contributes 13.04% of total nominal
return. It is the largest dense term, as expected from its weight, but does not
dominate the trajectory. Terminal and dense feedback are both material.

## 6. P3 EE-to-above-goal validation

### Transition and episode statistics

| Metric | Result |
|---|---:|
| P3 transitions | 49012 |
| Pure P3 reward mean | 0.07087575212984198 |
| Pure P3 reward std | 0.04265004857748118 |
| Median | 0.08497142017845036 |
| Min / max | -0.09113684332676521 / 0.2538956581340274 |
| P5 / P95 | 0.006271521395311216 / 0.12908169337660255 |
| Positive / negative / exactly near-zero | 48091 / 921 / 0 |
| Mean contribution over all episodes | 2.672124894913704 |
| Mean contribution among P3-reaching episodes | 2.7202524380484065 |
| Nominal mean / std contribution | 2.9817407076786857 / 0.0048839493025151165 |
| Normal-recovered mean / std contribution | 2.9594163966181233 / 0.02516327340958591 |

The sign is correct by construction and in the data: closer steps are positive,
farther steps are negative. Exact zero-motion would produce exactly zero reward;
the frozen active P3 controller has no exactly stationary transition. There are
24 transitions with `|r_P3| <= 1e-4`, confirming near-stationary motion produces
near-zero reward.

The maximum absolute P3 telescoping residual is
`1.1102230246251565e-16`. Gross positive P3 reward is `3486.509936793494`, gross
negative reward is `-12.747573405679411`, and net is `3473.7623633878156`.
There are 1114 adjacent sign reversals among 47755 comparable adjacent P3 pairs;
the negative leg is retained, so reversals cannot accumulate only their forward
part. No P3 completion bonus was added.

## 7. Events and SAC-v1 terminal semantics

| Event/terminal | Episodes | Maximum per episode |
|---|---:|---:|
| Stable grasp event | 1277 | 1 |
| Valid stable place event | 1078 | 1 |
| Full success | 1078 | 1 |
| Illegal-drop terminal | 190 | 1 |
| Other explicit true failure | 32 | 1 |

The 66 formal delayed-recovery episodes become 58 illegal-drop terminals and 8
explicit Expert-failure terminals. All stop before SETTLING recovery and all get
one `-5`; none receives place, retreat, or full-success reward. This exactly
implements Scheme B while retaining the original trajectory only for analysis.

The 156 formal failures become 132 illegal-drop and 24 explicit-failure
terminals. `time_limit` truncation would receive no terminal penalty; none of the
formal Rule Expert episodes in this validation ended that way under the SAC-v1
reconstruction.

## 8. Reward-hacking checks

| Check | Evidence | Result |
|---|---|---|
| Oscillation | Signed P3 negative sum `-12.747573405679411`; telescoping residual <= `1.11e-16` | Forward/backward motion is not one-sided reward accumulation |
| Staying | Progress is exactly zero when `d_before=d_after`; 24 near-stationary P3 samples have `|r|<=1e-4` | No staying reward |
| Holding object in goal | After the place event, P4 object-goal progress and place event are disabled | Cannot repeatedly farm goal occupancy |
| Early release | `+3` requires the four-step released-inside-goal window; opening alone is insufficient | No early-release bonus |
| Illegal drop | First illegal drop truncates the SAC trajectory; 190 episodes affected | No SETTLING/recovery reward after drop |
| Enter goal without release | Can receive only signed net place progress; no `+3`, retreat, or `+10` | Cannot approach full-success return |
| Release without retreat | Does not receive `+10`; no repeated place bonus | Not equivalent to full success |
| Long episode bias | Nominal `R`: Pearson `0.0493133`, Spearman `0.113150` | Undiscounted reward has no meaningful positive length incentive |

For nominal episodes, length versus discounted `G0` has Pearson `-0.99091885`.
This is the expected effect of `gamma=0.995`: later success bonuses are worth
less, so faster completion is preferred. It is not caused by per-step staying
reward. The old undiscounted collection return has Pearson `-0.92809142` with
length, whereas candidate undiscounted `R` is effectively length-neutral.

The smallest complete-success return is `22.460380536054537`; the largest
delayed/failure partial return is `3.794701841633337`. Thus the observed partial
behaviors—grasp/transport without a valid release and retreat—remain far below
full success.

## 9. Outcome ranking and overlap

Pairwise probabilities use half credit for ties.

| Comparison | Undiscounted R | Discounted G0 |
|---|---:|---:|
| P(nominal > failure) | 1.0 | 1.0 |
| P(normal recovered > failure) | 1.0 | 1.0 |
| P(delayed recovery > failure) | 0.9880536130536131 | 0.9882478632478633 |
| P(success-like > failure) | 1.0 | 1.0 |

Nominal and normal-recovered distributions are intentionally close, with normal
recovery discounted more because it is longer. Scheme-B delayed recovery is kept
separate from success and overlaps modestly with failure: some failed trajectories
made substantial P3 progress before their `-5`, while delayed trajectories
generally completed more net progress before the same terminal penalty. This
overlap is consistent with progress shaping and does not erase the success/failure
separation.

## 10. Reward scale

Across 157302 valid SAC-v1 transitions:

| Statistic | Reward |
|---|---:|
| Min | -5.080246300746435 |
| Max | 10.0371073055387 |
| Mean | 0.156830609355302 |
| Std | 0.8919209658003312 |
| P1 | -0.010922140084493565 |
| P5 | 0.0 |
| P50 | 0.059936171419740884 |
| P95 | 0.1429927360321864 |
| P99 | 3.000016811987911 |

Across episode `G0`, min/max/std are
`-1.782572635788985 / 14.92566434717863 / 5.123788693211364`.

No extreme numerical outlier is present. A transition can combine a terminal or
event reward with a small signed progress term, explaining extrema slightly past
`-5` and `+10`. The success terminal is not drowned by dense reward: it is 43.74%
of nominal undiscounted return. Dense progress is also not drowned: it contributes
34.40%, and the two event bonuses contribute 21.87%. No reward rescaling or
reweighting is supported by these data.

## 11. Candidate versus collection_reward_v1

Mean old undiscounted returns are:

| Outcome | collection_reward_v1 mean R | Candidate mean R |
|---|---:|---:|
| Nominal success | 7.451985677909517 | 22.864886012948375 |
| Normal recovered | 7.720939267885578 | 22.75714177304874 |
| Delayed recovery | -18.731948172075274 | 2.393860408839913 |
| Failure | -21.088113683100165 | -0.8215983680890884 |

`collection_reward_v1 = -d_object,goal + I(d_object,goal<0.055)` is dominated by
time spent far from the goal and does not explicitly represent grasp, legal
release, retreat, or full completion. The corrected candidate explicitly rewards
the ordered task semantics:

```text
EE approaches grasp
-> stable grasp
-> EE transports to frozen above_goal
-> object approaches goal
-> valid stable release
-> EE retreat
-> full success
```

It also makes full success strictly higher than every observed SAC-truncated
partial/failure trajectory. The corrected P3 therefore resolves the structural
blocker from v1 without inventing an object target.

## 12. Delayed-recovery interpretation

Delayed recovery has mean `R=2.393860408839913` and mean
`G0=2.5534079378051264`. These values consist of genuine progress before failure,
the one-time grasp event, and `-5`; they contain no reward from subsequent
SETTLING recovery. They are much lower than normal success/recovery and only
partly overlap failure.

For SAC v1, these episodes should remain a separately labelled diagnostic class,
not be relabelled as full successes. If later used for critic data, only the
prefix through the reconstructed true terminal is semantically compatible with
SAC v1. The post-terminal SETTLING suffix should not enter the SAC-v1 replay
return. This is a training-data recommendation, not a change made by this audit.

## 13. Conclusion

The P3 target is an exact frozen Rule Expert target, its signed reward has the
correct sign and machine-precision telescoping behavior, and its scale is balanced
against the other dense, event, and terminal components. Event rewards are
one-shot, illegal drops stop future accumulation, complete success is clearly
separated from observed partial/failure behavior, and no structural reward-hacking
path was found in the requested checks.

```text
SAC_REWARD_V1_CANDIDATE_VALIDATED
```

The candidate can proceed to formal reward freeze. This report does not itself
change the live environment reward or start any training.

## Machine-readable artifacts

Authoritative output directory:

```text
outputs/reward_validation/sac_reward_v1_candidate_v2_20260812T143907Z/
```

Files:

- `summary.json`
- `episode_returns.csv`
- `phase_statistics.csv`
- `reward_components.csv`

The validation implementation is `scripts/validate_sac_reward_v2.py`. All output
files are derived artifacts; the formal HDF5 dataset was opened read-only.
