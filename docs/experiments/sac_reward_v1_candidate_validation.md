# SAC Reward v1 Candidate Offline Validation

## Validation status

```text
Candidate: sac_reward_v1_candidate
Reference reward: collection_reward_v1
Manifest: manifests/rule_expert_v1_formal.json
Manifest content SHA: 5a3f3cfb76f37a95ceed5155c83158529f76cef6f82ea652b9a66ed050aa7c6b
Episodes: 1300
Transitions: 160932
Discount for requested return analysis: 0.995
Result: SAC_REWARD_V1_REDIRECT_REQUIRED
```

This was a read-only validation. No HDF5, manifest, split, environment reward, Rule Expert, Actor BC, Critic, or SAC implementation was modified. No training or collection was run.

The candidate cannot be fully recomputed without inventing a P3 target that was not part of the frozen task implementation. Consequently, candidate total rewards, full candidate returns, candidate ranking probabilities, and total candidate reward scale are reported as **N/A**, rather than filled using an assumed geometry.

## 1. Reward signal audit

### Available signals

| Required signal | Formal HDF5 source | Shape/type | Status |
|---|---|---|---|
| EE pose | `observations/ee_pose_xyz_wxyz`, `next_observations/...` | `(T,7)`, float64 | Available |
| Object pose | `observations/object_pose_xyz_wxyz`, `next_observations/...` | `(T,7)`, float64 | Available |
| Goal pose | `observations/goal_pose_xyz_wxyz`, `next_observations/...` | `(T,7)`, float64 | Available |
| Gripper opening | `observations/gripper_opening`, `next_observations/...` | `(T,)`, float64 | Available |
| `object_grasped` | `observations/object_grasped`, `next_observations/...` | `(T,)`, uint8 | Available |
| Task milestones | `labels/task_milestones` | `(T,5)`, uint8 | Available |
| Expert stage | `labels/expert_stage`, `labels/next_expert_stage` | `(T,)`, uint8 | Available |
| Coarse stage | `labels/stage`, `labels/next_stage` | `(T,)`, uint8 | Available |
| Termination reason | root attribute and `labels/termination_reason` | string | Available |
| Terminated/truncated | `labels/terminated`, `labels/truncated` | `(T,)`, uint8 | Available |
| Episode initial state | row-zero observation and reset metadata | structured | Available |

All observations use the transition convention:

```text
obs_t, action_t, reward_t, obs_t+1
```

### Reconstructable targets

The frozen `RuleExpertConfig` contains:

```text
grasp_offset_z_m = 0.012
lift_height_m = 0.16
place_height_m = 0.025
retreat_height_m = 0.16
position_tolerance_m = 0.008
settle_steps = 8
success_settle_steps = 4
goal tolerance = 0.055 m
```

The following targets can be reconstructed exactly from each transition and the frozen configuration:

| Target | Frozen formula | Evidence |
|---|---|---|
| Grasp position | `p_object + [0,0,0.012]` | Exact target used by `DESCEND` and `CLOSE_GRIPPER` |
| EE above-goal position | `p_goal + [0,0,0.16]` | Exact EE target used by `TRANSPORT` |
| Place position for EE | `p_goal + [0,0,0.025]` | Exact EE target used by `DESCEND_TO_GOAL` and `OPEN_GRIPPER` |
| Retreat position for EE | `p_goal + [0,0,0.16]` | Exact EE target used by `RETREAT` |

### Missing P3 signal

The candidate requires:

```text
d_op = ||p_object - p_pre-place||
```

However, the frozen implementation defines an **EE above-goal target**, not an object pre-place target. Neither HDF5 nor episode metadata stores `p_pre-place` for the object.

The actual EE–object offset at the stable-grasp transition is not constant:

```text
samples: 1277
mean EE-object offset: [-0.00245002, 0.00001919, 0.00999175] m
component std:          [ 0.00148911, 0.00050827, 0.00130571] m
offset norm min/max:    0.00176018 / 0.01679091 m
```

At the recorded P3→P4 transition:

```text
object-goal offset mean: [0.00163408, -0.00006543, 0.14911913] m
object-goal z min/max:   0.13543433 / 0.15750312 m

EE-goal offset mean:    [-0.00115310, -0.00006530, 0.15940408] m
```

Therefore, using any of the following would create a new, unfrozen geometry definition:

```text
p_pre-place = p_goal + [0,0,0.16]
p_pre-place = p_goal + [0,0,0.16] - nominal_grasp_offset
p_pre-place = empirical mean object position at P3 completion
```

No one of them is the exact object target used by the frozen Rule Expert. Per the validation constraint, it was not guessed.

```text
REWARD_SIGNAL_MISSING: object pre-place pose for P3
```

## 2. Old Expert stage to frozen four-phase mapping

The formal data contains the expected stage enum. Counts cover all 160932 transitions.

| Value | Old stage | Frozen phase | Formal transitions | Code semantics |
|---:|---|---|---:|---|
| 0 | `PRE_GRASP` | P1 Pre-Grasp | 33173 | Move EE to object hover position |
| 1 | `DESCEND` | P1 Pre-Grasp | 17276 | Move EE from hover to grasp position |
| 2 | `CLOSE_GRIPPER` | P2 Grasp | 15035 | Close and require stable grasp |
| 3 | `LIFT` | P3 Transport | 19882 | Lift while maintaining grasp |
| 4 | `TRANSPORT` | P3 Transport | 29286 | Move EE to above-goal target |
| 5 | `DESCEND_TO_GOAL` | P4 Place & Retreat | 16769 | Lower object toward placement height |
| 6 | `OPEN_GRIPPER` | P4 Place & Retreat | 9699 | Release and wait for release stability |
| 7 | `RETREAT` | P4 Place & Retreat | 16352 | Move EE to safe retreat target |
| 8 | `COMPLETE` | Terminal success state | 0 | Reached after retreat; collector terminates |
| 9 | `FAILED` | Failure transition state | 0 | Collector immediately maps it to SETTLING |
| 10 | `SETTLING` | Recovery observation, outside normal P1–P4 flow | 3460 | Safe open-gripper hold after failure/retreat |

The proposed P1–P4 mapping is consistent with the frozen code. `SETTLING` is not a fifth active-control phase and should not be silently assigned to P4 for future SAC v1.

## 3. Frozen event semantics

### Stable grasp

`object_grasped` itself is an instantaneous two-finger contact heuristic. The frozen stable-grasp event is the single transition:

```text
CLOSE_GRIPPER → LIFT
```

It occurs after eight consecutive Rule Expert steps for which `object_grasped` remains true.

Measured event counts:

| Category | Episodes | Stable-grasp events |
|---|---:|---:|
| Nominal success | 1000 | 1000 |
| Normal recovered | 78 | 78 |
| Delayed recovery | 66 | 66 |
| Failure | 156 | 133 |
| Total | 1300 | 1277 |

Maximum per episode: `1`. No repeated grasp bonus was detected under this definition.

### Successful place/release

The frozen collector success window requires:

```text
object-goal distance < 0.055 m
AND object_grasped == false
for 4 consecutive transitions
```

For full task success, the monotonic milestones and retreat requirement must also be satisfied. The candidate `+3` place event can be defined as the first completion of this four-step released-inside-goal window, once per episode.

| Category | Episodes | Reconstructed place events |
|---|---:|---:|
| Nominal success | 1000 | 1000 |
| Normal recovered | 78 | 78 |
| Delayed recovery | 66 | 66 |
| Failure | 156 | 1 |

Maximum per episode: `1`. The one failure episode briefly met the local release window but did not satisfy the complete frozen success protocol; this demonstrates why `+3 place` and `+10 full success` must remain separate.

### Full success

The formal success event is `labels/task_success=True`, backed by all five milestones and the four-step released-inside-goal window. It occurs at most once per episode.

## 4. Illegal-drop audit

The requested future SAC definition was applied conceptually:

```text
stable grasp established
AND object_grasped True→False
AND not a P4 release inside the goal
→ illegal drop, reward -5, true terminal
```

First-illegal-drop episode counts:

| Category | Episodes with first illegal drop |
|---|---:|
| Nominal success | 0 |
| Normal recovered | 0 |
| Delayed recovery | 58 / 66 |
| Failure | 132 / 156 |

The earlier count of all raw grasp-loss edges is not appropriate as an event count because SETTLING/contact chatter can create repeated True→False edges. Future SAC must terminate on the **first** illegal-drop edge.

Illegal-drop phase for the 190 affected episodes:

| Category/stage | Episodes |
|---|---:|
| Failure / CLOSE_GRIPPER | 20 |
| Failure / LIFT | 63 |
| Failure / TRANSPORT | 48 |
| Failure / DESCEND_TO_GOAL | 1 |
| Delayed / TRANSPORT | 27 |
| Delayed / DESCEND_TO_GOAL | 21 |
| Delayed / SETTLING | 10 |

Of the 58 delayed-recovery episodes with a first illegal drop, 47 later reacquired the instantaneous grasp heuristic in the recorded trajectory. Such recovery is precisely outside the proposed SAC v1 protocol and cannot be allowed to accumulate later rewards after an illegal-drop terminal.

## 5. Components that can be validated independently

Because P3 is missing, these are component-level checks rather than a complete candidate return.

### P1 EE-to-grasp signed progress

Using the candidate formula and the exact grasp target:

| Statistic | Transition reward |
|---|---:|
| Transitions | 50449 |
| Min | `-0.0752557651689895` |
| Max | `0.10047325083587265` |
| Mean | `0.05067940198838581` |
| Std | `0.018691272874806455` |
| Median | `0.056018466443135274` |
| P5 / P95 | `0.01171129161439646 / 0.07249018413504926` |

Per-episode P1 contribution:

```text
mean:   1.9667116545477508
std:    0.007285347114059729
min:    1.9206842056824387
max:    1.9959075577336332
```

The component behaves as intended on the recorded P1 trajectories: moving closer is positive and moving away is negative.

### P2 stable-grasp event

```text
Reward per event: +2
Total events: 1277
Maximum events per episode: 1
Dense reward while merely remaining grasped: 0
```

### P4 place progress

Using the exact goal position and stopping after the reconstructed place event:

| Statistic | Transition reward |
|---|---:|
| Transitions | 20000 |
| Min | `-0.12341291030993787` |
| Max | `0.2270293095207073` |
| Mean | `0.10660001744842568` |
| Std | `0.05285755978927719` |
| Median | `0.1314384946342499` |
| P5 / P95 | `-0.0018671031917641412 / 0.15294707164240334` |

Mean per-episode P4-place contribution:

```text
Nominal success:   1.9386109047993854
Normal recovered:  1.8754826885700306
Delayed recovery:  0.6793637331475174
Failure:           0.01451146200596006
```

### Place event

```text
Reward per valid event: +3
Maximum events per episode: 1
Opening gripper alone: no reward
Released outside goal: no reward
```

### P4 retreat progress

Using the exact retreat target and activating only after the place event:

| Statistic | Transition reward |
|---|---:|
| Transitions | 22820 |
| Min | `-0.053049577659749735` |
| Max | `0.10416203092193718` |
| Mean | `0.04611913958510067` |
| Std | `0.0344121520064955` |
| Median | `0.07120111894118433` |
| P5 / P95 | `-0.006034227910182158 / 0.08547915650937339` |

Mean per-episode contribution where normal retreat exists:

```text
Nominal success:  0.9772982334766104
Normal recovered: 0.9633401519921426
```

Delayed recovery and failure have zero normal retreat contribution under the frozen P1–P4 mapping.

## 6. Phase reward statistics status

| Phase | Transition count | Candidate reward status |
|---|---:|---|
| P1 | 50449 | Fully computable; statistics above |
| P2 | 15035 | Event component computable; one-time stable-grasp reward validated |
| P3 | 49168 | **N/A — object pre-place target missing** |
| P4 | 42820 normal mapped transitions | Place/event/retreat components independently computable |
| SETTLING | 3460 | Excluded from future normal P1–P4 control flow |

Whether P3 numerically dominates the trajectory, and therefore whether any phase dominates the complete candidate return, cannot be answered without the missing P3 target.

## 7. Progress and reward-hacking checks

### A. Oscillation

For a fixed phase target and denominator:

\[
\sum_t \frac{d_t-d_{t+1}}{d_0+\epsilon}
=\frac{d_{start}-d_{end}}{d_0+\epsilon}
\]

Thus forward/backward motion cancels algebraically. Signed progress does not create the positive-only oscillation exploit present with `max(progress,0)`.

In P1, 1443 adjacent sign-reversal pairs were observed. Individual pair sums can be nonzero because the two movements have unequal magnitude, but the full phase sum telescopes to its endpoints. Numerical residual relative to the same stored distance sequence is floating-point zero by construction.

P3 telescoping is **N/A** until the target is defined.

### B. Staying reward

For unchanged state:

```text
d_t - d_t+1 = 0
progress reward = 0
```

No constant living reward, time penalty, or action reward exists. Staying cannot accumulate positive progress reward.

### C. Holding an object in goal

There were 6646 recorded P4 transitions where the object was inside the goal while still grasped. Under the candidate:

- the one-time place event is not granted while grasped;
- the retreat component is not active;
- object-goal progress is zero if it remains stationary.

Therefore holding the object in goal cannot repeatedly collect `+3` or retreat reward.

### D. Early release

There were 703 raw grasp-loss edges outside P4 across the full recorded trajectories, including contact chatter/recovery. None qualifies for the place bonus. After stable grasp, the first illegal edge is instead a `-5` true terminal.

### E. Long-episode bias

Complete candidate length correlation is **N/A** because P3 and total return are undefined.

For the reference `collection_reward_v1`, nominal-success length bias is strong:

```text
length vs undiscounted return Pearson:  -0.9280914224007979
length vs undiscounted return Spearman: -0.9264332902300804
```

Longer episodes accumulate more negative distance reward. The signed-progress components that are defined depend primarily on endpoint progress and are structurally much less sensitive to trajectory length.

## 8. Event and terminal checks

| Event | Validated count/status |
|---|---|
| Stable-grasp event | 1277, at most one per episode |
| Successful place event | 1145, at most one per episode |
| Formal success event | 1144, at most one per episode |
| First illegal-drop terminal | 190 episodes under future SAC semantics |
| Failure penalty for `settling_timeout` without earlier drop | 24 episodes |
| Time-limit penalty | 0; no formal time-limit episode exists |

For episodes with an illegal drop, `reward_drop=-5` and failure termination are the same event; they must not also receive a second `reward_failure_terminal=-5` on that transition.

## 9. Delayed recovery analysis

There are 66 delayed-recovery trajectories.

### Scheme A: delayed recovery receives final +10

Relative to Scheme B, this adds:

```text
Undiscounted return difference: exactly +10 per episode
Discounted G0 difference at gamma 0.995:
  mean  5.745171298289674
  std   0.4338426848061236
  min   3.6329741745444855
  max   6.465587967553006
  p5    5.089336354090347
  p95   6.234855353089852
```

### Scheme B: delayed recovery is not SAC v1 full success

This is recommended for the candidate main protocol.

Reasons:

1. SAC v1 explicitly does not learn complex post-drop recovery.
2. 58/66 delayed recoveries contain a first illegal drop and would have terminated before the recorded final recovery.
3. The first illegal drop occurs at mean step `103.82758620689656` (min 79, max 196).
4. Allowing later SETTLING transitions and `+10` would assign reward to behavior unavailable under future online SAC termination semantics.
5. SETTLING is a collector recovery observation state, not one of the frozen four active phases.

Delayed recovery should be reported separately in reward validation and should not be automatically inserted into the formal SAC v1 replay baseline. Inclusion would require a separately specified recovery-capable MDP.

## 10. Reference collection reward

The unchanged reference is:

\[
r_t^{collection}=-d_{object,goal}+\mathbb{I}[d_{object,goal}<0.055]
\]

### Reference return at gamma 0.995

| Category | Undiscounted mean ± std | Discounted G0 mean ± std |
|---|---:|---:|
| Nominal success | `7.451985677909517 ± 3.6994210255501483` | `-0.8110383069632335 ± 3.283633615003317` |
| Normal recovered | `7.720939267885578 ± 5.50042371837945` | `-2.100339561593026 ± 4.241508490835544` |
| Delayed recovery | `-18.731948172075274 ± 6.499943190713195` | `-15.741503587734883 ± 4.27370811506683` |
| Failure | `-21.088113683100165 ± 5.223222573918792` | `-17.00405441298958 ± 3.774500058206659` |

The reference reward does not encode grasp, stable transport, legal release, or retreat. It also grants `+1` on every inside-goal step. The candidate's independently validated P1, grasp, place, retreat, one-time-event, and illegal-drop structure is qualitatively better aligned with the four-stage task. A complete numerical comparison remains impossible until P3 is precisely defined.

## 11. Requested candidate statistics that are not validly measurable

The following are **N/A**, not zero:

- total `sac_reward_v1_candidate` per transition distribution;
- total undiscounted candidate return by outcome;
- total discounted candidate (G_0) by outcome;
- candidate return versus episode-length Pearson/Spearman correlation;
- candidate pairwise ranking probabilities;
- success/failure distribution overlap under the complete candidate;
- whether P3 or another phase dominates total reward;
- whether terminal bonuses dominate or are dominated by complete dense reward;
- complete partial-task ordering (`grasp only`, `transport only`, `release without retreat`, full success).

Reporting these would require fabricating the missing P3 object target.

## 12. Structural issue and required redirect

The candidate's P3 definition is not executable from the frozen online task interface:

```text
P3 candidate: object → pre-place progress
Frozen controller: EE → above-goal target
Missing definition: object pre-place pose
```

This is not a weight imbalance. Reweighting `3.0` cannot repair an undefined distance target.

Before reward freeze, one explicit P3 definition must be chosen and frozen. Structurally valid directions include:

1. define and version an exact object pre-place pose in task geometry; or
2. reformulate P3 progress using the already frozen, directly reconstructable EE above-goal target.

The choice changes reward semantics and must be made as a task/reward design decision, not inferred from collected trajectories.

After that definition is frozen, rerun the same offline checks for:

- complete phase contributions;
- candidate return and scale;
- length bias;
- ranking and overlap;
- terminal-versus-dense balance;
- P3 telescoping consistency.

## Final conclusion

The signed-progress, one-time event, place/retreat switching, full-success, and first-illegal-drop concepts are structurally promising. P1, P2, and P4 behave consistently with their intended semantics on the formal data. Scheme B is the appropriate delayed-recovery treatment for SAC v1.

However, `sac_reward_v1_candidate` cannot be formally frozen because its P3 object pre-place target is undefined by the frozen task and cannot be reconstructed exactly from the stored data without adding a new assumption.

```text
SAC_REWARD_V1_REDIRECT_REQUIRED
```
