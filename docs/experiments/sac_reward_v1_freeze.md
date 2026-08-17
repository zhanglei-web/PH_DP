# SAC Reward v1 Freeze

## Identification

```text
Reward version: sac_reward_v1
Task: four-phase MuJoCo pick-box SAC Expert task
Discount gamma: 0.995
Candidate validation status: SAC_REWARD_V1_CANDIDATE_VALIDATED
```

This document freezes the only reward definition intended for the first online
SAC Expert baseline. `gamma=0.995` is a return/training parameter and is not part
of the instantaneous reward implementation.

## Frozen phase definition

| Phase | Meaning | Ground-truth transition |
|---|---|---|
| P1 Pre-Grasp | EE approaches the grasp pose | EE reaches frozen grasp tolerance |
| P2 Grasp | Close gripper and establish stable grasp | Eight consecutive grasped steps |
| P3 Transport | Lift and transport EE plus object | EE reaches frozen `above_goal` |
| P4 Place & Retreat | Descend, release stably, and withdraw EE | Stable legal release and retreat complete |

Lift, release and retreat are milestones/substates, not additional phases.
`SACPickPlaceProtocol` supplies these phases deterministically from simulator task
state. It does not use Rule Expert stages at online training time, a learned
phase classifier, or a recovery phase.

## Exact geometry

```text
p_grasp      = p_object + [0, 0, 0.012] m
p_above_goal = p_goal   + [0, 0, 0.16] m
p_retreat    = p_goal   + [0, 0, 0.16] m
goal tolerance       = 0.055 m
position tolerance   = 0.008 m
stable grasp window  = 8 steps
stable release window = 4 steps
epsilon              = 1e-6
```

The `above_goal` and retreat definitions exactly reuse frozen Rule Expert v1
geometry. No object pre-place target is defined or inferred.

## Reward equation

At every step:

```text
r_t = phase_progress + event + terminal
```

All progress terms use signed, phase-entry-normalized progress:

```text
weight * (d_before - d_after) / (d_phase,0 + 1e-6)
```

| Component | Exact value |
|---|---:|
| P1 EE-to-grasp progress | 2.0 × normalized signed progress |
| P2 first stable-grasp event | +2.0 |
| P3 EE-to-`above_goal` progress | 3.0 × normalized signed progress |
| P4 object-to-goal progress before release | 2.0 × normalized signed progress |
| First stable legal place/release event | +3.0 |
| P4 EE-to-retreat progress after release | 1.0 × normalized signed progress |
| Full task success | +10.0 |
| Illegal drop / true failure | -5.0 |
| Time/action/collision/jerk penalty | 0 |

P3 contains no completion bonus. Object-goal progress stops once successful
release is established; retreat progress then becomes active.

## Event semantics

Stable grasp reuses the frozen eight-consecutive-step grasp protocol and is
rewarded once. Merely keeping `object_grasped=True` does not repeatedly reward it.

Legal release requires the object to be released inside the goal and remain
released and inside for four consecutive steps. The `+3` is emitted once. Opening
outside the goal, early release, or merely entering the goal does not receive it.

After stable grasp, `object_grasped: True -> False` outside a legal P4 release is
an `illegal_drop`: reward `-5`, `terminated=True`, `truncated=False`. The episode
ends immediately. SETTLING, delayed recovery and re-grasp are not part of SAC v1.
Formal delayed-recovery data is interpreted using Scheme B: 58 episodes stop at
illegal drop and the remaining 8 stop at explicit failure.

Full success requires stable grasp, transport, stable legal place/release and
safe retreat. It receives `+10` once and returns `terminated=True`.

IK safety failure, invalid physical state, explicit task failure, and equivalent
task-wrapper safety terminals receive `-5` and `terminated=True`. A pure horizon
returns `terminated=False`, `truncated=True`, and no terminal penalty. A true
failure takes precedence if it coincides with the horizon.

## Runtime state and API

`SACRewardV1.reset()` clears phase-entry distances, all one-shot event flags,
stable-grasp/release state and terminal state. Accumulating reward after a terminal
or truncation raises an error, preventing delayed-recovery leakage.

The normal Gymnasium environment API remains:

```text
observation, reward, terminated, truncated, info
```

The existing collection reward remains the default so Rule Expert v1 data
reproducibility is unchanged. Online SAC must opt in explicitly with:

```python
PickPlaceEnv(reward_version="sac_reward_v1")
```

`info["reward_components"]` exposes all progress/event/terminal components plus
`reward_total`; `info` also exposes phase names, event state, and termination
reason. External adapters may pass a true safety failure and its reason into
`step()`.

## Validation basis

Validation report:

```text
docs/experiments/sac_reward_v1_candidate_validation_v2.md
```

Candidate validation artifacts:

```text
outputs/reward_validation/sac_reward_v1_candidate_v2_20260812T143907Z/
```

Formal implementation regression:

```text
outputs/reward_validation/sac_reward_v1_regression_20260812T150245Z/
```

Headline evidence:

```text
Episodes: 1300
SAC-v1 semantic transitions: 157302
P3 telescoping max error: 1.11e-16
Nominal mean discounted return: 14.247517
Failure mean discounted return: -0.096547
P(nominal > failure): 1.0

Official implementation reward max absolute difference: 1.3114509478384662e-15
Terminal decision mismatch count: 0
Phase mismatch count: 0
Illegal-drop mismatch count: 0
```

The implementation validated here is `SACRewardV1` in
`src/mujoco_shared_control/tasks/sac_reward.py`. The historical v2 validation
script name denotes the second validation pass; the frozen reward version remains
`sac_reward_v1`.

```text
SAC_REWARD_V1_FROZEN
```
