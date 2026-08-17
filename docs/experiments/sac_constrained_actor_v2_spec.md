# Native Constrained SAC Actor v2 Specification

Status: `SAC_CONSTRAINED_ACTION_V2_READY`

This revision repairs the mismatch documented in `sac_action_consistency_audit.md`.
It preserves the failed v1 artifacts and changes neither `sac_reward_v1` nor the
frozen SAC Core hyperparameters. No SAC or Critic training was run.

## Version and artifacts

```text
Actor revision: native constrained SAC Actor v2
Run ID: sac_constrained_actor_v2_20260812T165925Z
Artifact: outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt
BC teacher: outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt
Validation split: 100 episodes / 12,817 transitions
Evaluation seeds: 300000-300099
```

## One RL action variable

`policy_action` is the agent-selected native admissible action. Actor loss,
target-Q, Q, log probability, and Replay all use it. `deployed_action` is the
post-adapter/IK command and is diagnostic only. If safety executes fallback B
after attempted action A, Replay still stores A and the failure reward/terminal
attributes the consequence to A. Trainer info exposes `policy_action`,
`deployed_action`, `adapter_projected`, and `fallback_used`.

## Architecture and transform

The network remains `42 -> 256 -> 256 -> 256`, SiLU, with separate `256 -> 7`
mean and log-std heads. Log-std is state-dependent, bounded to `[-5,2]`, and
initialized to `-3`; sampling uses `rsample()`.

For each 3-D block, where `r=||u||`, the transform is `a=tanh(r)u/r`. Translation
and rotation therefore lie in separate open unit L2 balls. Gripper uses scalar
`tanh`. Deterministic evaluation applies this transform to the mean.

For each radial block:

```text
log|det J| = log(sech(r)^2) + 2 log(tanh(r)/r)
```

The 7-D transformed log probability subtracts both radial determinants and the
scalar tanh determinant from the summed Gaussian density. Series expansions
handle zero/small radius and stable softplus handles large radius.

## BC deployed-command redistillation

Target: `project_frozen_adapter(clip(BC(policy_state_42),-1,1))`. Student trunk
and mean head were jointly optimized in output space; teacher, normalization,
action spec, and log-std head stayed frozen.

| Metric | Result |
|---|---:|
| Normalized MSE | 6.416109386009339e-7 |
| Normalized MAE | 0.00039006402948871255 |
| Max normalized error | 0.02985846996307373 |
| XYZ physical vector error | 0.000024113058316288516 m |
| Rotation physical vector error | 0.00004283602538635023 rad |
| Gripper physical MAE | 0.000029823668228345923 m |
| Translation norm max | 0.47053706645965576 |
| Rotation norm max | 0.006445207167416811 |
| Unit-ball violations | 0 / 0 |

## Log-probability and stochastic audit

Five samples for every validation state produced 64,085 samples. All actions and
log probabilities were finite; constraint violations were zero.

```text
translation norm max: 0.5795027613639832
rotation norm max:    0.24890291690826416
gripper range:        [-0.4218999743461609, 0.9998969435691833]
```

Analytic radial log determinants agree with autograd through radius 10 with max
absolute error `9.337149720067828e-9`. At radius 50 the direct determinant
underflows, while the stable analytic log determinant and gradient remain finite.

## Adapter identity audit

Across 24,008 accepted/non-fallback transitions:

```text
translation projection: 0
rotation projection:    0
gripper clipping:       0
all adapter clipping:   0
mean L2 action delta:   1.4503297902519368e-12
max L2 action delta:    1.1900708037930496e-8
```

The adapter is a numerical identity in the normal path and remains a safety guard.

## Initialization-only closed loop

No policy/RL update was performed.

| Metric | Constrained v2 | Reference |
|---|---:|---:|
| Success | 35 / 100 | 22 / 100 |
| Grasp | 86% | 87% |
| Lift | 68% | 64% |
| Transport | 68% | 62% |
| Release | 64% | 61% |
| Retreat | 35% | 22% |

This clears the declared 15/100 lower bound and preserves BC-derived behavior.
The teacher target is now explicitly the adapter's admissible deployed command.

## Machine-readable evidence

The run directory contains `actor_initialized.pt`, `distillation_config.json`,
`training_history.json`, `action_equivalence.json`, `closed_loop_evaluation.json`,
`logprob_validation.json`, and `adapter_identity_audit.json`. Old v1 artifacts and
the mismatch audit remain unchanged.
