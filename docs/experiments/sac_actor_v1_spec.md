# SAC Actor v1 Initialization Audit and Provisional Specification

## Status

```text
Audit date: 2026-08-12
BC run: actor_bc_v1_20260812T170000Z
BC checkpoint: outputs/actor_bc/actor_bc_v1_20260812T170000Z/checkpoint_best.pt
Manifest SHA: 5a3f3cfb76f37a95ceed5155c83158529f76cef6f82ea652b9a66ed050aa7c6b
Evaluation seeds: 300000-300099
Status: SAC_ACTOR_V1_READY
```

The accepted initialization is BC-initialized full mean-path action distillation,
run `sac_actor_v1_full_distill_20260812T180000Z`. It restored deterministic
closed-loop success to 22/100 while improving all pre-retreat milestone rates.
The earlier direct-copy and frozen-trunk atanh runs remain rejected audit results.
No SAC, Critic, reward, entropy, replay, or online update was run.

## 1. BC Actor audit

The checkpoint's model class is `ActorBC`. Its exact network is:

```text
network.0 Linear(42,256)  weight [256,42], bias [256]
network.1 SiLU
network.2 Linear(256,256) weight [256,256], bias [256]
network.3 SiLU
network.4 Linear(256,256) weight [256,256], bias [256]
network.5 SiLU
network.6 Linear(256,7)   weight [7,256], bias [7]
```

State-dict keys are:

```text
network.0.weight  network.0.bias
network.2.weight  network.2.bias
network.4.weight  network.4.bias
network.6.weight  network.6.bias
```

Parameter count is `144391`. The checkpoint stores 42-D
`observation_mean`/`observation_std`; these match `normalization.json` and were
fitted only on 900 nominal-success training episodes. Constant dimensions remain
present with scale 1. BC inference computes the raw 7-D output, clips it to
`[-1,1]`, then calls the frozen `ExpertActionSpec.denormalize`. No action
mean/std is fitted from data.

## 2. Provisional SAC Actor v1 specification

| Item | Frozen value pending initialization unblock |
|---|---|
| Input | normalized `policy_state_42` |
| Architecture | `42 -> 256 -> 256 -> 256` |
| Activation | SiLU |
| Mean head | `256 -> 7` |
| Log-std head | `256 -> 7`, state-dependent |
| Actor parameters | 146190 |
| Action | continuous `normalized_action_7` |
| Squashing | tanh |
| Reparameterization | yes, `Normal.rsample()` |
| Log-std parameterization | raw state-dependent head, hard clamp |
| Log-std bounds | `[-5, 2]` |
| Proposed initial log-std | `-3` (`std=0.049787068367863944`) |
| Evaluation | deterministic `tanh(mean)` |
| Training | stochastic squashed Gaussian sample |
| Observation preprocessing | exact BC checkpoint mean/std |
| Physical mapping | frozen `ExpertActionSpec` |

Hard clamping was selected over a tanh remap because the bounds and chosen
initial value have direct log-standard-deviation meaning. It is simple and
standard; gradients are zero only at saturated bound violations. The initialized
head uses zero weights and a constant bias, but becomes state-dependent as soon
as future SAC updates modify its weights.

## 3. Three action representations

```text
u = mu(s) + exp(log_std(s)) * epsilon, epsilon ~ N(0,I)
a = tanh(u), a in (-1,1)^7
physical_action = frozen ExpertActionSpec.denormalize(a)
```

The replay buffer and future Critic must use `a`, not `u` and not joint/IK
commands. Dimension order is:

```text
[dx, dy, dz, drx, dry, drz, gripper]
```

Physical scales remain translation `0.025 m`, rotation vector `0.10 rad`, and
gripper width `[0,0.08] m` with `n_g=2*g/0.08-1`.

## 4. BC-to-SAC layer mapping

| BC key | SAC key | Treatment |
|---|---|---|
| `network.0.weight/bias` | `trunk.0.weight/bias` | exact copy |
| `network.2.weight/bias` | `trunk.2.weight/bias` | exact copy |
| `network.4.weight/bias` | `trunk.4.weight/bias` | exact copy |
| `network.6.weight/bias` | `mean_head.weight/bias` | exact initial copy, then jointly distilled with the SAC trunk in squashed-action space |
| N/A | `log_std_head.weight` | new, zeros |
| N/A | `log_std_head.bias` | new, `-3` |

The accepted mapping starts from an exact BC trunk/head copy, freezes the BC
teacher and SAC log-std head, then optimizes the SAC trunk and mean head against
the deployed clipped BC actions after SAC tanh squashing. The earlier frozen-trunk
atanh-head calibration is retained below as a rejected historical candidate.

## 5. Initialization alternatives

All 12817 fixed validation transitions were evaluated against the deployed BC
reference `clip(raw_BC,-1,1)`.

| Option | Normalized MSE | MAE | Max abs | XYZ error | Rotation error | Gripper MAE |
|---|---:|---:|---:|---:|---:|---:|
| A direct head copy | 0.003931030631065369 | 0.01874850131571293 | 0.23840558528900146 | 0.3291247 mm | 2.5866e-11 rad | 4.702915 mm |
| B trunk only, fixed seed new head | 0.022591717541217804 | 0.11204399168491364 | 0.4809015989303589 | 3.314613 mm | 0.02414941 rad | 8.782087 mm |
| C ideal atanh target | 5.835976369045551e-14 | 5.91963456031408e-8 | 1.0132789611816406e-6 | 2.0197e-7 mm | 2.2808e-12 rad | 1.62369e-5 mm |

Per-dimension MAE `[dx,dy,dz,drx,dry,drz,g]`:

```text
A: [0.0022577897, 0.0032822820, 0.0081265038,
    1.0655e-10, 1.2965e-10, 6.8903e-11, 0.1175730452]

B: [0.0580370948, 0.0589035787, 0.0866385326,
    0.0791791901, 0.1307116300, 0.1512859762, 0.2195527852]

C: [1.9627e-9, 1.9397e-9, 4.5241e-9,
    8.8340e-12, 8.5680e-12, 7.8824e-12, 4.0592e-7]
```

Option A is not behavior preserving because it changes deployed action from
`clip(a_BC)` to `tanh(a_BC)`. Option B discards the learned action mapping.
Option C was the first mathematically behavior-preserving target considered:

```text
target_mu = atanh(clip(a_BC, -1+1e-6, 1-1e-6))
freeze trunk
fit mean_head only to target_mu
```

The reported C oracle is the mathematical target, but it is not exactly
representable by one linear head over frozen BC features. The linear-head fit
used Adam, lr `1e-3`, batch 256, gradient clipping 1.0, maximum 100 epochs and
seed `20260812`. Best validation pre-tanh MSE was `0.08695729821920395` at epoch
99. This confirms that atanh is not exactly linear in the frozen BC features.

Actual calibrated action equivalence on all 12817 validation transitions:

| Metric | Direct copy | Calibrated head |
|---|---:|---:|
| Normalized MSE | 0.003931030631065369 | 0.0006077067228034139 |
| Normalized MAE | 0.01874850131571293 | 0.005390338134020567 |
| Maximum absolute error | 0.23840558528900146 | 0.5973540544509888 |
| XYZ vector error | 0.329125 mm | 0.032457 mm |
| Rotation vector error | 2.5866e-11 rad | 0.000120608 rad |
| Gripper MAE | 4.702915 mm | 1.378910 mm |

Calibrated per-dimension MAE is:

```text
[0.0003212209, 0.0009014438, 0.0006711521,
 0.0000335797, 0.0011944013, 0.0001378083, 0.0344722308]
```

The calibration is substantially better in mean error but has a larger worst
outlier, concentrated around nonlinear gripper boundary states.

## 6. Exploration initialization

Around the behavior-preserving atanh oracle center, one sampled action per each
of 12817 validation states gave:

| log_std | std | Normalized MAE | XYZ vector error | Rotation vector error | Gripper MAE |
|---:|---:|---:|---:|---:|---:|---:|
| -1 | 0.36787944 | 0.24777561 | 13.023596 mm | 0.05334190 rad | 5.339999 mm |
| -2 | 0.13533528 | 0.09702700 | 5.123607 mm | 0.02131788 rad | 2.062918 mm |
| -3 | 0.04978707 | 0.03611875 | 1.911469 mm | 0.00792637 rad | 0.771015 mm |

`-1` is too disruptive. `-2` is still large relative to the BC physical imitation
error (`0.2215 mm` validation XYZ) and this task's observed closed-loop
sensitivity. The frozen initial value is therefore `-3`, while
`[-5,2]` leaves standard SAC enough future range. This is an initialization
choice, not an entropy schedule or SAC update rule.

## 7. Log probability and Actor API

`sample_action()` returns stochastic tanh action, corrected log probability with
shape `[B,1]`, and deterministic mean action. Sampling uses `rsample()`.

The implemented stable correction is equivalent to:

```text
sum_i log N(u_i; mu_i, sigma_i)
- sum_i log(1 - tanh(u_i)^2)
```

using `2 * (log(2) - u - softplus(-2u))`; tests cover `u=0`, `u=+50`, and
`u=-50` with finite results. `deterministic_action()` returns `tanh(mu)`.
`distribution_stats()` returns pre-squash mean, bounded log-std, and std.

Tests also verify `[B,42] -> [B,7]`, `[B,1]` log probability, finite values,
strict action bounds, log-std bounds, and gradients through reparameterized
samples.

## 8. Continuous gripper

The seventh dimension remains continuous. A normalized action of `0.999` maps to
`0.07996 m`, only `0.00004 m` below fully open `0.08 m`; this is operationally
equivalent for the existing adapter. Tanh's inability to equal `+1` exactly does
not justify a hybrid action space. The direct-copy problem is much larger:
`tanh(1)=0.761594`, corresponding to `0.070464 m`, about `9.536 mm` below fully
open. Behavior-preserving atanh calibration resolves this while retaining
continuous Gaussian SAC.

## 9. Initialization-only closed-loop evaluation

Option A was evaluated deterministically with zero gradient updates and no
stochastic sampling on the exact BC seeds `300000-300099`.

| Metric | BC Actor | Direct-copy SAC | Calibrated SAC |
|---|---:|---:|---:|
| Success | 22/100 | 0/100 | 4/100 |
| Grasp | 82% | 28% | 64% |
| Lift | 54% | 0% | 21% |
| Transport | 54% | 0% | 17% |
| Release | 52% | 0% | 15% |
| Retreat | 22% | 0% | 4% |
| IK-failure episodes | 70 | 81 | 82 |
| Time-limit episodes | 8 | 19 | 14 |
| Adapter-clipping episodes | 80 | N/A | 94 |
| Drop episodes | 38 | N/A | 58 |
| Wrong-switch episodes | 77 | N/A | 98 |

The calibrated Actor used deterministic `tanh(mean)`, zero stochastic samples and
zero RL updates. It improves over direct copy but remains far below the 22%
baseline, with failures already concentrated between grasp and lift. It is rejected
by the explicit behavior-preserving acceptance criterion.

Calibration numerical state remained finite. Across validation states, pre-tanh
mean min/max were `-0.9104931951 / 7.6345767975`; absolute P95/P99/max were
`6.3929624557 / 7.2766699791 / 7.6345767975`. Gripper mean min/max were the same
extrema, with absolute P95/P99 `7.3044743538 / 7.4019646645`. This expected large
positive mean represents the near-`+1` open-gripper action; tanh remains finite.
Log-std mean/std/min/max stayed exactly `-3 / 0 / -3 / -3`, with
`exp(log_std)=0.0497870557`.

## 10. Answers to the freeze questions

1. BC output and SAC pre-tanh mean cannot be directly copied while preserving behavior.
2. Full mean-path squashed-action distillation is the accepted mapping; validation MAE is `0.0003714417`.
3. Frozen-trunk atanh calibration alone was insufficient; full trunk+mean distillation was required.
4. Initial `log_std=-3`, supported by the measured perturbation scale.
5. Log-std bounds are `[-5,2]` using a raw state-dependent head plus clamp.
6. Continuous seventh-dimension gripper is feasible; `0.999` causes only `0.04 mm` opening error.
7. Yes. Full distillation restores aggregate success to 22/100, exactly the BC baseline.
8. Yes. The initialized Gaussian Actor is ready for the later standard SAC stage.

## 11. Accepted full mean-path distillation

The formal initialization is:

```text
initialize SAC trunk and mean head by exact BC copy
freeze BC teacher in eval mode
train SAC trunk + mean head
freeze SAC log_std head
target a_bc = clip(BC(state), -1, 1)
loss = MSE(tanh(mu_sac(state)), a_bc)
```

Run `sac_actor_v1_full_distill_20260812T180000Z` used Adam, lr `1e-4`, batch
256, gradient clipping 1.0 and fixed seed `20260812`; early stopping selected
epoch 36. Validation results:

```text
normalized MSE: 6.384428843375645e-7
normalized MAE: 0.0003714416525326669
max abs error: 0.029857933521270752
XYZ physical vector error: 0.0224891 mm
rotation physical vector error: 0.0000380675 rad
gripper physical MAE: 0.0302602 mm
```

For 6217 teacher samples with gripper `>0.95`, teacher/student means were
`0.99956656 / 0.99913645`; MAE was `0.00083618`, max error `0.02985793`.
Pre-tanh gripper mean min/max were `-0.28681934 / 4.81776476`, absolute
P95/P99/max `4.51921463 / 4.61492586 / 4.81776476`; all values were finite.
Log-std remained exactly `-3` with zero parameter change.

### Closed loop and paired seeds

| Metric | BC | Full-distilled SAC |
|---|---:|---:|
| Success | 22 | 22 |
| Grasp | 82% | 87% |
| Lift | 54% | 64% |
| Transport | 54% | 62% |
| Release | 52% | 61% |
| Retreat | 22% | 22% |
| IK fallback episodes | 70 | 62 |
| Adapter clipping episodes | 80 | 74 |
| Drop episodes | 38 | 37 |
| Wrong gripper switch episodes | 77 | 74 |

Paired categories are: both success 6, BC success/SAC failure 16, BC
failure/SAC success 16, both failure 62. Thus aggregate behavior is preserved,
but individual chaotic closed-loop outcomes are not identical. Five representative
lost-success rollouts first crossed the diagnostic threshold in P1 at steps
19-38. At those crossings gripper class, IK fallback and adapter clipping still
matched; divergence was primarily accumulated translation/state difference, not
gripper switch timing. Given normalized MAE `3.71e-4` and symmetric 16-for-16
outcome changes, remaining mismatch is better characterized as environment
sensitivity to small closed-loop perturbations than inadequate aggregate policy
approximation.

## Artifacts

```text
Implementation: src/mujoco_shared_control/sac/actor.py
Unit tests: tests/test_sac_actor.py
Audit statistics: outputs/sac_actor/sac_actor_v1_initialization_audit.json
Closed-loop direct-copy evaluation: outputs/sac_actor/sac_actor_v1_direct_copy_evaluation.json
Rejected calibrated run: outputs/sac_actor/sac_actor_v1_20260812T160000Z/
Calibration checkpoint: outputs/sac_actor/sac_actor_v1_20260812T160000Z/actor_initialized.pt
Calibration equivalence: outputs/sac_actor/sac_actor_v1_20260812T160000Z/action_equivalence.json
Calibration closed loop: outputs/sac_actor/sac_actor_v1_20260812T160000Z/closed_loop_evaluation.json
Audit script: scripts/audit_sac_actor_initialization.py
Evaluation script: scripts/evaluate_sac_actor_initialization.py
Calibration script: scripts/calibrate_sac_actor_v1.py
Accepted full-distilled run: outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/
Full distillation script: scripts/distill_sac_actor_v1.py
Paired analysis: outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/paired_seed_analysis.json
Divergence analysis: outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/trajectory_divergence.json
```

```text
SAC_ACTOR_V1_READY
```
