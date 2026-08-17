# SAC Action Consistency Audit

Status: `SAC_ACTION_SEMANTICS_MISMATCH_CONFIRMED`

This is a read-only code/data audit. It did not modify reward, networks, replay
schema, action limits, optimization parameters, or checkpoints, and it ran no new
training experiment.

Machine-readable results:

```text
outputs/sac_training/sac_action_consistency_audit_20260813T020000Z/
  action_flow.json
  action_projection_stats.json
  q_raw_vs_projected.json
```

## 1. Executed code path

| Step | Code variable | Shape/range | Transformation | Q input | Replay |
|---|---|---|---|---|---|
| Gaussian | `pre_squash` | `[B,7]`, real | `Normal(mean,std).rsample()` | no | no |
| Policy | `action` / `actor_action` | `[B,7]`, each component `(-1,1)` | componentwise `tanh(pre_squash)` | **yes, raw** | no |
| Physical request | `spec.denormalize(actor_action)` | `[7]` | xyz ×0.025 m, rot ×0.10 rad, gripper affine to `[0,.08]` | no | no |
| Adapter limit | `adapted.clipped` | `[7]` physical | radial xyz and rotation L2 projection; scalar gripper clip | no | indirectly |
| IK command | `adapted.joint_target` | `[8]` | Cartesian target integration + IK, or safe hold fallback | no | no |
| Deployed policy action | `replay_action` | `[7]`, admissible normalized | accepted: `adapted.normalized`; fallback: zero Cartesian delta + safe gripper | no | **yes** |
| Critic training | `batch.action` | `[B,7]` | loaded from Replay | **yes, deployed** | source |

Code evidence:

- `actor.py:80-88` samples `u`, applies componentwise tanh, and computes log-prob.
- `agent.py:148-156` sends the raw sampled next action directly to target Q.
- `agent.py:190-197` sends the raw sampled policy action directly to online Q.
- `trainer.py:310-322` denormalizes raw action, adapts it, then stores
  `_actual_normalized_action` rather than raw policy action.
- `expert_command_adapter.py:61-70` applies the physical vector-norm limits.
- `replay_buffer.py:72-85` writes the supplied deployed normalized action.

### IK fallback boundary

For an accepted IK command, Replay and the adapter-limited Cartesian command are
identical after normalization. If IK rejects the request, the environment holds its
previous joint/Cartesian target while applying a safe gripper command; Trainer
records the equivalent policy-level command `[0,0,0,0,0,0,g_safe]`. Thus Replay is
consistent with the deployed policy-level action abstraction, not with the rejected
raw request. The environment itself ultimately receives an 8-D joint target; that
joint command is intentionally not the Critic action space.

Historical checkpoints did not retain each raw requested action, clipping flag, or
adapter state. Therefore exact historical raw-vs-deployed deltas including IK
fallback cannot be reconstructed. The policy statistics below resample the saved
Actor on recorded states with fixed seeds and apply the exact deterministic radial
projection. Replay boundary counts are separately labeled as projection proxies.

## 2. Actual admissible normalized set

`ExpertActionSpec` first clips each normalized component to `[-1,1]` when
denormalizing. The adapter then applies the following physical limits:

```text
t = 0.025 * a_xyz
if ||t||2 > 0.025: t <- 0.025 * t / ||t||2

w = 0.10 * a_rot
if ||w||2 > 0.10: w <- 0.10 * w / ||w||2

g <- clip(g, 0, 0.08)
```

In normalized space this is exactly:

```text
P(v) = v / max(1, ||v||2), separately for xyz and rotation
g remains in [-1,1]
```

Therefore the actual 7-D admissible set is:

```text
||a_xyz||2 <= 1
||a_rot||2 <= 1
a_gripper in [-1,1]
```

Componentwise tanh only guarantees each coordinate lies in `(-1,1)`; a 3-D group
can reach norm nearly `sqrt(3)`. The adapter transform is continuous and piecewise
differentiable (not differentiable exactly at radius 1), many-to-one outside the
unit ball, and non-invertible globally because every point on an exterior ray maps
to the same boundary point.

## 3. Transition-level policy projection statistics

Each row is one reproducible stochastic policy sample per recorded state. It
excludes history-dependent IK fallback and isolates vector-norm projection.

| Stage/checkpoint | Samples | xyz projected | rotation projected | gripper projected | any projected |
|---|---:|---:|---:|---:|---:|
| Initial Actor proxy | 12,817 | 701 (5.4693%) | 0 | 0 | 701 (5.4693%) |
| COLLECT states 1-10k | 10,000 | 544 (5.4400%) | 0 | 0 | 544 (5.4400%) |
| WARMUP states 10,001-20k | 10,000 | 643 (6.4300%) | 0 | 0 | 643 (6.4300%) |
| FULL SAC 25k | 25,000 | 21,715 (86.8600%) | 17,903 (71.6120%) | 0 | 23,645 (94.5800%) |

Raw-to-projected 7-D L2 distance:

| Stage | Mean | Median | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Initial | 0.009999 | 0 | 0 | 0.008428 | 0.343461 | 0.729993 |
| COLLECT | 0.009532 | 0 | 0 | 0.007505 | 0.330691 | 0.728418 |
| WARMUP | 0.011006 | 0 | 0 | 0.026975 | 0.360577 | 0.729574 |
| FULL SAC 25k | 0.444945 | 0.450199 | 0.759392 | 0.830800 | 0.952520 | 1.033494 |

Mean per-dimension absolute difference `[dx,dy,dz,drx,dry,drz,g]`:

```text
Initial:
[0.005075, 0.005461, 0.005649, 0, 0, 0, 0]

COLLECT:
[0.004837, 0.005251, 0.005295, 0, 0, 0, 0]

WARMUP:
[0.005352, 0.006133, 0.006424, 0, 0, 0, 0]

FULL SAC 25k:
[0.186870, 0.147597, 0.209526, 0.140281, 0.110333, 0.130819, 0]
```

The 25k raw norms themselves have means `1.3014` (translation) and `1.1584`
(rotation), with P95 `1.6749/1.6368`, demonstrating that Actor updates strongly
favor points outside both executable balls.

### Replay transition support

Every saved Replay action satisfies both unit-ball constraints and gripper bounds:

| Replay checkpoint | Transitions | xyz > 1 | rotation > 1 | gripper invalid |
|---|---:|---:|---:|---:|
| Old full-SAC 20k | 20,000 | 0 | 0 | 0 |
| Warmup/full-SAC 25k | 25,000 | 0 | 0 | 0 |

Actions lying numerically on either unit-ball boundary are projection signatures,
not exact clipping counts because an unprojected action can theoretically land near
the boundary. The proxy is 7,353/20,000 (36.765%) for the old 20k Replay and
4,704/25,000 (18.816%) for the warmup run. Exact episode-level clipping figures
from earlier reports cannot substitute for transition-level raw clipping, and the
missing historical raw actions prevent recovering exact values retroactively.

## 4. Q(raw) versus Q(projected)

The audit uses saved Replay states, reproducible stochastic Actor samples, the exact
normalized radial projection, and the online Critic from each checkpoint.

### Old simultaneous-update checkpoint at 20k

Policy samples require projection in 18,416/20,000 states (92.08%).

| Delta | Mean | Median | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Q1(raw)-Q1(projected) | 0.0277 | 0.0121 | 0.0670 | 0.1024 | 0.2660 | 2.1984 |
| Q2(raw)-Q2(projected) | 0.0286 | 0.0128 | 0.0664 | 0.1008 | 0.2693 | 3.9644 |

The raw-action advantage is already positive, though small at this checkpoint.

### Critic-warmup run, 25k checkpoint

Policy samples require projection in 23,718/25,000 states (94.872%).

| Delta | Mean | Median | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Q1(raw)-Q1(projected) | 50.6670 | 26.5699 | 135.7513 | 166.2560 | 230.3173 | 531.4243 |
| Q2(raw)-Q2(projected) | 46.7090 | 24.9777 | 122.1244 | 150.6310 | 237.9754 | 597.1681 |

On only the out-of-ball subset, the mean advantages increase to `53.4057` (Q1)
and `49.2337` (Q2). These are direct evidence that the Actor can optimize toward
raw actions that the environment projects away, while Critic regression data remain
inside the admissible set.

The 20k warmup Critic state was not retained: `latest.pt` was subsequently replaced
at 25k and `best.pt` is the unchanged 10k model. Therefore exact Q(raw)-Q(projected)
at the warmup boundary cannot be reconstructed. What is directly known is:

- frozen-Actor target actions were outside the translation ball 5-6% of the time;
- target Q was queried on those raw actions while Replay regression used projected actions;
- logged warmup Q mean/variance/loss continuously drifted.

The mismatch is thus present during warmup, but this audit cannot prove it is the
sole cause of the negative Q drift. In particular, the positive squashed-policy
log-density and fixed entropy term can also move the bootstrapped soft-value scale.
For the later positive Q explosion and Actor exploitation, the 25k direct Q evidence
is strong.

## 5. Probability semantics

`log_prob` is the density of the **raw componentwise-tanh action**. Its Jacobian
correction is only:

```text
sum_i log(1 - tanh(u_i)^2)
```

It contains no correction for the later radial adapter projection or IK fallback.
Because radial clipping is many-to-one and puts exterior probability mass on the
unit-ball boundary, the deployed action is not distributed according to the logged
continuous tanh-Gaussian density. Thus entropy optimization, Actor Q evaluation,
environment execution, and Replay regression do not share one action random variable.

## 6. Required answers

**Q1. Is Replay action consistent with the environment action?**

Yes at the defined 7-D policy-command level. Accepted commands store the exact
adapter-projected normalized action; IK fallback stores the equivalent hold delta
and safe gripper. Replay intentionally does not store the 8-D IK joint target.

**Q2. Does Actor loss evaluate the action actually executed?**

No. It evaluates `Q(s, raw componentwise-tanh action)` without radial projection or
IK fallback semantics.

**Q3. Does the TD target use an executable next action?**

No. It queries target Q with the raw componentwise-tanh sample. Translation and
rotation can lie outside their unit L2 balls.

**Q4. Does log-prob correspond to the executed action?**

No whenever adapter projection/fallback changes the action. It is the density of
the raw componentwise-tanh action only.

**Q5. Is there a policy→projection→environment versus Critic/entropy mismatch?**

Yes, explicitly and structurally.

**Q6. Is it a credible principal cause of Q drift/explosion?**

It is a major, directly evidenced candidate for Actor exploitation and the later Q
explosion: 94.9% of 25k policy samples require projection and Q(raw) exceeds
Q(projected) by about 47-51 on average. It is not proven to be the sole cause, and
the saved artifacts are insufficient to attribute all Stage-B negative Q drift to
this mismatch alone.

## 7. Next-stage design direction (not implemented)

A native constrained policy should make sampled policy action, Critic action,
Replay action, and normally executed adapter action identical. For each 3-D
pre-squash Gaussian vector `u`, use radial squashing:

```text
r = ||u||2
a = tanh(r) * u/r                 if r > 0
a = u                             in the stable r -> 0 limit
```

Apply this independently to translation and rotation. Keep scalar tanh for the
gripper. Then both vector norms are strictly below one and ordinary adapter radial
clipping should be approximately zero, excluding IK feasibility failures.

For dimension `d=3`, radial output radius is `rho=tanh(r)`. The map is one-to-one
from `R^3` to the open unit ball, with inverse:

```text
u = atanh(||a||) * a/||a||
```

Its Jacobian determinant is:

```text
|det J| = sech(r)^2 * (tanh(r)/r)^(d-1)
        = sech(r)^2 * (tanh(r)/r)^2   for d=3
```

and therefore:

```text
log|det J| = log(1-tanh(r)^2) + 2[log(tanh(r))-log(r)]
```

The `r -> 0` limit is zero log-correction (determinant one); implementation must use
stable series/`log1p` forms near zero and stable softplus identities at large r.
The full 7-D log determinant is the sum of translation radial, rotation radial, and
scalar gripper tanh corrections.

This transform is differentiable and invertible on the open ball, unlike hard
projection. A new behavior-preserving conversion would also be required: the
existing componentwise-tanh SAC Actor cannot simply copy its heads. BC deployed
actions should first be mapped through the radial inverse for xyz/rotation and
scalar atanh for gripper, followed by supervised full mean-path distillation in the
final constrained action space. Stochastic log-std behavior and deterministic
closed-loop equivalence would need fresh audits before any SAC training.

No part of this proposed design was implemented in this audit.

```text
SAC_ACTION_SEMANTICS_MISMATCH_CONFIRMED
```
