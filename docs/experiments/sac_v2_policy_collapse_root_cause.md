# SAC v2 Policy-Collapse Root-Cause Record

## Status and scope

This document records the evidence available after the native constrained-action
repair. It separates confirmed observations from the next controlled intervention.
It does **not** reinterpret a failed run as a successful SAC result, and it does not
change any frozen environment, reward, action, replay, or SAC-Core semantics.

Relevant runs and audits:

```text
Native constrained Actor artifact:
outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/

Clean constrained SAC run:
outputs/sac_training/sac_v2_constrained_clean_sanity_20260813T033000Z/

Constrained critic-only warmup control:
outputs/sac_training/sac_v2_constrained_critic_warmup_20260813T040000Z/

Earlier action-semantics audit:
docs/experiments/sac_action_consistency_audit.md
```

## 1. What the constrained-action repair solved

The v2 Actor makes one RL random variable serve all algorithmic consumers:

```text
policy action
= Actor-Q action
= Target-Q action
= log-prob random variable
= Replay action
```

The adapter is an identity guard on normal transitions. Across the 30,000-step
clean v2 run, translation and rotation adapter projection counts were both zero,
the mean non-fallback adapter difference was `3.09426573634661e-13`, and Replay
policy mismatches were zero. The former out-of-support `Q(raw) > Q(projected)`
exploitation route is therefore absent.

This repair materially improved numerical behavior. At the end of the clean v2
run, Q1 mean/std/min/max were `3.5946785295 / 4.6710790975 /
-8.0415532589 / 21.3943095045`, target mean/std were
`3.5948290241 / 4.6909161150`, and Critic loss was `0.3242303185`.
The old componentwise-tanh experiment had reached Q values in the thousands.

The repair did not preserve the initialized behavior once Actor updates began:

| Env step | Actor updates | Success | Grasp | Lift | Transport | Release | Retreat | Action MAE from initialization |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 6/20 | 85% | 65% | 65% | 55% | 30% | 0 |
| 10,000 | 0 | 6/20 | 85% | 65% | 65% | 55% | 30% | 0 |
| 15,000 | 5,000 | 0/20 | 15% | 0% | 0% | 0% | 0% | 0.2935163081 |
| 20,000 | 10,000 | 0/20 | 5% | 0% | 0% | 0% | 0% | 0.3392893970 |
| 25,000 | 15,000 | 0/20 | 20% | 0% | 0% | 0% | 0% | 0.3783040643 |
| 30,000 | 20,000 | 0/20 | 0% | 0% | 0% | 0% | 0% | 0.3875429034 |

The action-semantics mismatch was therefore a real cause of Q explosion, but it
was not the only cause of early BC policy forgetting.

## 2. Entropy and temperature audit

### 2.1 Formula audit

The constrained Actor uses the correct transformed density:

\[
\log \pi_\theta(a\mid s)
=
\log \mathcal N(u;\mu_\theta,\sigma_\theta)
-\log |\det J_T(u)|,
\]

where `T` is the two radial squashes plus scalar gripper tanh. The analytic
Jacobian was previously checked against autograd with maximum absolute error
`9.33715e-9`. The Actor loss, alpha loss, and their signs match standard SAC:

\[
L_\pi=\mathbb E[\alpha\log\pi_\theta(a\mid s)-Q_{\min}(s,a)],
\]

\[
L_\alpha=-\mathbb E[\log\alpha
(\log\pi_\theta(a\mid s)+H_{target})_{\mathrm{stopgrad}}].
\]

With `H_target=-7`, the alpha equilibrium condition is
`E[log pi] = 7`. No sign or Jacobian bug was found.

### 2.2 The frozen target is far from the initialized policy

Monte-Carlo transformed-density measurements give:

| State population | Initial `E[log pi]` | Initial entropy `-E[log pi]` | Target entropy |
|---|---:|---:|---:|
| 12,817 fixed BC validation states | 14.7158203 | -14.7158203 | -7 |
| First 10,000 online COLLECT states | 16.7136250 | -16.7136250 | -7 |

On the actual COLLECT-state population, automatic entropy tuning therefore asks
the warm-start policy to increase entropy by approximately `9.7136` nats.
The initial transformed log-density decomposes as:

```text
translation E[log pi]: 5.325875
rotation E[log pi]:    4.793827
gripper E[log pi]:     6.593924
```

The gripper contribution is largest. Its mean log-Jacobian contribution was
`-5.010846`, consistent with the BC teacher frequently operating near the
open-gripper boundary. This is not a log-prob error: it is the expected density
geometry of a concentrated, boundary-adjacent transformed policy.

The entropy term also shifts Critic targets before any Actor update. Initially,
`-alpha * E[log pi]` is approximately `-1.6714` per bootstrapped transition on
COLLECT states. Ignoring rewards, terminations, and state variation only for scale
intuition, its infinite-horizon contribution at `gamma=0.995` would be about
`-334`. This explains why a frozen Actor and finite Critic can still acquire a
large negative soft-Q scale.

### 2.3 First-update gradient audit

At the initial Actor/Critic state, on the same normalized state batch:

| Actor-loss component | Total gradient norm |
|---|---:|
| `alpha * log pi` | 1.500180 |
| `-Q_min` | 0.038511 |

The entropy gradient is about `39` times the random-Critic policy gradient.
Parameter-group norms were:

| Parameter group | Entropy gradient | Q gradient |
|---|---:|---:|
| Trunk | 0.592512 | 0.008199 |
| Mean head | 0.874681 | 0.037628 |
| Log-std head | 1.065084 | 0.000228 |

Consequently, freezing only the log-std head would not preserve the policy:
the entropy objective also strongly moves the trunk and mean head.

### 2.4 Observed feedback trajectory

The clean v2 run follows the behavior predicted by the gradient audit:

```text
10k: log_std mean = -3.000; alpha = 0.100000
11k: log_std mean = -0.266820; alpha = 0.086597; E[log pi] = -2.370023
20k: log_std mean = -0.819105; alpha = 0.006292; E[log pi] = 0.260629
27k: log_std mean = -1.876216; alpha = 0.001791; E[log pi] = 7.188850
30k: log_std mean = -2.059785; alpha = 0.002637; E[log pi] = 7.233913
```

The policy first expands far away from its BC initialization; alpha reacts only
after that expansion. Later `E[log pi]` approaches the configured equilibrium near
7. Thus the falling alpha is a consequence of the initial entropy mismatch and
feedback lag, rather than evidence of a reversed alpha gradient.

The values above are stochastic training-batch estimates. Evaluation files also
report density at the deterministic mean; that diagnostic is not used as a
Monte-Carlo entropy estimate here.

## 3. Critic-only warmup controls

The original v1 critic-only warmup is not a clean test of the present hypothesis
because it still had the raw-policy/projected-Replay mismatch. It preserved the
Actor only while explicitly frozen, then reached 0/20 after 5,000 Actor updates.

A constrained-action control removes that confound. Its schedule was 10k COLLECT,
10k Critic-only warmup, then Full SAC. Actor, log-std, and alpha remained exactly
unchanged through step 20k, and deterministic success remained 6/20. Nevertheless:

| Step | Stage | Q1 mean | Target std | Critic loss |
|---:|---|---:|---:|---:|
| 11,000 | Critic warmup | -5.097402 | 1.945407 | 1.731926 |
| 15,000 | Critic warmup | -38.419411 | 11.124445 | 10.950986 |
| 19,000 | Critic warmup | -70.921414 | 17.566454 | 17.133452 |
| 20,000 | Critic warmup | -78.873692 | 19.138811 | 17.690150 |

Once Full SAC began, success was 0/20 at 25k and 30k. Action MAE was
`0.3914427459` at 25k and `0.3622735441` at 30k. Q1 mean/target std grew to
`72.522881 / 100.369734` at 25k and `207.351743 / 221.106809` at 30k.

This control shows that more Critic updates alone do not protect the BC policy.
It also supports the soft-target-scale diagnosis: with the concentrated Actor and
fixed `alpha=0.1`, Critic-only warmup repeatedly bootstraps a substantial negative
entropy term. The test does not prove that entropy is the only remaining issue,
but it rules out “the Critic merely needs 10k more updates” as a sufficient fix.

## 4. Root-cause assessment

The evidence supports this hierarchy:

1. **Confirmed and repaired:** v1 mixed raw policy actions with projected Replay
   actions. This enabled out-of-support Q exploitation and the largest Q explosion.
2. **Confirmed remaining warm-start conflict:** the standard entropy objective at
   `alpha=0.1`, `target_entropy=-7` is initially much stronger than the Q policy
   gradient and deliberately moves a highly concentrated BC policy away from its
   behavior.
3. **Confirmed insufficiency of a delay-only fix:** critic-only warmup preserves
   behavior only while Actor updates are disabled; it does not produce a stable
   handoff to unconstrained Full SAC.
4. **Not established as a sole cause:** Critic approximation error, online state
   distribution shift, and task sensitivity can still contribute. The next test
   must therefore be controlled and must not claim causality before its result.

## 5. Proposed minimal intervention: decaying KL policy anchor

### 5.1 Frozen reference policy

Let `pi_0` be the frozen native constrained Actor v2 immediately after successful
BC-to-SAC redistillation. Let `pi_theta` be the current SAC Actor. Both share the
same observation normalization and the same invertible transform
`T: R^7 -> B3 x B3 x (-1,1)`.

Their pre-transform distributions are diagonal Gaussians:

\[
\pi_\theta^u=\mathcal N(\mu_\theta,\operatorname{diag}(\sigma_\theta^2)),
\qquad
\pi_0^u=\mathcal N(\mu_0,\operatorname{diag}(\sigma_0^2)).
\]

Because the same bijection is applied to both distributions, KL divergence is
invariant under the transform:

\[
D_{KL}(\pi_\theta^a\Vert\pi_0^a)
=D_{KL}(\pi_\theta^u\Vert\pi_0^u).
\]

The dimension-normalized anchor is therefore computed exactly and stably in base
Gaussian space:

\[
L_{anchor}=
\mathbb E_{s\sim D_{BC,train}}
\left[
\frac{1}{7}\sum_{i=1}^{7}
\left(
\log\frac{\sigma_{0,i}}{\sigma_{\theta,i}}
+\frac{\sigma_{\theta,i}^2+(\mu_{\theta,i}-\mu_{0,i})^2}
{2\sigma_{0,i}^2}
-\frac{1}{2}
\right)
\right].
\]

The direction is `KL(current || initial)`. This penalizes both mean-path drift and
premature variance expansion, including the gripper-boundary failure mode.

### 5.2 Actor objective and state source

The proposed Actor-only objective is:

\[
L_{actor,total}=
\mathbb E_{s\sim Replay}
[\alpha\log\pi_\theta(a\mid s)-Q_{min}(s,a)]
+\lambda(k)L_{anchor}.
\]

`D_BC,train` is strictly the existing fixed Actor training split:

```text
manifest: manifests/rule_expert_v1_formal.json
category: nominal success only
episodes: 900
transitions: 115,021
observation: policy_state_42
```

Validation, perturbed, recovered, failed, BC evaluation, and SAC rollout states
are excluded from the anchor dataset. The BC transitions are used only as a state
population for the frozen-policy KL; expert actions are not inserted into Replay
and no offline Critic target is constructed.

### 5.3 Proposed decay schedule

For Actor-update index `k`, the first controlled candidate is:

\[
\lambda(k)=
\begin{cases}
0.1, & k<50{,}000,\\
0.1\left(1-\frac{k-50{,}000}{150{,}000}\right),
&50{,}000\le k<200{,}000,\\
0, & k\ge 200{,}000.
\end{cases}
\]

The anchor is temporary: after 200k Actor updates, the objective is the original
standard SAC objective. `lambda=0.1` and the decay boundaries are a single proposed
candidate, not validated hyperparameters and not a sweep result.

### 5.4 Why this is the selected next test

- It directly constrains the quantity observed to fail: the complete stochastic
  policy, including both mean and standard deviation.
- Unlike deterministic action MSE, it preserves uncertainty semantics and exposes
  a dimensionless, monitorable divergence.
- Unlike permanently freezing Actor components, it permits policy improvement.
- Unlike expert Replay or Critic pretraining, it does not alter the online RL data
  distribution or Q-learning target.
- Its decay makes the intervention a warm-start trust region rather than a new
  permanent task objective.

## 6. Frozen semantics that remain unchanged

The proposed controlled experiment must not modify:

```text
sac_reward_v1 or its terminal/truncation rules
policy_state_42 and frozen observation normalization
native radial constrained action transform or Jacobian/log-prob
7-D continuous action semantics
Replay action = attempted policy_action
empty online Replay initialization
Twin Critic architecture and random initialization
target Critic and bootstrap mask = 1 - terminated
gamma = 0.995
tau = 0.005
actor_lr = critic_lr = alpha_lr = 3e-4
batch_size = 256
replay_capacity = 1,000,000
learning_starts = 10,000
target_entropy = -7
alpha_init = 0.1
UTD = 1
Actor/Critic/alpha update frequency = 1
training and evaluation seed partitions
```

The frozen reference Actor must remain in eval mode with `requires_grad=False` and
must be protected by a parameter checksum. The KL term may update only the current
Actor. Critic, target Critic, and alpha losses must remain byte-for-byte equivalent
to their current definitions.

## 7. Required implementation guards and measurements

Before interpreting a future run, tests should establish:

- KL is finite, non-negative up to numerical tolerance, and zero for identical
  current/reference policies.
- The reference Actor and normalization never change.
- Only Actor gradients receive the anchor term.
- The schedule uses Actor-update count, is checkpointed, and resumes without a
  discontinuity.
- Anchor states come only from the fixed 900-episode training split and never enter
  Replay.
- Training logs keep SAC Actor loss, unweighted KL, `lambda`, weighted anchor loss,
  mean/log-std drift, alpha, Q, and task metrics separate.

The controlled 30k comparison should retain the clean v2 schedule and seeds. This
isolates the policy anchor as the only algorithmic difference.

## 8. Pending experiment result

The following fields are deliberately unresolved:

```text
Anchor run ID:                         TBD
10k / 15k / 20k / 25k / 30k success: TBD
Final action MAE:                      TBD
Final gripper physical MAE:            TBD
Final KL(current || initial):          TBD
Q/target numerical health:             TBD
Alpha/log-std trajectory:              TBD
Adapter/Replay consistency:            TBD
Checkpoint/resume result:              TBD
Conclusion:                            TBD
```

No claim is made yet that the KL anchor solves policy collapse. The evidence above
justifies it as the next minimal controlled experiment.
