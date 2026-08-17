# SAC v2 early-collapse resolution

Date: 2026-08-13  
Validated run: `sac_v2_final_safe_trust_20260813T100000Z`  
Status: bounded 30k sanity passed; no formal long-run training was started.

## Root cause

The native constrained action repair removed the original policy/Replay mismatch,
but it did not remove early policy forgetting. The remaining failure had three
measured causes:

1. The first 10k stochastic rollout contained no complete success, so a random
   Critic initially saw only failed behavior.
2. The initialized policy had mean `log pi = 16.713625`, while the generic
   target entropy `-7` and `alpha=0.1` made the entropy gradient about 39 times
   larger than the initial Q gradient.
3. A soft anchor on formal demonstration states did not constrain states reached
   online. Small demonstration-state error hid large online-state and parameter
   drift.

A separate implementation bug was also fixed: the Bellman target used a fixed
`alpha_init` while Actor and alpha losses used learned alpha. There is now one
temperature only: `self.alpha.detach()` in both the Critic target and Actor loss.

## Implemented safe warm start

- Steps 0-10k: deterministic initialized-Actor interaction, Replay collection,
  no optimizer update.
- Steps 10k-20k: stochastic constrained interaction and Critic-only updates.
  Actor and alpha remain unchanged. This combines successful deterministic
  transitions with stochastic action support.
- Steps 20k+: full SAC.
- Entropy warm start: `alpha_init=0.0025`, target entropy `-16.7` through 50k
  Actor updates, then configured to transition to `-7` by 200k updates.
- Initial-policy protection: KL on a fixed mixture of formal train and current
  Replay states, plus an initial-centered per-module parameter radius. For this
  contact-sensitive sanity run:
  - `KL(current || initial) / 7 <= 1e-5`
  - group relative parameter radius `<= 1e-4`
  - proposal updates are backtracked along `initial -> proposal`.

The trust-region value is an empirically validated warm-start safety bound. It
must not silently be treated as a proven long-horizon optimum; relaxation during
a future formal run requires its own bounded validation.

## 30k result

| Step | Success | Grasp | Lift | Transport | Release | Retreat | Actor MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6/20 | 85% | 65% | 65% | 55% | 30% | 0 |
| 10k | 6/20 | 85% | 65% | 65% | 55% | 30% | 0 |
| 15k | 6/20 | 85% | 65% | 65% | 55% | 30% | 0 |
| 20k | 6/20 | 85% | 65% | 65% | 55% | 30% | 0 |
| 25k | 6/20 | 90% | 70% | 70% | 65% | 30% | 0.0000680977 |
| 30k | 10/20 | 90% | 80% | 75% | 70% | 50% | 0.0000518655 |

Final numerical health (last 1k-step window):

- Critic loss: `0.3591892442`
- Q1/Q2 mean: `-4.85718664 / -4.85725728`
- Q1/Q2 std: `3.27250139 / 3.27207991`
- TD target mean/std: `-4.85825585 / 3.29945853`
- TD error std: `0.40676915`
- alpha: evaluation value `0.0014035053`
- evaluation mean log-std: `-2.99999666`
- NaN/Inf or Q explosion: none

Action semantics remained exact:

- translation projection: `0`
- rotation projection: `0`
- gripper clipping: `0`
- Replay-policy mismatch: `0`
- normal adapter delta mean: `6.8276e-17`
- `Q(policy_action) - Q(adapter(policy_action))` max absolute difference: `0`

The first deterministic 10k contained 15 task successes. The following 20k
stochastic behavior still contained no task success, which remains a limitation
for long-horizon policy improvement even though deterministic evaluation no
longer collapses.

## Verification

- SAC focused tests: `57 passed`
- Full pytest: `131 passed`
- Checkpoint restored at step 30k with 20k Critic updates, 10k Actor/alpha
  updates, and all 30k Replay transitions.

