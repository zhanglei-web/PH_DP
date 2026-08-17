# SAC v2 Mean-Policy Improvement Protocol

## Motivation

The previous medium-horizon release coupled two incompatible controls: a hard
initial-policy trust region kept `log_std` near -3 while a time-driven entropy
schedule demanded substantially higher entropy. Automatic temperature tuning
therefore increased alpha, enlarging soft Bellman targets without allowing the
policy distribution to satisfy the target.

This revision decouples deterministic policy improvement from entropy release.

## Starting point

- Resume the validated 30,000-step checkpoint from
  `sac_v2_final_safe_trust_20260813T100000Z`.
- Preserve Actor, critics, target critics, replay, optimizers, RNG and counters.
- Native constrained action semantics remain unchanged.

## Mean-only phase

- `target_entropy = -16.7`
- alpha is frozen at the resumed value (`0.0014035053` in the audited run)
- `log_std_head` is frozen
- Actor trunk and mean head remain trainable
- Critic and target-Critic updates are unchanged
- gamma, tau, learning rates, batch size and UTD are unchanged

## Online data mixture

After 30k, behavior alternates at whole-episode boundaries:

- 50% deterministic current mean-policy episodes
- 50% stochastic current-policy episodes

Both are online data from the current Actor. No Expert/HDF5 replay is preloaded.
Replay continues to store the agent-selected constrained `policy_action`.

## Stable-policy trust region

- Initial stable anchor: validated 30k Actor
- Empirical KL is evaluated separately on formal-train and online Replay states
- Per-dimension KL limit: `1e-4` initially
- Module-relative parameter limit: `1e-2`, anomaly guard only
- Evaluation promotion margin: +2 percentage points on 100 fixed validation episodes
- Performance rollback threshold: -15 percentage points
- After rollback, KL is multiplied by 0.3 with floor `1e-5`
- A promoted 100-episode checkpoint becomes the next stable anchor
- Rollback restores only Actor and resets its Adam moments; Critic, Replay and
  corrected action attribution are retained

## Entropy gate

This implementation does not release entropy during the mean-only validation.
An entropy change is permitted only in a later explicitly authorized stage after:

1. 100-seed deterministic performance improves over the stable anchor;
2. stochastic online interaction produces full-success trajectories;
3. Critic/target statistics remain finite and stationary;
4. Actor movement is nonzero but passes the stable-policy guard.

## Validation findings

The first bounded proposal used KL `1e-3`. At 40k it remained numerically stable
but achieved only 5/100 successes versus the 30k baseline of 41/100; it was
rejected as behaviorally unsafe.

The guarded `1e-4` run achieved 21/100 at 40k and triggered the configured
20-point rollback. A subsequent `3e-5` window reached 26/100 at 50k. Numerical
health remained stable, and deterministic online interaction added 15 success
episodes, but stochastic interaction still added zero successes. The policy was
therefore not promoted and entropy was not released.

This is a conservative protocol implementation and bounded validation, not a
claim that online SAC improvement has been demonstrated.
