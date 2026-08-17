# Aligned Actor-Critic v1 Spec

## Scope and status

This revision packages the frozen native-constrained BC Actor and the mixed-data
MC-pretrained Critic into one **offline initialization**. It performs no Actor,
Critic, alpha, target-network, replay, or online-environment training update.

The package is semantically aligned, but the policy-improvement gate is blocked:
the held-out Expert-vs-Actor counterfactual audit is unreliable in P4. The artifact
is therefore an auditable initialization candidate, not authorization to run AWAC
or online SAC.

## Sources

- Actor: `outputs/sac_actor/sac_constrained_actor_v2_20260812T165925Z/actor_initialized.pt`
  - SHA-256: `688fdc1c275160e5a0c2956f7275d77488d00923bd308b08030860b7461520cd`
  - learned from nominal-success Rule Expert demonstrations only (1000 episodes;
    frozen 900/100 Actor split).
  - deterministic seeds 300000--300099: 35/100 success; grasp 86%, lift 68%,
    transport 68%, release 64%, retreat 35%.
- Critic: `outputs/sac_critic/sac_critic_pretrain_v1_20260813T210000Z/critic_pretrained_best.pt`
  - two independent `49 -> 256 -> 256 -> 256 -> 1` SiLU Q networks.
  - trained on all 1300 Rule Expert trajectories: 1000 nominal success plus 300
    perturbed/recovered/failure episodes.
  - objective: twin MC regression to finite recorded returns under gamma 0.995.
- Dataset: `manifests/rule_expert_v1_formal.json`.
- Reward: frozen `sac_reward_v1` and `SACPickPlaceProtocol`.

## Shared semantics

Both networks consume the same float32 `policy_state_42` ordering and exactly the
same frozen mean/std. Actor output and Critic action input are the same native
constrained normalized action:

```text
||a_xyz||_2 <= 1
||a_rot||_2 <= 1
a_gripper in [-1, 1]
```

The Critic data uses attempted policy commands, including IK-fallback rows; it
does not relabel failures with the deployed safe fallback. Formal data has zero
adapter-projected rows.

## Reward and value target

Every formal episode is reinterpreted with the production frozen reward implementation:

```text
G_t = sum_{k=0}^{T-t-1} 0.995^k r_{t+k}
```

The reconstruction covers P1 progress, P2 stable-grasp event, P3 progress, P4
place and retreat progress, legal place event, success terminal, illegal drop,
and true failure. Production regression gives reward max absolute difference
`1.31145e-15` and zero terminal, phase, and illegal-drop mismatches.

The aligned held-out test reproduction is:

```text
transitions  = 15,505
MAE          = 2.122162
RMSE         = 4.035522
Pearson      = 0.664198
Spearman     = 0.601387
```

Target Q1/Q2 are exact, frozen copies of the pretrained online Q1/Q2.

## Expert-vs-Actor value audit

The audit uses 100 exact reconstructed held-out nominal states (seeds
100900--100999), 25 per phase. At each state, the only branch difference is the
first action: recorded Rule Expert attempted action versus deterministic
constrained Actor action. Both branches then use the frozen Rule Expert feedback
controller. MuJoCo state reconstruction and recorded expert-action reproduction
both have max absolute error 0.

At H=20:

| Phase | Spearman(Delta Q, Delta G) | Sign agreement | Correct when Expert truly better |
|---|---:|---:|---:|
| Overall | 0.538 | 85% | 91.2% (31/34) |
| P1 | 0.818 | 100% | 100% (6/6) |
| P2 | 0.901 | 100% | 100% (17/17) |
| P3 | 0.312 | 84% | 66.7% (4/6) |
| P4 | -0.312 | 56% | 80% (4/5) |

The overall result is meaningful and much stronger than the old arbitrary-local
action audit, but it is not uniformly reliable. P4 ordering is anticorrelated and
near chance by sign. Since place/release/retreat is a safety- and success-critical
boundary, overall averaging cannot waive this failure.

## Artifact

`outputs/sac_aligned/aligned_actor_critic_v1_20260813T231500Z/aligned_actor_critic_v1.pt`

It contains Actor, Q1/Q2, exact-copy Target Q1/Q2, observation normalization,
observation/action specs, source hashes, manifest content hash, reward version,
and gamma. It deliberately contains no Replay and no optimizer/online resume state.

## Gate conclusion

The Actor-Critic artifact is mechanically and semantically aligned, and the
behavior-manifold return fit is reproduced. It is **not yet qualified for
conservative Actor improvement**, because Expert-vs-Actor value ordering fails
the phase-specific P4 reliability requirement. No Actor improvement is started.

`ALIGNED_CRITIC_NOT_READY_FOR_POLICY_IMPROVEMENT`
