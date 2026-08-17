# SAC Core v1 Specification Freeze

Status: `SAC_CORE_V1_READY`

## 1. Scope

Algorithm: BC-initialized online off-policy Soft Actor-Critic. This specification
freezes only the optimization core and its in-memory replay schema. The formal
online training and evaluation protocol is intentionally out of scope.

SAC v1 starts with:

- Actor: frozen SAC Actor v1 initialization artifact;
- Q1/Q2: independent random initialization;
- target Q1/Q2: exact copies of Q1/Q2 with gradients disabled;
- replay: empty;
- data source: future online MuJoCo interaction only.

It does not use Critic pretraining, expert/formal HDF5 replay, offline RL,
prioritized replay, N-step returns, random-action warmup, or external action noise.

## 2. Frozen Actor and preprocessing

Artifact:

```text
outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/actor_initialized.pt
```

The loader requires artifact format
`sac_actor_v1_full_mean_path_distilled`, checks its SHA-256, requires the frozen
42-dimensional observation mean/std, validates the frozen action specification,
and verifies initial `log_std=-3` with zero log-std weights.

Observation is raw `policy_state_42`, normalized at model input with the Actor
artifact's train-only mean/std. No new SAC normalization is fitted.

The Actor samples with `rsample`, applies tanh, and computes the summed 7-D
tanh-corrected log probability with shape `[B,1]`. Therefore the entropy convention
is compatible with `target_entropy=-action_dim=-7.0`.

## 3. Action and Replay semantics

The action is continuous normalized action 7:

```text
[dx, dy, dz, drx, dry, drz, gripper], each in [-1,1]
```

During rollout, the stochastic Actor output is denormalized with the frozen
`ExpertActionSpec`, passed through the existing Cartesian/IK adapter, and executed.
Replay stores the adapter's post-limit normalized 7-D policy command
(`AdaptedCommand.normalized`) that corresponds to the deployed Cartesian command.
It never stores Gaussian pre-squash `u`, mean action, physical command, or joint/IK
target. An IK safety failure is a true terminal under `sac_reward_v1`.

Each transition contains:

| Field | Shape | dtype | Meaning |
|---|---:|---|---|
| observation | `[42]` | float32 | raw policy state before action |
| action | `[7]` | float32 | deployed normalized policy action |
| reward | scalar | float32 | `sac_reward_v1` |
| next_observation | `[42]` | float32 | raw next policy state |
| terminated | scalar | bool | true MDP/task/safety terminal |
| truncated | scalar | bool | external horizon only |

Replay is an in-memory circular array with uniform sampling and no disk backing.
At capacity 1,000,000 it allocates 370,000,000 bytes (370 MB decimal, about
352.86 MiB): 84 float32 observation values, 7 float32 actions, one float32 reward,
and two boolean flags per transition.

## 4. Twin Critic

Q1 and Q2 are independent networks with independent random initialization:

```text
concat(normalized policy_state_42, normalized action_7)
49 -> 256 -> SiLU -> 256 -> SiLU -> 256 -> SiLU -> 1
```

There is no normalization layer, dropout, phase/history input, or shared Critic
trunk. Target Q1/Q2 begin as exact online copies, have `requires_grad=False`, and
are excluded from optimizers.

## 5. Terminal mask and TD target

Only `terminated` suppresses bootstrapping:

```text
m = 1 - terminated
```

Consequently a pure time-limit transition (`terminated=False`, `truncated=True`)
continues to bootstrap. `done = terminated OR truncated` must never be used as the
bootstrap mask.

With a stochastic next action and its corrected log probability:

```text
soft_value = min(target_Q1(next_obs,next_action),
                 target_Q2(next_obs,next_action)) - alpha * next_log_prob
y = reward + 0.995 * (1 - terminated) * soft_value
```

The full target calculation runs under `torch.no_grad()`.

Critic loss:

```text
L_Q = MSE(Q1(obs,action), y) + MSE(Q2(obs,action), y)
```

## 6. Actor and entropy losses

Actor loss:

```text
policy_action, log_prob = actor.rsample_and_tanh(obs)
Q_min = min(Q1(obs,policy_action), Q2(obs,policy_action))
L_actor = mean(alpha.detach() * log_prob - Q_min)
```

Critic parameters are temporarily gradient-disabled during this backward pass;
the differentiable action-to-Q path remains intact, so gradients reach the Actor.

Automatic entropy tuning uses a scalar `log_alpha`:

```text
alpha = exp(log_alpha)
log_alpha_init = log(0.1)
target_entropy = -7.0
L_alpha = -mean(log_alpha * detach(log_prob + target_entropy))
```

The initial alpha `0.1` is a conservative standard SAC value: it enables automatic
tuning without immediately making entropy dominate the BC-initialized policy.

## 7. Target update

Every SAC update performs:

```text
target = (1 - 0.005) * target + 0.005 * online
```

The direction is explicitly tested with online parameters equal to 1 and target
parameters equal to 0; one update produces 0.005.

## 8. Frozen hyperparameters

| Item | Value |
|---|---:|
| gamma | 0.995 |
| tau | 0.005 |
| Actor optimizer | Adam |
| Critic optimizer | Adam |
| Alpha optimizer | Adam |
| actor learning rate | 3e-4 |
| critic learning rate | 3e-4 |
| alpha learning rate | 3e-4 |
| batch size | 256 |
| replay capacity | 1,000,000 |
| learning starts | 10,000 transitions |
| target entropy | -7.0 |
| alpha initial value | 0.1 |
| update-to-data ratio | 1 |
| Actor update frequency | 1 |
| Target update frequency | 1 |

Before `learning_starts`, transitions are collected with the stochastic
BC-initialized SAC policy. Uniform random actions, epsilon-greedy, OU noise, and
external Gaussian noise are forbidden. Afterward, every environment transition
causes one update in this order:

1. uniformly sample one replay batch;
2. update Q1/Q2;
3. update Actor;
4. update alpha;
5. Polyak-update target Q1/Q2.

There is no TD3-style delayed Actor update.

## 9. Verification

The dedicated tests cover:

- Critic `[B,42]+[B,7] -> [B,1]` shapes and independent parameters;
- target exact-copy initialization and gradient exclusion;
- replay push/sample, terminal separation, deployed action identity, and wraparound;
- true-terminal target `y=r` and time-limit bootstrap;
- positive finite alpha and alpha gradient direction;
- Polyak direction and exact coefficient;
- finite Critic, Actor, and alpha losses with working backward passes;
- exact frozen Actor artifact and normalization loading;
- a two-transition MuJoCo `sac_reward_v1` smoke plus one synthetic SAC update.

The smoke test is an interface check only and is not an online training result.

The current PyTorch 2.13 environment lazily imports its Triton/Dynamo optimizer
backend. The full pytest process preloads this backend before MuJoCo/Pinocchio test
collection to avoid a native dynamic-loader crash caused by the reverse import
order. This changes neither SAC computations nor production parameters; the SAC
pipeline must likewise construct/load the Core before constructing MuJoCo workers.

```text
SAC_CORE_V1_READY
```
