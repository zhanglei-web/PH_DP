# Online SAC Training Pipeline v1 Protocol

Status: `SAC_TRAINING_PIPELINE_V1_READY`

This protocol executes the frozen `SAC Core v1`; it does not change reward,
Actor/Critic architecture, loss definitions, or hyperparameters.

## Initialization

- Actor artifact: `outputs/sac_actor/sac_actor_v1_full_distill_20260812T180000Z/actor_initialized.pt`
- Q1/Q2: independent random initialization.
- Target Q1/Q2: exact online copies, gradients disabled.
- Alpha: automatic, initialized to `0.1`.
- Replay: empty; no expert, HDF5, BC, recovery, or offline preload.
- Construction order: load SAC Core/optimizers before constructing MuJoCo workers.

The entrypoint asserts `PickPlaceEnv(reward_version="sac_reward_v1")`, obtains raw
`policy_state_42`, and uses the Actor artifact's observation mean/std. Training
actions are stochastic squashed Gaussian samples. There is no random-action warmup
or external exploration noise.

## Frozen optimization schedule

```text
gamma                   = 0.995
tau                     = 0.005
actor_lr                = 3e-4
critic_lr               = 3e-4
alpha_lr                = 3e-4
batch_size              = 256
replay_capacity         = 1_000_000
learning_starts         = 10_000
target_entropy          = -7.0
alpha_init              = 0.1
UTD                     = 1
actor_update_frequency  = 1
target_update_frequency = 1
```

`global_env_steps` counts real MuJoCo transitions. The first 10,000 transitions
perform no optimization. Transition 10,001 triggers the first update; therefore a
20,000-step run performs exactly 10,000 updates. `gradient_updates` and
`episode_count` are separate counters.

Each post-warmup transition is pushed, then one replay batch drives Q1/Q2, Actor,
alpha, and target-Polyak updates in that order.

At every true terminal or time-limit truncation the training environment resets.
This resets `sac_reward_v1` phase/runtime/event state, but not replay or networks.

## Seed separation

Repository audit found `400000` already used by a Core smoke test, so the proposed
400000 range was not frozen.

```text
Initialization reference: 300000-300099 (reference only)
Validation pool:          410000-410099
Final test pool:          500000-500099
Training stream:          700000, 700001, ... per reset
```

Online evaluations use the first 20 validation seeds (`410000-410019`). A future
important/final checkpoint may use all 100 validation seeds. Final test seeds must
not be used for model selection.

## Evaluation

Every 10,000 environment steps, an independent environment evaluates 20 episodes
with `deterministic_action=tanh(mean)`. Evaluation performs no replay writes and no
Actor, Critic, target, or alpha updates and does not affect the training environment.

Metrics include success and five milestones, termination reasons, undiscounted and
discounted returns, episode length, mean deterministic-policy log probability,
log-std min/mean/max, alpha, replay size, and the latest training-window Actor/Q/loss
metrics.

Best-checkpoint ordering is lexicographic:

1. higher deterministic success rate;
2. lower illegal-drop rate;
3. higher mean evaluation return.

Training return never selects the best checkpoint.

## Logging and checkpoints

- aggregate training log: every 1,000 environment steps;
- deterministic evaluation: every 10,000 steps;
- periodic checkpoint: every 50,000 steps;
- latest: at every evaluation and clean run completion;
- best: whenever the deterministic selection metric improves.

Artifacts:

```text
config.json
training_metrics.jsonl
episode_metrics.csv
evaluation_metrics.jsonl
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/step_NNNNNNNNN.pt
sanity_report.md
```

Checkpoint contents include Actor, online and target Critics, all optimizers,
log-alpha, counters, frozen config/normalization, Actor artifact checksum, Python /
NumPy / PyTorch RNG states, and initialized replay arrays plus replay RNG state.
Replay is compactly serialized only through its current size, not its empty capacity.

MuJoCo simulator state and an in-flight partial episode are not serialized. Resume
therefore starts a new training seed/reset boundary while preserving every completed
transition, replay RNG, optimizer, network, and counter. It is a true optimization
resume, but not bit-exact continuation of a partially executed physical episode.

## Safety monitoring

Every update requires finite losses/Q/alpha. The run aborts on NaN/Inf, absolute Q
mean above `1e6`, or alpha outside `(1e-8,1e3)`. Logs expose reward component sums,
reward mean, Critic/Actor/alpha losses, alpha, Q1/Q2/target means, and log-std range.

The short integration smoke is not a formal training result. Formal long training
requires a separate user instruction after review of the sanity report.

```text
SAC_TRAINING_PIPELINE_V1_READY
```
