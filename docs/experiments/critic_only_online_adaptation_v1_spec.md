# Critic-Only Online Adaptation v1

## Purpose

This is not online SAC. “Online” only means that a frozen deterministic Native
Constrained Actor is rolled out in the real MuJoCo task to create a fixed
current-policy corpus. The entire corpus is collected before any optimizer is
created. The Actor, its log standard deviation, reward, observation, and action
semantics remain frozen.

## Dataset

- Actor: Step-0 Actor from aligned Actor-Critic v2.
- Reward: `sac_reward_v2_candidate`.
- New seeds: `800000-800999`, disjoint from historical training/development
  pools and sealed final seeds `500000-500099`.
- Episode split: first 800 train, next 100 validation, final 100 test.
- Action: deterministic constrained mean action; no Gaussian, random, or
  epsilon noise.
- Stored transitions include state42, attempted policy action7, Reward v2,
  next state, separate terminated/truncated, phase, outcome, episode, and step.

The fixed corpus contains 1,000 episodes and 175,118 transitions. Outcome
counts are 476 success, 154 illegal drop, 37 timeout, and 333 other failures
(318 IK limits and 15 unstable releases). No adapter projection occurred.

## Critic adaptation

Q1/Q2 start at Mixed Critic v2. Target critics are exact copies. Every update
uses exactly 128 transitions from the frozen Rule Expert v2 train split and 128
from the frozen Actor train split.

For both sources the target is deterministic fixed-policy evaluation:

```text
a_next = pi0(s_next)
y = r + 0.995 * (1 - terminated) * min(target_Q1, target_Q2)(s_next,a_next)
```

Truncation bootstraps. There is no entropy/log-probability/alpha term. Q1/Q2
use summed MSE and Adam at `3e-4`; targets use Polyak `tau=0.005`. Checkpoints
are fixed at 0, 5k, 10k, and 20k updates.

## Selection and result

Selection is lexicographic rather than a tuned composite: maximize online
validation Spearman among checkpoints whose offline validation Spearman is no
more than 0.05 below step 0. No nonzero checkpoint passed the retention gate.

At step 10k, online Actor test Spearman improved from 0.463 to 0.682 and MAE
from 7.800 to 5.825. However, offline Expert test Spearman collapsed from
0.579 to 0.051 and MAE increased from 2.073 to 3.877. Fixed-policy full-return
ranking improved only from -0.071 to 0.184 overall, while P3 remained negative
(-0.332) and P4 reversed from +0.577 to -0.363. Illegal-drop predecessor MAE
improved in scale, but ranking remained negative/unreliable.

The experiment is therefore a regression: fixed-policy TD on a nominal 50/50
transition mixture adapts to current-policy returns but catastrophically erases
the original Expert value ordering and does not make P3/P4 action replacement
reliable. The selected checkpoint remains step 0. No Actor release is allowed.
