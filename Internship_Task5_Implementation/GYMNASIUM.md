# 🏋️ Part 5(B): OpenAI Gym / Gymnasium

<p align="center">
  <img src="https://img.shields.io/badge/Library-Gymnasium%201.3.0-green?logo=openaigym&logoColor=white" alt="Gymnasium 1.3.0"/>
  <img src="https://img.shields.io/badge/Environment-FrozenLake--v1-9cf" alt="FrozenLake-v1"/>
  <img src="https://img.shields.io/badge/State%20Space-Discrete(16)-informational" alt="Discrete(16)"/>
  <img src="https://img.shields.io/badge/Action%20Space-Discrete(4)-informational" alt="Discrete(4)"/>
</p>

> Documents the exploration of the **Gymnasium** library required for Task 5,
> Part 5(B) — what it is, how its interface works, which environments were
> considered, and why **FrozenLake-v1** was ultimately selected for the
> practical implementation in this repository.

---

## 📑 Table of Contents

1. [What is Gymnasium?](#-what-is-gymnasium)
2. [Core Building Blocks](#-core-building-blocks)
3. [The Training Loop](#-the-training-loop)
4. [Environments Considered](#-environments-considered)
5. [Environment Selection: Why FrozenLake-v1?](#-environment-selection-why-frozenlake-v1)
6. [FrozenLake-v1 Specification](#-frozenlake-v1-specification)
7. [Gymnasium API in Practice](#-gymnasium-api-in-practice)
8. [Where This Is Implemented](#️-where-this-is-implemented)
9. [Further Reading](#-further-reading)

---

## 📌 What is Gymnasium?

**Gymnasium** is the actively maintained, community-governed successor to
**OpenAI Gym** (maintained by the Farama Foundation). It provides a
**standard, unified interface** for reinforcement learning environments —
this single design decision is what makes it possible to plug the *same*
agent code into completely different problems, from balancing a pole to
navigating a grid to playing Atari, with almost no changes to the agent
itself.

Without a standard like Gymnasium, every environment would expose its own
bespoke API, and an RL agent written for one simulator would need to be
rewritten from scratch for another. Gymnasium solves this the same way a
USB standard solves device compatibility: one interface, many devices.

---

## 🧩 Core Building Blocks

Every Gymnasium environment is built from the same handful of pieces:

| Component | Description |
|---|---|
| **`observation_space`** | Shape/range of values the agent can observe as the "state" — e.g. a `Box` space of continuous values (like a robot's joint angles), or a `Discrete` space of a fixed number of integer states (like a grid tile index). |
| **`action_space`** | Set of actions the agent may take — `Discrete` (e.g. left/right/up/down) or continuous (`Box`, e.g. motor torque). |
| **`reset()`** | Starts a new episode, returns `(observation, info)`. Called once before training begins, and again after **every** episode ends. |
| **`step(action)`** | The core function. Takes an action, returns `(next_observation, reward, terminated, truncated, info)`. |
| **`terminated`** | `True` when the episode ends **naturally** — e.g. the goal is reached, or the agent falls into a hole. |
| **`truncated`** | `True` when the episode is cut off **externally** — e.g. a step-count time limit is hit, regardless of what the agent was doing. |
| **Episode termination** | An episode ends when *either* `terminated` or `truncated` is `True`. The environment must then be reset before any further `step()` calls. |

> 💡 **Why split `terminated` and `truncated`?** They mean different things
> for learning: a `terminated` failure (e.g. falling in a hole) should
> usually be treated as "no future reward" when updating Q-values, while a
> `truncated` timeout doesn't necessarily mean the agent's strategy was
> bad — it just ran out of time. Older Gym versions merged both into a
> single `done` flag, which lost this distinction.

---

## 🔁 The Training Loop

Training an agent in *any* Gymnasium environment reduces to the same loop:

```text
┌─────────────────────────────────────────────────────────┐
│  1. reset()             → get the initial observation    │
│  2. select_action()     → choose an action                │
│  3. step(action)        → get next_obs, reward, done flags│
│  4. store + update       → record reward, update policy    │
│  5. terminated/truncated?                                  │
│         │yes → back to 1 (new episode)                     │
│         │no  → back to 2 (same episode continues)          │
└─────────────────────────────────────────────────────────┘
```

This exact loop is implemented in [`rl/train.py`](rl/train.py) (with
learning/exploration active) and [`rl/evaluate.py`](rl/evaluate.py) (run
greedily, with exploration switched off, to test the final policy).

---

## 🌍 Environments Considered

Gymnasium ships with a wide range of built-in environments, grouped by
family:

| Family | Examples | State/Action space |
|---|---|---|
| **Toy Text** | `FrozenLake-v1`, `Taxi-v3`, `CliffWalking-v0` | Discrete, small — fully enumerable |
| **Classic Control** | `CartPole-v1`, `MountainCar-v0`, `Pendulum-v1` | Continuous or mixed |
| **Box2D** | `LunarLander-v3`, `BipedalWalker-v3` | Continuous, physics-based |
| **Atari** | `Breakout`, `Pong`, etc. | Raw pixels — very large state space |
| **MuJoCo** | `Hopper-v4`, `Ant-v4` | Continuous, high-dimensional robotics |

Two candidates were shortlisted for this task: **`FrozenLake-v1`** and
**`CartPole-v1`** — both explicitly recommended for a first practical
implementation because of their small, easy-to-reason-about state spaces.

---

## 🎯 Environment Selection: Why FrozenLake-v1?

**`FrozenLake-v1` (4×4 map)** was selected over `CartPole-v1` because:

- ✅ **Small, fully enumerable state space** — `Discrete(16)` observations
  and `Discrete(4)` actions mean the *entire* Q-table (16 × 4 = 64 values)
  can be printed, inspected, and reasoned about directly — see the Q-table
  heatmap in the full report.
- ✅ **Full transparency, no black box** — every part of the training
  process can be understood and explained line-by-line, in line with the
  task requirement to not simply run AI-generated code without
  understanding it. `CartPole-v1`'s continuous observation space would
  require discretisation or function approximation before tabular
  Q-Learning could even be applied.
- ✅ **Genuinely stochastic MDP** — with `is_slippery=True`, actions
  succeed as intended only part of the time (the rest of the probability
  mass slips sideways), so the environment still exercises real RL
  concepts — exploration vs exploitation, delayed/sparse reward,
  convergence under uncertainty — without the added complexity of a
  continuous state space.
- ✅ **Matches the algorithm** — Q-Learning (Table 1 in the full report)
  is described as *"guaranteed to converge in tabular settings"* — that
  guarantee only applies to environments like FrozenLake, not to
  continuous ones.

---

## 📋 FrozenLake-v1 Specification

| Property | Value |
|---|---|
| **Observation space** | `Discrete(16)` — one integer per grid tile (0–15) |
| **Action space** | `Discrete(4)` — `0`=Left, `1`=Down, `2`=Right, `3`=Up |
| **Reward** | `+1` on reaching the goal tile, `0` otherwise |
| **`terminated`** | `True` on reaching the goal (G) or falling into a hole (H) |
| **`truncated`** | `True` at the 100-step limit |
| **Map** | `4x4`, `is_slippery=True` (stochastic transitions) |

**The map layout** (`S`=start, `F`=frozen/safe, `H`=hole, `G`=goal):

S F F F
F H F H
F F F H
H F F G


---

## 💻 Gymnasium API in Practice

How the concepts above map onto actual code used in this repository:

```python
import gymnasium as gym

# Create the environment (Part 5B: environment construction)
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)

print(env.observation_space)   # Discrete(16)
print(env.action_space)        # Discrete(4)

# Start an episode
observation, info = env.reset(seed=42)

# Take one step
action = env.action_space.sample()   # random action, for illustration
next_observation, reward, terminated, truncated, info = env.step(action)

if terminated or truncated:
    observation, info = env.reset()   # must reset before stepping again
```

This is exactly what [`rl/environment.py`](rl/environment.py) wraps, and
what [`rl/train.py`](rl/train.py) / [`rl/evaluate.py`](rl/evaluate.py)
loop over thousands of times during training and evaluation.

---

## 🗂️ Where This Is Implemented

| File | Role |
|---|---|
| [`rl/environment.py`](rl/environment.py) | Thin wrapper around `gym.make()`; seeds `reset()` and the action space for reproducibility |
| [`rl/train.py`](rl/train.py) | The training loop described above, with epsilon-greedy exploration |
| [`rl/evaluate.py`](rl/evaluate.py) | Same loop, run **greedily** (no exploration) to test the trained policy |
| [`config.py`](config.py) | `ENV_NAME`, `MAP_NAME`, `IS_SLIPPERY`, `MAX_STEPS_PER_EPISODE` — exact environment configuration used |

---

## 📚 Further Reading

- [Gymnasium official documentation](https://gymnasium.farama.org/)
- [FrozenLake-v1 environment page](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- Full theory on RL algorithms, MDPs, and Game Theory: [`report/Task5_Complete_Report.pdf`](report/Task5_Complete_Report.pdf)

---

<p align="center">
  <sub>Part of Task 5 — Reinforcement Learning, Gymnasium &amp; Game Theory · Muzammil Ahmed</sub>
</p>