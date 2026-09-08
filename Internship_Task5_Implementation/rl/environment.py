"""
Gymnasium environment wrapper.

Gymnasium (the maintained successor to OpenAI Gym) provides a standard
interface for RL environments built around a few core pieces:

- observation_space: the shape/range of values the agent can observe as
  the "state" (Discrete(16) for FrozenLake-v1 — one integer per grid tile).
- action_space: the set of actions the agent may take (Discrete(4) for
  FrozenLake-v1 — Left, Down, Right, Up).
- reset(): starts a new episode and returns the initial observation.
  Called once at the start of training and again after every episode ends.
- step(action): the core function. Takes an action and returns
  (next_observation, reward, terminated, truncated, info).
    - terminated=True means the episode ended naturally (goal reached or
      agent fell into a hole).
    - truncated=True means the episode was cut off by a step limit.
- Episode termination: an episode ends when either terminated or
  truncated is True; the environment must then be reset before further
  actions can be taken.

See GYMNASIUM.md for the full write-up of these concepts.
"""

import gymnasium as gym


def make_env(env_name, seed=None, **env_kwargs):
    env = gym.make(env_name, **env_kwargs)

    if seed is not None:
        # Gymnasium seeds are passed into reset(), not the constructor,
        # so the very first reset() call is what makes episodes reproducible.
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env