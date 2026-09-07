# Add FrozenLake environment factory
# Version: 1.0

import gymnasium as gym


def make_env(env_name, seed=None, **env_kwargs):
    env = gym.make(env_name, **env_kwargs)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env