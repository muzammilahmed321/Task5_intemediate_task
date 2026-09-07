# Commit: Initial FrozenLake Q-Learning configuration
# Version: 1.0
# Seed: 42

import os

SEED = 42

ENV_NAME = "FrozenLake-v1"
MAP_NAME = "4x4"
IS_SLIPPERY = True
MAX_STEPS_PER_EPISODE = 100

ENV_KWARGS = {
    "map_name": MAP_NAME,
    "is_slippery": IS_SLIPPERY
}

ALPHA = 0.1
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

N_EPISODES = 15_000
N_EVAL_EPISODES = 1_000

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

Q_TABLE_PATH = os.path.join(RESULTS_DIR, "q_table.npy")
EPISODE_REWARDS_PATH = os.path.join(RESULTS_DIR, "episode_rewards.npy")
EVAL_LOG_PATH = os.path.join(RESULTS_DIR, "evaluation_log.txt")

REWARD_CURVE_PATH = os.path.join(PLOTS_DIR, "reward_curve.png")
Q_HEATMAP_PLOT_PATH = os.path.join(PLOTS_DIR, "q_table_heatmap.png")