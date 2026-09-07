import numpy as np
import matplotlib.pyplot as plt
import config


# Plot training reward curve with moving average
def plot_reward_curve(
    rewards_path=config.EPISODE_REWARDS_PATH,
    save_path=config.REWARD_CURVE_PATH,
    window=100
):
    # Load episode rewards
    rewards = np.load(rewards_path)

    # Calculate moving average
    moving_avg = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    # Create and configure reward plot
    plt.figure(figsize=(8, 5))
    plt.plot(moving_avg)
    plt.xlabel("Episode")
    plt.ylabel(f"Average reward (window={window})")
    plt.title("Q-Learning Training Progress on FrozenLake-v1")
    plt.grid(True)

    # Save reward curve
    plt.savefig(save_path)
    plt.close()

    print(f"Reward curve saved to {save_path}")


# Visualize learned Q-table as a heatmap
def plot_q_table_heatmap(
    q_table_path=config.Q_TABLE_PATH,
    save_path=config.Q_HEATMAP_PLOT_PATH
):
    # Load trained Q-table
    q_table = np.load(q_table_path)

    # Extract maximum Q-value for each state
    best_values = np.max(q_table, axis=1)

    # Convert state values into FrozenLake grid
    grid_size = int(np.sqrt(len(best_values)))
    grid = best_values.reshape(grid_size, grid_size)

    # Create heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="viridis")
    plt.colorbar(label="Max Q-value")
    plt.title("Learned State Values (max Q per state) - FrozenLake-v1")

    # Display Q-values inside grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(
                j,
                i,
                f"{grid[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8
            )

    # Save heatmap
    plt.savefig(save_path)
    plt.close()

    print(f"Q-table heatmap saved to {save_path}")


# Generate all training visualizations
if __name__ == "__main__":
    plot_reward_curve()
    plot_q_table_heatmap()