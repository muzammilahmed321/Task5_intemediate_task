import os
import numpy as np

from utils import set_seed
import config
from rl.train import train
from rl.evaluate import evaluate
from visualize import plot_reward_curve, plot_q_table_heatmap


# Run the complete Q-learning pipeline
def main():

    # Create directories for generated plots
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # Set random seeds for reproducibility
    set_seed(config.SEED)

    # Train the Q-learning agent
    print("Starting training...")
    agent, rewards = train()

    # Save the trained Q-table
    agent.save(config.Q_TABLE_PATH)

    # Save episode rewards
    np.save(config.EPISODE_REWARDS_PATH, np.array(rewards))

    print("Training finished.\n")

    # Evaluate the trained agent
    print("Running evaluation...")
    evaluate()

    print("Evaluation finished.\n")

    # Generate training visualizations
    print("Generating visualizations...")
    plot_reward_curve()
    plot_q_table_heatmap()

    print("All done. Check the 'results' folder.")


# Run the complete pipeline when executed directly
if __name__ == "__main__":
    main()