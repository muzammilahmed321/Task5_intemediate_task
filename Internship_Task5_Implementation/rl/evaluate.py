import numpy as np
from rl.environment import make_env
from rl.agent import QLearningAgent
import config


# Load trained Q-table and evaluate the agent
def evaluate(
    q_table_path=config.Q_TABLE_PATH,
    n_episodes=config.N_EVAL_EPISODES,
    render=False
):
    # Create evaluation environment
    env = make_env(
        config.ENV_NAME,
        seed=config.SEED + 1,
        **config.ENV_KWARGS
    )

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize agent and load trained Q-table
    agent = QLearningAgent(n_states, n_actions)
    agent.load(q_table_path)

    # Track evaluation performance
    successes = 0
    episode_lengths = []
    logs = []

    # Run evaluation episodes
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        steps = 0

        # Run the current episode
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Select the best action without exploration
            action = agent.select_action(state, greedy=True)

            # Execute action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state
            steps += 1

            # Check whether the episode has ended
            if terminated or truncated:
                done = True

                # Count successful episodes
                if reward > 0:
                    successes += 1

                break

        # Store episode statistics
        episode_lengths.append(steps)

        logs.append(
            f"Episode {episode + 1}: "
            f"steps={steps}, "
            f"success={reward > 0 if done else False}"
        )

    # Close the environment
    env.close()

    # Calculate evaluation metrics
    success_rate = successes / n_episodes
    avg_length = np.mean(episode_lengths)

    # Generate evaluation summary
    summary = (
        f"Evaluation over {n_episodes} episodes\n"
        f"Success rate: {success_rate:.2%}\n"
        f"Average episode length: {avg_length:.2f} steps\n"
    )

    # Save evaluation results to log file
    with open(config.EVAL_LOG_PATH, "w") as f:
        f.write(summary + "\n")
        f.write("\n".join(logs))

    # Display evaluation results
    print(summary)

    return success_rate, avg_length


# Run evaluation when this file is executed directly
if __name__ == "__main__":
    evaluate()