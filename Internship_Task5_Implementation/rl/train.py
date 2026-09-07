import numpy as np
from rl.environment import make_env
from rl.agent import QLearningAgent
import config


# Create environment and initialize Q-learning agent
def train():
    env = make_env(
        config.ENV_NAME,
        seed=config.SEED,
        **config.ENV_KWARGS
    )

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(
        n_states,
        n_actions,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
    )

    # Store rewards for each episode
    episode_rewards = []

    # Run training episodes
    for episode in range(config.N_EPISODES):
        state, _ = env.reset()
        total_reward = 0.0

        # Run steps within the current episode
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Select action using epsilon-greedy policy
            action = agent.select_action(state)

            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q-table
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        # Reduce exploration after each episode
        agent.decay_epsilon()

        # Record episode reward
        episode_rewards.append(total_reward)

    # Close environment and return training results
    env.close()
    return agent, episode_rewards


# Run training when this file is executed directly
if __name__ == "__main__":
    agent, rewards = train()

    # Save trained Q-table
    agent.save(config.Q_TABLE_PATH)

    # Calculate average reward from the last 100 episodes
    window = 100
    avg_last = np.mean(rewards[-window:])

    print(f"Training complete over {config.N_EPISODES} episodes.")
    print(f"Average reward over last {window} episodes: {avg_last:.3f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table saved to {config.Q_TABLE_PATH}")

    # Save episode rewards for later analysis and plotting
    np.save(config.EPISODE_REWARDS_PATH, np.array(rewards))