import numpy as np


# Initialize Q-learning agent
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))


# Select actions using epsilon-greedy policy
    def select_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))


# Update Q-table using Q-learning algorithm
    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


# Decay exploration rate
    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# Save and load Q-table
    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)