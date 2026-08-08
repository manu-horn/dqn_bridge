import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self._buffer = None
        self._pos = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done):
        if self._buffer is None:
            self._buffer = {
                "state": np.zeros((self.capacity, *state.shape), dtype=np.float32),
                "action": np.zeros(self.capacity, dtype=np.int64),
                "reward": np.zeros(self.capacity, dtype=np.float32),
                "next_state": np.zeros((self.capacity, *next_state.shape), dtype=np.float32),
                "done": np.zeros(self.capacity, dtype=np.float32),
            }

        idx = self._pos % self.capacity
        self._buffer["state"][idx] = state
        self._buffer["action"][idx] = action
        self._buffer["reward"][idx] = reward
        self._buffer["next_state"][idx] = next_state
        self._buffer["done"][idx] = float(done)
        self._pos += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self._size, batch_size, replace=False)
        return (
            torch.FloatTensor(self._buffer["state"][idx]),
            torch.LongTensor(self._buffer["action"][idx]).unsqueeze(1),
            torch.FloatTensor(self._buffer["reward"][idx]).unsqueeze(1),
            torch.FloatTensor(self._buffer["next_state"][idx]),
            torch.FloatTensor(self._buffer["done"][idx]).unsqueeze(1),
        )

    def __len__(self):
        return self._size


class DoubleDQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        min_replay_size=1000,
        target_update_freq=500,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def _get_epsilon(self):
        frac = self.steps / self.epsilon_decay
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(0.0, 1.0 - frac)

    def select_action(self, state):
        if np.random.random() < self._get_epsilon():
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state).unsqueeze(0))
            return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()


def moving_average(data, window):
    return np.convolve(data, np.ones(window), "valid") / window


def plot_training_curve(rewards, window=100, save_path="training_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")
    if len(rewards) >= window:
        ma = moving_average(rewards, window)
        plt.plot(range(window - 1, len(rewards)), ma, label=f"Rolling Avg ({window} eps)")
    plt.axhline(y=500, color="g", linestyle="--", label="Target Score (500)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Double DQN Training on CartPole-v1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim)

    episode_rewards = []

    for episode in range(500):
        state, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            agent.update()

        episode_rewards.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {episode_reward:5.1f} | "
                f"Avg(50): {avg:5.1f} | "
                f"Eps: {agent._get_epsilon():.3f}"
            )

    env.close()
    plot_training_curve(episode_rewards)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "double_dqn_rewards.json")
    with open(log_path, "w") as f:
        json.dump({"algorithm": "Double DQN", "rewards": episode_rewards}, f)
    print(f"Rewards saved to {log_path}")

    best = max(episode_rewards)
    print(f"Training complete. Best episode reward: {best}")


if __name__ == "__main__":
    train()