import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from utils import set_seed


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return td.Categorical(logits=self.net(x))


class ReinforceAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.episode_returns = []

    def select_action(self, state):
        dist = self.policy(torch.FloatTensor(state).unsqueeze(0))
        action = dist.sample()
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def _compute_returns(self):
        returns = np.zeros(len(self.rewards), dtype=np.float32)
        g = 0.0
        for t in range(len(self.rewards) - 1, -1, -1):
            g = self.rewards[t] + self.gamma * g
            returns[t] = g
        return returns

    def update(self):
        returns = self._compute_returns()

        if len(self.episode_returns) > 1:
            mean = np.mean(self.episode_returns)
            std = np.std(self.episode_returns)
        else:
            mean = returns.mean()
            std = returns.std()
        standardized = (returns - mean) / (std + 1e-8)
        self.episode_returns.append(returns[0])

        log_probs = torch.stack(self.log_probs)
        standardized_t = torch.FloatTensor(standardized)
        loss = -(log_probs * standardized_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._wipe_memory()

        return loss.item()

    def _wipe_memory(self):
        del self.states, self.actions, self.rewards, self.log_probs
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []


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
    plt.title("REINFORCE Training on CartPole-v1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def train(seed=42, num_episodes=1000):
    env = gym.make("CartPole-v1")
    set_seed(seed, env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ReinforceAgent(state_dim=state_dim, action_dim=action_dim)

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            state = next_state
            episode_reward += reward

        agent.update()
        episode_rewards.append(episode_reward)

        if (episode + 1) % 250 == 0:
            avg = np.mean(episode_rewards[-250:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Reward: {episode_reward:5.1f} | "
                f"Avg(250): {avg:5.1f}"
            )

    env.close()
    plot_training_curve(episode_rewards)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "reinforce_rewards.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            data = json.load(f)
    else:
        data = {"algorithm": "REINFORCE", "seeds": {}}
    data.setdefault("seeds", {})[str(seed)] = episode_rewards
    with open(log_path, "w") as f:
        json.dump(data, f)
    print(f"Rewards saved to {log_path} (seed {seed})")

    best = max(episode_rewards)
    print(f"Training complete (seed={seed}). Best episode reward: {best}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    train(seed=args.seed, num_episodes=args.episodes)