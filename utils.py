import json
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch


MA_WINDOW = 30


def set_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)
    return env


def scan_logs(log_dir="logs"):
    runs = {}
    if not os.path.isdir(log_dir):
        return runs
    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(log_dir, filename)
        with open(path, "r") as f:
            data = json.load(f)
        algorithm = data.get("algorithm", filename.split("_")[0])
        seeds = data.get("seeds")
        if seeds is None:
            seeds = {str(data.get("seed", 0)): data.get("rewards", [])}
        for seed, rewards in seeds.items():
            runs.setdefault(algorithm, {})[seed] = np.asarray(rewards, dtype=np.float32)
    return runs


def moving_average(data, window):
    if len(data) < window:
        return np.array(data, dtype=np.float32)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_comparison(algorithms, save_path="comparison.png"):
    plt.figure(figsize=(12, 6))
    for algorithm, seed_rewards in algorithms.items():
        if len(seed_rewards) == 0:
            print(f"Skipping {algorithm}: no rewards")
            continue
        ma_list = [moving_average(r, MA_WINDOW) for r in seed_rewards.values()]
        length = min(len(m) for m in ma_list)
        ma_arr = np.stack([m[:length] for m in ma_list])
        mean_ma = ma_arr.mean(axis=0)
        std_ma = ma_arr.std(axis=0)
        offset = MA_WINDOW - 1 if length else 0
        x = np.arange(offset, offset + length)
        plt.plot(x, mean_ma, label=algorithm)
        plt.fill_between(x, mean_ma - std_ma, mean_ma + std_ma, alpha=0.25)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("CartPole-v1 Benchmark (mean \u00b1 std across seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def run_comparison(log_dir="logs"):
    runs = scan_logs(log_dir)
    if not runs:
        print(f"No JSON files found in {log_dir}/")
        return

    budgets = {}
    for algorithm, seed_rewards in runs.items():
        n = next(iter(seed_rewards.values())).shape[0] if seed_rewards else 0
        budgets.setdefault(n, {})[algorithm] = seed_rewards

    for episode_count, algorithms in budgets.items():
        n_seeds = len(next(iter(algorithms.values())))
        for algorithm, seed_rewards in algorithms.items():
            rewards = np.stack(list(seed_rewards.values()))
            ma_arr = np.stack([moving_average(r, MA_WINDOW) for r in rewards])
            mean_ma = ma_arr.mean(axis=0)
            std_ma = ma_arr.std(axis=0)
            stats = {
                "seeds": len(rewards),
                "mean": float(rewards.mean()),
                "std": float(rewards.std()),
                "last_ma_mean": float(mean_ma[-1]),
                "last_ma_std": float(std_ma[-1]),
                "best": float(rewards.max()),
            }
            print(f"{algorithm} ({episode_count} episodes, {stats['seeds']} seeds): {stats}")

        save_path = f"comparison_{episode_count}ep.png"
        plot_comparison(algorithms, save_path)


if __name__ == "__main__":
    run_comparison()