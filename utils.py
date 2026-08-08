import json
import os

import numpy as np
import matplotlib.pyplot as plt


MA_WINDOW = 30
STD_WINDOW = 30


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
        rewards = data.get("rewards", [])
        runs[algorithm] = np.asarray(rewards, dtype=np.float32)
    return runs


def moving_average(data, window):
    if len(data) < window:
        return np.array(data, dtype=np.float32)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def moving_std(data, window):
    if len(data) < window:
        return np.zeros_like(data, dtype=np.float32)
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    cumsum_sq = np.cumsum(np.insert(np.square(data), 0, 0.0))
    count = np.arange(window, len(data) + 1, dtype=np.float32)
    mean = (cumsum[window:] - cumsum[:-window]) / window
    var = (cumsum_sq[window:] - cumsum_sq[:-window]) / window - np.square(mean)
    return np.sqrt(np.clip(var, 0.0, None))


def plot_comparison(runs, save_path="comparison.png"):
    plt.figure(figsize=(12, 6))
    for algorithm, rewards in runs.items():
        if len(rewards) == 0:
            print(f"Skipping {algorithm}: no rewards")
            continue
        ma = moving_average(rewards, MA_WINDOW)
        std = moving_std(rewards, STD_WINDOW)
        offset = MA_WINDOW - 1 if len(rewards) >= MA_WINDOW else 0
        x = np.arange(offset, offset + len(ma))
        plt.plot(x, ma, label=algorithm)
        plt.fill_between(x, ma - std, ma + std, alpha=0.25)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("CartPole-v1 Benchmark")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")


def run_comparison(log_dir="logs", save_path="comparison.png"):
    runs = scan_logs(log_dir)
    if not runs:
        print(f"No JSON files found in {log_dir}/")
        return
    for algorithm, rewards in runs.items():
        stats = {"mean": float(np.mean(rewards)), "std": float(np.std(rewards))}
        if len(rewards) >= MA_WINDOW:
            ma = moving_average(rewards, MA_WINDOW)
            stats["mean_moving_avg"] = float(ma[-1])
            stats["best"] = float(np.max(rewards))
        print(f"{algorithm}: {stats}")
    plot_comparison(runs, save_path)


if __name__ == "__main__":
    run_comparison()