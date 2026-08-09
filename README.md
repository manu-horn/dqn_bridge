# DQN Bridge: Replicating Classic RL Papers on CartPole-v1

A self-contained project that re-implements three well-known reinforcement learning algorithms from their original papers and benchmarks them on the Gymnasium `CartPole-v1` environment.

## Objective

Demonstrate faithful replication of ideas from foundational RL papers:

| Algorithm | Reference | Key Idea |
| :--- | :--- | :--- |
| **DQN** | Mnih et al. (2013, 2015) | Experience replay, target network, Huber loss to stabilize Q-learning with function approximation |
| **Double DQN** | van Hasselt et al. (2015) | Decoupled action selection (online net) and evaluation (target net) to reduce overestimation bias |
| **REINFORCE** | Williams (1992), Sutton et al. (1999) | Monte Carlo policy gradient with a categorical policy and a cross-episode standardized-return baseline |

Each agent is designed to match its paper's specifications (replay buffer size, batch size, target update interval, loss function, exploration schedule) while sharing an identical training loop and logging format so results are directly comparable.

## Project Structure

```
dqn_bridge/
├── dqn.py                  # DQN agent + training loop (Mnih 2013/2015)
├── double_dqn.py           # Double DQN agent + training loop (van Hasselt 2015)
├── reinforce.py            # REINFORCE agent + training loop (Williams 1992)
├── utils.py                # Aggregates runs across seeds and plots comparison charts
├── AGENTS.md               # Agent architecture & hyperparameter specifications
├── requirements.txt
├── run_all.sh               # Trains all agents across seeds, then runs utils.py to compare
├── logs/                   # Per-algorithm reward histories (JSON), grouped by seed key
├── comparison_500ep.png   # All algorithms, 500-episode budget, mean ± std across seeds
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train all three algorithms for 500 episodes across 3 seeds, then compare:

```bash
bash run_all.sh
```

By default this runs seeds `42 43 44` at 500 episodes each, then `utils.py` to aggregate and plot. Both can be overridden with environment variables:

```bash
SEEDS="42 43 44 45 46" EPISODES=1000 bash run_all.sh
```

You can also train a single agent manually. Defaults: `--seed 42`, `--episodes 500`. Each run's raw per-episode rewards are appended to `logs/<algorithm>_rewards.json`, grouped by seed key.

```bash
python dqn.py --seed 42
python double_dqn.py --seed 43
python reinforce.py --seed 44
```

Compare trained runs:

```bash
python utils.py
```

This reads every JSON in `logs/`, groups curves by episode budget, and averages each algorithm's 30-episode moving average **across seeds**, shading mean ± standard deviation. Since all three agents train for the same budget (500 episodes by default), one figure, `comparison_500ep.png`, contains all three for a fair comparison.

## Logging Contract

Each training run writes its raw per-episode total rewards to `logs/<algorithm>_rewards.json`, merging under a seed key:

```json
{"algorithm": "DQN", "seeds": {"42": [24.0, 12.0, ...], "43": [...]}}
```

## Environment

- `CartPole-v1` (Gymnasium)
- 4-dimensional continuous observation, 2 discrete actions
- Terminal reward = 500 per episode; solve threshold typically `>= 475.0`

## References

- Mnih, V. et al. (2013). Playing Atari with Deep Reinforcement Learning.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning.
- van Hasselt, H. et al. (2015). Deep Reinforcement Learning with Double Q-learning.
- Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.
- Sutton, R. S. et al. (1999). Policy Gradient Methods for Reinforcement Learning with Function Approximation.