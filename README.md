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
├── utils.py                # Compares runs from logs/ and plots per-budget comparison charts
├── AGENTS.md               # Agent architecture & hyperparameter specifications
├── requirements.txt
├── logs/                   # Per-algorithm reward histories (JSON)
│   ├── dqn_rewards.json
│   ├── double_dqn_rewards.json
│   └── reinforce_rewards.json
├── comparison_500ep.png    # DQN vs Double DQN (500-episode budget)
└── comparison_1000ep.png   # REINFORCE (1,000-episode budget)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train each algorithm:

```bash
python dqn.py          # 500 episodes → logs/dqn_rewards.json
python double_dqn.py   # 500 episodes → logs/double_dqn_rewards.json
python reinforce.py    # 1,000 episodes → logs/reinforce_rewards.json
```

Each run saves its raw per-episode rewards to a JSON log in `logs/`. REINFORCE uses a larger budget because it performs only a single gradient update per episode and, being on-policy, requires more rollouts to converge.

Compare trained runs:

```bash
python utils.py
```

This reads every JSON in `logs/` and groups curves by episode budget so runs are only plotted on comparable axes. It produces one figure per budget, each with 30-episode moving-average lines and shaded standard-deviation bands: `comparison_500ep.png` (DQN vs Double DQN) and `comparison_1000ep.png` (REINFORCE).

## Logging Contract

Each training script writes its raw per-episode total rewards to `logs/<algorithm>_rewards.json` in the format:

```json
{"algorithm": "DQN", "rewards": [24.0, 12.0, ...]}
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