# AGENTS.md — dqn_bridge

Single-file PyTorch DQN implementation for CartPole-v1 (Phase 1).

## Commands

```bash
pip install -r requirements.txt   # install deps (torch, gymnasium, numpy, matplotlib)
python dqn.py                      # train DQN for 500 episodes
```

## Structure

- `dqn.py` — self-contained training script with `QNetwork`, `ReplayBuffer`, `DQNAgent`, and `train()`
- `requirements.txt` — exact dependencies
- `training_curve.png` — output plot (auto-generated after training)

## Code conventions

- Ring-buffer `ReplayBuffer` uses NumPy arrays internally; `sample()` returns PyTorch tensors
- Epsilon decays linearly over `epsilon_decay` steps, not episodes
- Target network syncs every `target_update_freq` gradient steps
- MSE loss on TD targets: $r + \gamma (1-d) \max_{a'} Q(s', a'; \theta^-)$
- Running `python dqn.py` is the only entrypoint; no test framework configured yet
