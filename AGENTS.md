# Agent Architecture & Specifications (`CartPole-v1`)

This document details the mathematical specifications, loss formulations, and hyperparameter configurations for the three agents implemented in this project, replicating implementations from well-known reinforcement learning papers.

---

## 1. Deep Q-Network (DQN)
* **Reference:** Mnih et al. (2013, 2015)
* **Paradigm:** Off-policy, Value-based Deep Reinforcement Learning

### Core Architecture
* **Policy/Value Network:** MLP ($4 \rightarrow 64 \rightarrow 2$) with ReLU activations.
* **Replay Buffer:** FIFO memory queue $\mathcal{D}$ with maximum capacity $N = 10,000$. Samples mini-batches of size $B = 64$ uniformly at random $(s, a, r, s', d) \sim U(\mathcal{D})$ once buffer contains $\ge 1,000$ steps.
* **Target Network:** Independent network weights $\theta^-$ updated via hard copy ($\theta^- \leftarrow \theta$) every $C = 500$ environment steps.

### Mathematical Formulation
* **TD Target Calculation:**
  $$y_i = r_i + \gamma (1 - d_i) \max_{a'} Q(s'_i, a'; \theta^-)$$
* **Loss Function:** Smooth L1 (Huber) Loss to stabilize gradient steps against outlier TD errors:
  $$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \text{Huber}\left( y_i - Q(s_i, a_i; \theta) \right)$$
* **Exploration Schedule:** $\epsilon$-greedy annealed linearly from $1.0$ down to $0.05$ over $5{,}000$ environment steps.

---

## 2. Double Deep Q-Network (Double DQN)
* **Reference:** van Hasselt et al. (2015)
* **Paradigm:** Off-policy, Value-based (Decoupled Action Selection & Evaluation)

### Core Architecture
* **Identical Baseline:** Uses the same network architecture, replay buffer capacity, learning rate ($10^{-3}$), and target copy frequency ($C = 500$) as standard DQN to isolate the impact of overestimation bias reduction.

### Mathematical Formulation
* **Decoupled TD Target Calculation:**
  1. *Action Selection* via online network $\theta$:
     $$a^* = \arg\max_{a'} Q(s'_i, a'; \theta)$$
  2. *Action Evaluation* via target network $\theta^-$:
     $$y_i = r_i + \gamma (1 - d_i) Q(s'_i, a^*; \theta^-)$$
* **Loss Function:** Smooth L1 (Huber) Loss over the decoupled target $y_i$.

---

## 3. REINFORCE (Monte Carlo Policy Gradient)
* **Reference:** Williams (1992), Sutton et al. (1999)
* **Paradigm:** On-policy, Direct Policy Optimization

### Core Architecture
* **Policy Network:** MLP ($4 \rightarrow 128 \rightarrow 2$) outputting unnormalized logits passed into a `Categorical` action distribution $\pi_\theta(a|s)$.
* **Memory Strategy:** Full trajectory on-policy rollouts $(s_0, a_0, r_0), \dots, (s_T, a_T, r_T)$. No replay buffer or target network is used. Trajectory memory is wiped completely after every gradient update.
* **Training Budget:** Trained for $1{,}000$ episodes (versus $500$ for DQN and Double DQN), since REINFORCE performs a single gradient update per episode and requires more on-policy rollouts to converge.

### Mathematical Formulation
* **Backward Discounted Return:**
  $$G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$$
* **Variance Reduction (Cross-Episode Standardized Baseline):**
  $$\hat{G}_t = \frac{G_t - \mu_G}{\sigma_G + 10^{-8}}$$
  where $\mu_G$ and $\sigma_G$ are the mean and standard deviation of episode returns accumulated **across previously seen episodes**, so the signal preserves whether an episode performed above or below the running performance average. (Standardizing within a single trajectory would erase this between-episode signal.)
* **Objective & Loss Function:** Minimizes the negative log-likelihood weighted by standardized return:
  $$\mathcal{L}(\theta) = - \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) \cdot \hat{G}_t$$
* **Update Frequency:** Once per complete episode using Adam optimizer ($\alpha = 10^{-3}$).

---

## Summary Hyperparameter Comparison

| Parameter | DQN | Double DQN | REINFORCE |
| :--- | :--- | :--- | :--- |
| **Learning Rate ($\alpha$)** | $10^{-3}$ | $10^{-3}$ | $10^{-3}$ |
| **Discount Factor ($\gamma$)** | $0.99$ | $0.99$ | $0.99$ |
| **Data Collection** | Off-policy (Replay Buffer) | Off-policy (Replay Buffer) | On-policy (Episode Rollout) |
| **Buffer Capacity** | $10,000$ | $10,000$ | $N/A$ |
| **Batch Size** | $64$ | $64$ | Complete Episode |
| **Target Update ($C$)** | Every $500$ steps | Every $500$ steps | $N/A$ |
| **Loss Criteria** | Huber Loss | Huber Loss | Policy Negative Log-Likelihood |
| **Episode Budget** | $500$ | $500$ | $1{,}000$ |
| **Output File** | `logs/dqn_rewards.json` | `logs/double_dqn_rewards.json` | `logs/reinforce_rewards.json` |