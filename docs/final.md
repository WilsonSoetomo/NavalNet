---
layout: default
title: Final Report
---

# Final Report

## Video

*Embed your project video here. Example for YouTube:*

```html
<iframe width="560" height="315" src="https://www.youtube.com/embed/YOUR_VIDEO_ID"
  title="NavalNet Demo" frameborder="0" allowfullscreen></iframe>
```

---

## Project Summary

NavalNet is a reinforcement learning project that trains AI agents to play the classic board game Battleship. The game is played on a 10×10 grid where each player places five ships of varying lengths (sizes 2, 3, 3, 4, and 5) and then takes turns shooting at coordinates on the opponent's hidden board. The goal is to sink all five of the opponent's ships before they sink yours.

Battleship is a deceptively rich environment for machine learning research. The core difficulty is **partial observability**: the agent can only see the results of its own shots — hits and misses — but never the positions of the enemy's ships. This means the agent must reason under uncertainty, balancing *exploration* (scanning unknown cells) with *exploitation* (following up on known hits). The game also requires strategic **ship placement**, as a well-placed fleet forces the opponent to spend more shots finding ships. Together, these two subproblems — where to shoot, and where to place your ships — make Battleship a compelling benchmark for evaluating the sample efficiency, exploration behavior, and long-term planning of RL algorithms.

Beyond partial observability, Battleship presents the challenge of a **large, discrete action space** (100 possible shot coordinates) with highly **sparse rewards**: in a 100-shot game, the agent might only observe a handful of meaningful hits, while every other step is noise. Classical approaches such as the Hunt/Target strategy (shoot randomly until a hit, then systematically sink the ship) achieve roughly 53 average shots per game. We asked whether a learned RL policy — without any hard-coded game logic — could discover similar or superior strategies purely from trial and error.

The project explores two major algorithm families: **Deep Q-Networks (DQN)**, a value-based off-policy method that learns expected cumulative rewards (Q-values) for each action in a given state, and **Proximal Policy Optimization (PPO)**, an on-policy policy-gradient method that learns a probability distribution over actions. We built a full Battleship environment from scratch, instrumented it with TensorBoard logging, developed an interactive human evaluation UI, and ran extensive experiments on a university HPC cluster (UCI's HPC3, via SLURM) to compare the two approaches.

---

## Approach

### Environment

We implemented a fully custom Battleship environment in Python, following an OpenAI Gym-style interface. The environment supports two players and handles all game rules: ship placement validation, shot resolution (hit/miss/sink), turn management, and win detection. The environment can be configured with different opponents and reward functions at runtime, making it easy to swap in algorithmic baselines (Random, Hunt/Target, Curriculum) without changing training code.

The environment exposes **three training modes**:

- **Full mode**: Both placement and shooting are trained simultaneously in a live two-player game.
- **Shooting mode**: The agent only trains the shooting head; ship placement is random, and the opponent's turn is disabled. This isolates shot efficiency from win/loss noise.
- **Placement mode**: The agent only trains the placement head, scored by how many shots a Hunt/Target attacker needs to sink all five ships (more shots = better placement).

Training was run on UCI's HPC3 cluster using SLURM batch jobs, with 16 CPU cores and 20 GB RAM per job, running for up to 24 hours per experiment.

### Observation Space

The earliest version of the agent received a single-channel 10×10 grid of integers (0 = unknown, 1 = miss, 2 = hit, 3 = sunk). This proved insufficient: the CNN had to discover through pure reward signal that integer value 2 means "shoot adjacent to this cell," a relationship that requires thousands of episodes to infer from scratch.

We progressively expanded the observation to a **6-channel binary representation**, where each channel encodes a distinct, semantically meaningful aspect of the board state:

| Channel | Meaning |
|---------|---------|
| 0 | Unknown (unshot) cells |
| 1 | Missed cells |
| 2 | Unsunk hit cells (active targets) |
| 3 | Sunk hit cells |
| 4 | Unshot cells adjacent to unsunk hits — high-priority targets |
| 5 | Ship probability heatmap |

**Channel 4** was the most impactful addition. Instead of requiring the CNN to learn the spatial relationship "if (3,4) is a hit and (3,5) is unknown, then (3,5) is a good shot" through trial and error, this adjacency mask is pre-computed and injected directly as an explicit feature plane. The agent can immediately use it to prioritize follow-up shots without discovering the rule from scratch.

**Channel 5** adds a ship probability heatmap, computed by enumerating all valid placements of remaining (unsunk) ships across the board, counting how many placements pass through each unknown cell, and normalizing to [0, 1]. This heatmap naturally encodes the *checkerboard parity* heuristic (ships of length 2 or more cannot be on adjacent cells of the same parity), highlights cells that extend known hit lines, and updates dynamically as ships are sunk. The observation shape is `(6, 10, 10)`, verified to have heatmap values in `[0.29, 1.0]` at game start.

### Neural Network Architecture

**Shooting head (`DQNNetwork` / Actor-Critic policy for PPO):**

The shooting network takes the `(6, 10, 10)` observation and outputs Q-values (DQN) or logits (PPO) over the 100 possible shot coordinates:

```
Input: (6, 10, 10)
Conv2d(6 → 64, kernel=3, pad=1) + ReLU
Conv2d(64 → 64, kernel=3, pad=1) + ReLU
Conv2d(64 → 64, kernel=3, pad=1) + ReLU
Flatten → 6400
Linear(6400 → 256) + ReLU
Linear(256 → 256) + ReLU
Linear(256 → 100)     ← Q-values / logits over 100 cells
```

**Placement head (`PlacementDQNNetwork`):**

The placement network takes the agent's 10×10 board plus a one-hot ship index and outputs Q-values over 200 placement actions (row × col × orientation):

```
Input board: (1, 10, 10)
Conv2d(1 → 32, kernel=3, pad=1) + ReLU
Conv2d(32 → 64, kernel=3, pad=1) + ReLU
Flatten → 6400
Concat with ship one-hot (5-dim) → 6405
Linear(6405 → 256) + ReLU
Linear(256 → 256) + ReLU
Linear(256 → 200)     ← Q-values over 200 placement actions
```

Action masking is applied at inference time: Q-values for already-shot cells (shooting) or invalid placements (placement) are set to `-∞` before the argmax, guaranteeing only legal actions are selected.

### DQN Algorithm

DQN learns a state-action value function Q(s, a) using the Bellman equation:

$$Q(s, a) \leftarrow r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$

The loss minimized is the mean squared error between the predicted Q-value and the Bellman target:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( Q_\theta(s,a) - \left( r + \gamma \max_{a'} Q_{\bar\theta}(s',a') \right) \right)^2 \right]$$

where $\mathcal{D}$ is a replay buffer, and $Q_{\bar\theta}$ is a periodically-synced target network.

**Key DQN hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (Adam) |
| Discount factor γ | 0.99 |
| Replay buffer size | 50,000 |
| Batch size | 64 |
| Target network update | every 500 steps |
| ε start | 1.0 (from scratch) / 0.3–0.4 (from weights) |
| ε end | 0.05–0.10 |
| ε decay | 0.99995 (per step) |
| Gradient clipping | 1.0 (max norm) |
| Buffer reset interval | every 5,000 episodes |

The placement head uses Monte Carlo targets: the episode return is computed at the end of the game and assigned to all placement transitions, rather than bootstrapping with a next-state value.

### PPO Algorithm

PPO is an on-policy policy gradient method that maximizes a clipped surrogate objective to prevent destructive policy updates:

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_\text{old}}(a_t | s_t)$ is the probability ratio and $\hat{A}_t$ is the generalized advantage estimate (GAE). An entropy bonus $\mathcal{H}[\pi_\theta]$ is added to encourage exploration.

**Key PPO hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Entropy coefficient | 0.08 |
| Update epochs per rollout | 8 |
| Rollout steps | 20 |
| Clip range ε | 0.2 (default) |
| Value loss coefficient | 0.5 |

### Reward Function

Reward design was one of the most iterative aspects of the project. Early experiments used a single large win/lose reward (`+100`/`-100`), which drowned out the sparse hit/sink signals. We progressively rebalanced the reward function to provide denser, more informative feedback. The final shooting-mode reward function (which zeros out win/lose signals entirely) is:

| Signal | Value |
|--------|-------|
| Miss penalty (per step) | −1.5 |
| Per-turn living penalty | −0.15 |
| Efficient sink reward | +12.0 |
| Shots-between-sinks penalty | +0.1 (reward for minimizing gaps) |
| Win/loss reward | 0.0 (disabled in shooting mode) |
| Direct hit reward | 0.0 (disabled in shooting mode) |

The **efficient sink** reward is the squared reciprocal of the average shots used to sink a ship relative to its size, rewarding the agent more for sinking ships quickly rather than just eventually.

### Training Curriculum

A key challenge was that the Hunt/Target opponent completes a game in roughly 54 shots, while an untrained agent needs approximately 90 shots. The agent could not win early in training, so every episode yielded the large lose penalty, masking all learning from hits and sinks.

We developed a **gated curriculum opponent** that starts at a specified difficulty level (0 = full random) and ramps toward Hunt/Target behavior as the agent's rolling win rate passes a threshold (40%). This allows the agent to accumulate positive experiences early and only face harder opponents once it has learned basic shot efficiency. The curriculum parameters used in final runs: start difficulty 0.0, end difficulty 0.8, gate win rate 40%, ramp over 10,000 episodes.

We also experimented with initializing DQN from previously-trained weights (setting ε to 0.3 instead of 1.0) so that the agent does not start from random exploration when facing a harder curriculum stage.

---

## Evaluation

### Quantitative Results

After training, we evaluated all agents and human play over a common set of episodes using our interactive web-based evaluation tool. The primary metrics are average shots per episode (lower = better), average shots needed to sink each ship, and shot efficiency (ship size / shots used to sink it, as a percentage).

| Metric | Human (Michael, 9 ep) | Random (100 ep) | Hunt/Target (100 ep) | DQN (10 ep, ep5000 weights) |
|--------|----------------------|-----------------|----------------------|-----------------------------|
| Avg shots/episode | **48.2** | 95.6 | 53.4 | 60.9 |
| Avg shots to sink | 5.5 | 54.5 | 5.8 | 18.1 |
| Avg efficiency | 68.7% | 8.3% | **72.9%** | 39.9% |
| Avg reward | −26.1 | −140.5 | −31.6 | −64.7 |
| Best game (min shots) | 32 | 71 | 30 | 45 |
| Worst game (max shots) | 64 | 100 | 65 | 83 |

The DQN agent substantially outperforms the random baseline (60.9 vs 95.6 average shots) and demonstrates learned behavior (best game: 45 shots), but does not yet match the Hunt/Target algorithmic bot (53.4 shots) or human play (48.2 shots). The gap in average efficiency (39.9% vs 72.9%) reflects the DQN's occasional tendency to wander away from active hits rather than finishing a ship, a failure mode described further below.

### Training Curves and DQN Decay

TensorBoard logging revealed a consistent pattern across DQN runs: performance improves rapidly in the first ~5,000 episodes, then plateaus and gradually decays. This was observed in both full-game and isolated shooting-mode runs.

In the shooting-mode curriculum runs, the best DQN checkpoint (at episode 5,000) achieved approximately 28 average shots to sink all ships. After 50,000 episodes, performance had degraded back toward 55+ shots. The final model used for evaluation was therefore the **weights extracted at the peak (episode 5,000)**, rather than the end of training.

We identified two likely causes of this decay:

1. **Replay Buffer Homogenization**: As the replay buffer fills with on-policy experiences from one strategy, it loses diverse exploratory data. The agent overfits to a local policy and forgets edge cases that were captured earlier in training.
2. **Overestimation Bias**: Standard DQN overestimates Q-values, a bias that compounds over time, causing the agent to prefer actions that were once rewarded but are no longer optimal.

Attempted mitigations included periodic buffer resets (every 5,000 episodes), slower epsilon decay, and higher epsilon floors (ε_end = 0.10). While these helped slow the decay, they did not eliminate it. Proposed future solutions include Prioritized Experience Replay, Double DQN, and cyclic epsilon schedules.

### DQN vs. PPO Comparison

DQN consistently outperformed PPO across all experiments. The smoothed average shots-to-sink for the best DQN run was approximately 28.5 vs. 52.8 for the comparable PPO run at the same episode count.

We attribute this gap to several structural factors:

**Experience Replay vs. On-Policy Discard**: DQN's off-policy replay buffer retains rare successful hit sequences and reuses them for many updates. PPO discards all data after each policy update, losing the few positive experiences from early training when the agent rarely hit anything.

**Discrete Action Fit**: Battleship requires selecting one exact coordinate out of 100. DQN directly assigns a scalar Q-value to each cell and picks the maximum — a natural fit for this structure. PPO must maintain and update a 100-way probability distribution, which is slower to concentrate probability mass onto the correct cells.

**Forced Exploration vs. Entropy Collapse**: DQN's ε-greedy strategy mathematically guarantees board exploration (visiting random cells with probability ε). PPO relies on policy entropy; once PPO found a mediocre but safe strategy, its entropy collapsed and exploration essentially stopped, locking it into a suboptimal policy.

### Qualitative Behavior

Replays recorded via TensorBoard and the interactive game UI show that the DQN agent has learned recognizable Battleship strategies:

- **Hit following**: When the agent lands a hit, it reliably shoots adjacent cells in subsequent turns rather than returning to random search. This behavior was explicitly encouraged by Channel 4 (adjacency mask) and is clearly visible in game replays.
- **Imperfect chaining**: The agent sometimes abandons an active hit cluster to probe a different region of the board, returning to the first cluster only after a miss elsewhere. This reduces efficiency compared to the Hunt/Target bot, which always finishes the current target before moving on. We attempted to fix this by adding a "most recent hit adjacency" channel (biasing the agent toward the *latest* hit rather than *any* unsunk hit), but this change made performance worse rather than better, suggesting the signal introduced conflicting gradients.
- **Parity preference**: The heatmap channel (Channel 5) appears to induce mild checkerboard parity in the agent's search pattern, though this is less crisp than the algorithmic checkerboard search.

### Human Evaluation Tool

We built an interactive browser-based evaluation tool that allows a human player to play against the DQN agent or algorithmic bots (Random, Hunt/Target). The tool renders both the player's defense board and the attack board in real time, logs every shot and its result, and displays running statistics (shots, ships sunk, shot efficiency, reward). A separate "Human Eval" mode records the human player's own statistics over multiple episodes to enable direct head-to-head comparison.

One developer (Michael) completed 9 evaluation episodes and averaged 48.2 shots per game with 68.7% efficiency — better than the DQN agent, confirming that the DQN has not yet reached human-level play, but substantially better than random targeting.

---

## Resources Used

**Papers and Algorithmic References:**

- Mnih et al. (2015), "Human-level control through deep reinforcement learning" (DQN, Nature) — core DQN algorithm and replay buffer design.
- Schulman et al. (2017), "Proximal Policy Optimization Algorithms" — PPO clipped objective and GAE.
- DataGenetics, "Battleship" (http://www.datagenetics.com/blog/december32011/) — analysis of optimal Battleship strategies including Hunt/Target and parity-based search; provided the theoretical baseline and motivation for the ship probability heatmap.

**Libraries and Tools:**

- PyTorch — neural network implementation, training loop, CUDA acceleration.
- NumPy — environment logic and array operations.
- TensorBoard — training visualization and game replay rendering.
- SLURM / UCI HPC3 — distributed job scheduling for 24-hour training runs.
- Python standard library (`collections.deque`, `random`, `pathlib`) — replay buffer implementation.

**AI Tool Usage:**

Claude (Anthropic) was used as a coding assistant throughout the project for: debugging environment logic, suggesting reward function formulations, identifying potential causes of DQN decay (replay buffer homogenization and overestimation bias), and drafting portions of this report. All code was reviewed, tested, and integrated by team members. Claude did not run experiments or make training decisions; those were performed entirely by the team on HPC3.

---

## Future Work

Several directions remain unexplored due to time constraints:

- **Double DQN**: Decoupling action selection from value estimation would reduce the overestimation bias identified as a likely cause of performance decay after episode 5,000.
- **Prioritized Experience Replay (PER)**: Sampling transitions weighted by their temporal-difference error would focus learning on rare, informative experiences (edge cases and efficient sinks) rather than the homogenized majority of the replay buffer.
- **Cyclic Epsilon Schedule**: Periodically resetting ε to a higher value after it decays could inject fresh exploratory data and counteract buffer homogenization without requiring a full training restart.
- **Dueling DQN**: Separating the value stream (how good is this board state?) from the advantage stream (how much better is this shot than average?) may produce more stable learning in the sparse-reward Battleship setting.
- **Evolutionary / Grid-Search Hyperparameter Optimization**: Reward weights (miss penalty, efficient sink bonus, per-turn penalty) were tuned manually through trial and error. A systematic grid search or evolutionary strategy would likely find better configurations.
- **Self-Play**: Training the agent against its own previous checkpoints (rather than fixed algorithmic opponents) is a natural next step toward discovering emergent strategies that go beyond what the Hunt/Target baseline can teach.

---

*NavalNet — Michael Ip, Dylan Tanaka, Wilson Soetomo   — CS 175, Spring 2026*