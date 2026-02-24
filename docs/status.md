how do ---
layout: default
title:  Status
---

# {{ page.title }}

## Project Overview

NavalNet is a reinforcement learning project that trains AI agents to play Battleship using Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO). The project aims to develop agents that can learn effective strategies through self-play and outperform baseline heuristic strategies.

## Completed Components

### 1. Game Engine & Environment
- ✅ **Battleship Game Engine**: Fully implemented game logic with ship placement, shooting mechanics, and win/loss detection
- ✅ **Gymnasium Environment**: Custom Battleship environment (`BattleshipEnv`) compatible with RL algorithms
- ✅ **Board State Representation**: Efficient encoding of game state for neural network input
- ✅ **Opponent Strategies**: Implemented baseline opponents including:
  - Random opponent
  - Hunt and target strategy
  - Checkerboard pattern strategy

### 2. Reinforcement Learning Agents
- ✅ **DQN Agent**: Deep Q-Network implementation with:
  - Experience replay buffer
  - Target network for stable training
  - Epsilon-greedy exploration strategy
  - Configurable hyperparameters (learning rate, epsilon decay, etc.)
- ✅ **PPO Agent**: Proximal Policy Optimization implementation with:
  - Actor-critic architecture
  - Generalized Advantage Estimation (GAE)
  - Policy clipping for stable updates
  - Value function and entropy regularization

### 3. Training Infrastructure
- ✅ **Training Scripts**: Separate training scripts for DQN and PPO with comprehensive logging
- ✅ **SLURM Integration**: Batch job scripts for running training on HPC clusters
- ✅ **TensorBoard Logging**: Real-time visualization of:
  - Training metrics (rewards, win rates, average shots)
  - Evaluation metrics
  - Game board visualizations
  - Step-by-step game replays

### 4. Visualization & Evaluation
- ✅ **Board Rendering**: Visual representation of game boards for TensorBoard
- ✅ **Testing Framework**: Scripts to evaluate trained models against baseline strategies
- ✅ **Performance Metrics**: Tracking of win rates, average shots per game, and other key statistics

## Current Progress

### Minimum Goal: ✅ Achieved
The project has successfully created agents that outperform random strategies. Both DQN and PPO agents have been trained and demonstrate learning capabilities.

### Realistic Goal: 🚧 In Progress
- ✅ Agents can learn from playing against heuristic strategies
- ✅ Both DQN and PPO implementations are functional
- 🚧 Ongoing training and hyperparameter tuning
- 🚧 Performance optimization against various baseline strategies

### Moonshot Goal: 📋 Planned
- Self-play system for continuous improvement
- Learning optimal ship placement strategies
- Advanced strategy development through iterative training

## Technical Implementation Details

### Architecture
- **Framework**: PyTorch for neural network implementation
- **RL Library**: Stable-Baselines3 compatible architecture
- **Environment**: Custom Gymnasium environment
- **Visualization**: TensorBoard for training monitoring

### Training Configuration
- **Episodes**: 10,000 episodes per training run
- **Evaluation**: Every 100 episodes with 20 evaluation games
- **Model Saving**: Checkpoints every 500 episodes
- **Logging**: Comprehensive TensorBoard logs with game visualizations

### Infrastructure
- **HPC Cluster**: Training on UCI HPC cluster using SLURM
- **Resource Allocation**: 8 CPU cores, 20GB memory per job
- **Job Management**: Automated job submission and monitoring

## Challenges & Solutions

### Challenge 1: State Space Representation
**Problem**: Efficiently encoding the Battleship game state for neural networks.

**Solution**: Developed a compact state representation that includes:
- Hit/miss history
- Ship placement information
- Game phase tracking

### Challenge 2: Training Stability
**Problem**: Initial training instability with both DQN and PPO.

**Solution**: 
- Implemented target networks for DQN
- Added proper value function clipping for PPO
- Tuned hyperparameters (learning rates, discount factors, etc.)

### Challenge 3: HPC Integration
**Problem**: Setting up training pipeline on HPC cluster with proper resource allocation.

**Solution**: Created SLURM batch scripts with appropriate resource requests and environment setup.

## Results & Metrics

### Training Progress
- Models are currently being trained on the HPC cluster
- TensorBoard logs show learning curves and game visualizations
- Both DQN and PPO agents demonstrate improving performance over time

### Evaluation Metrics (To be updated with actual results)
- Win rate against random opponent
- Average shots per game
- Comparison between DQN and PPO performance
- Performance against heuristic baselines

## Next Steps

### Short-term Goals
1. **Complete Training Runs**: Finish full training cycles for both DQN and PPO
2. **Performance Evaluation**: Comprehensive testing against all baseline strategies
3. **Hyperparameter Tuning**: Optimize learning rates, network architectures, and training schedules
4. **Model Comparison**: Detailed analysis of DQN vs PPO performance

### Medium-term Goals
1. **Advanced Opponents**: Implement more sophisticated baseline strategies
2. **Strategy Analysis**: Visualize and understand learned strategies
3. **Monte Carlo Tree Search**: Explore MCTS integration for improved decision-making

### Long-term Goals
1. **Self-Play System**: Implement iterative self-play for continuous improvement
2. **Ship Placement Learning**: Extend agents to learn optimal ship placement
3. **Human-Level Performance**: Achieve performance comparable to skilled human players

## Repository & Resources

- **Source Code**: [GitHub Repository](https://github.com/WilsonSoetomo/NavalNet)
- **Training Logs**: Available via TensorBoard on HPC cluster
- **Documentation**: See README files and SLURM documentation for usage instructions

## Team Contributions

- **wsoetomo**: [Contributions]
- **tanakadm**: [Contributions]
- **mip1**: [Contributions]

---

*Last Updated: [Current Date]*