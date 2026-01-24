---
layout: default
title:  Proposal
---

# {{ page.title }}


## 2.2 Summary of the Project

The goal of this project is to build an AI agent that is competent in playing the classic board game Battleship by learning effective strategies. The agent first places boats in a certain pattern then receives an observed game board which logs previous hits and misses as inupt, and continuously outputs the next location to attack. The system aims to minimize the number of moves required to sink all of the opponent's ships while maximizing win rate against baseline strategies. We will compare two methods of reinforcement learning, Deep Q Networks (DQN) and Proximal Policy Optimization (PPO). We will also explore how Monte Carlo Tree Search can improve these algorithms. 

## 2.3 Project Goals

#### Minimum Goal
Create a basic agent that uses a probabilistic / heuristic based strategy to choose successive bombing locations on a game board in a way that at least outperforms choosing random bombing locations.

#### Realistic Goal
Train an agent using reinforcement learning, using both DQN and PPO, that independently learns bombing strategies from playing against heuristic strategies such as hunt and target or checkerboard search.

#### Moonshot Goal
Create an evolving self-play system where models learn successful/optimal strategies for both bombing and ship placement by playing itself over many iterations and consistently matches or even exceeds performance of competent human opponents.

## 2.4 AI/ML Algorithms
The main algorithm we will be using is both q-learning or proximal probability optimization, assisted by Monte Carlo Tree Search to improve likelihood of finding ship locations.

## 2.5 Evaluation Plan
#### Quantitative Evaluation
We will evaluate the agent by running large batches of simulated Battleship games and measuring statistics like the average number of moves to win, number of moves to sink a ship and number of moves between finding ships. Baseline strategies that can be used for comparison will include random bombing, checkerboard bombing pattern, and hunt and target (that is, upon obtaining a hit, attack around the hit to find the remaining parts of the ship). Using this measurement, we can also compare the effectiveness of different training methods, such as q-learning and policy gradients.

#### Qualitative Evaluation
We can visualize games played by our agent using heatmaps of shot probabilities and keeping a record of games. This way, we can verify that the agent is indeed learning from its mistakes, and if not, observe the cases where the agent performed poorly and make corrections. Another way we can qualitatively evaluate our agent is simply to have a human play some games against it, and observing the bombing locations of the agent during the game. If possible, we can also have the model play itself and self-learn for multiple iterations. For a successful result, we expect to observe non-random bombing patterns and/or some sort of clear strategy.

## 2.6 AI tool usage
We will be utilizing LLMs such as Claude, Gemini and ChatGPT for idea evaluation, advice, and for obtaining baseline understanding of machine learning concepts, but no direct implementation. Code and logic will be explicitly overlooked and validated by the project team to ensure no “hallucinations” are to occur.
