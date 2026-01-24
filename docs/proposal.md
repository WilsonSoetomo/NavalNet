---
layout: default
title:  Proposal
---

# {{ page.title }}


## 2.2 Summary of the Project

The goal of this project is to build an AI agent that is competent in playing the classic board game Battleship by learning effective strategies. The agent receives as input an observed game board which logs previous hits and misses, and continuously outputs the next location to attack. The system aims to minimize the number of moves required to sink all of the opponent's ships while maximizing win rate against baseline strategies. Additionally, it should be able to rate on a scale of 1-10 the effectiveness of various ship placement compositions.

## 2.3 Project Goals

#### Minimum Goal
Create a basic agent that uses a probabilistic or heuristic based strategy to choose successive bombing locations on a game board in a way that at least outperforms choosing random bombing locations.

#### Realistic Goal
Train an agent using reinforcement learning, such as Q-learning or Monte Carlo, that independently learns bombing strategies from self-play and consistently beats standard heuristic strategies such as hunt and target or checkerboard search.

#### Moonshot Goal
Create an evolving system where two agents learn successful/optimal strategies for both bombing and ship placement, dynamically exploits opponent behaviour, and consistently matches or even exceeds performance of competent human opponents.

## 2.4 AI/ML Algorithms
The main algorithm we will be using is either q-learning or proximal probability estimation, assisted by Monte Carlo Tree Search to improve likelihood of finding ship locations.

## 2.5 Evaluation Plan
#### Quantitative Evaluation
We will evaluate the agent by running large batches of simulated Battleship games and measuring the average number of moves to win. Baseline strategies that can be used for comparison will include random bombing, checkerboard bombing pattern, and hunt and target (that is, upon obtaining a hit, attack around the hit to find the remaining parts of the ship). Using this measurement, we can also compare the effectiveness of different training methods, such as q-learning and policy gradient. We estimate our agent should be able to improve on the average moves required of at least the basic heuristics (random, checkerboard) by 15- 30%.

#### Qualitative Evaluation
We can visualize games played by our agent using heatmaps of shot probabilities and keeping a record of games. This way, we can verify that the agent is indeed learning from its mistakes, and if not, observe the cases where the agent performed poorly and make corrections. Another way we can qualitatively evaluate our agent is simply to have a human play some games against it, and observing the bombing locations of the agent during the game. If possible, we can also have the model play itself and self-learn for multiple iterations. For a successful result, we expect to observe non-random bombing patterns and/or some sort of strategy.

## 2.6 AI tool usage
We will be utilizing LLMs such as Claude, Gemini and ChatGPT for idea evaluation, advice, and for obtaining baseline understanding of machine learning concepts, but no direct implementation. Code and logic will be explicitly overlooked and validated by the project team to ensure no “hallucinations” are to occur.


---
layout: default
title: Proposal
---

## 2.2 Summary

The goal of this project is to build an AI agent that is competent in playing the classic game of Battleship by learning effective bombing and ship placement strategies. The agent receives as input an observed game board and previous hits and misses, which it will then output the next grid to attack. The system aims to minimize the number of moves required to sink all of the opponent's ships while maximizing win rate against baseline strategies. It should also on the side be able to rate on a scale of 1-10 how certain battleship compositions perform when we set one up.

## 2.3 Project Goals

### Minimum Goal
Implement basic agents that use probabilistic or heuristic based strategy to choose the next bombing locations that outperform random play. 

### Realistic Goal
Train a reinforcement learning agent such as Q-learning or Monte Carlo that learns bombing strategies from self-play and consistently beats standard heuristic strategies such as a hunt and target mechanism or checkerboard search.

### Moonshot Goal
Create an evolving system where two agents learn both bombing and ship placement strategies that dynamically exploit opponent behaviour where it matches or even beats challenging a competent human opponent.

## 2.4 AI/ML Algorithms
The main algorithm we will be using value based learning combiner with bayesian probability estimation and Monte Carlo simulation to model ship likelihood under some observability. 

## 2.5 Evaluation Plan

### Quantitative Evaluation
We will evaluate the agent by running large batches of simulated Battleship games and measuring average number of moves to win, win rate against baseline agents. The baselines will be random bombing agent, checkerboard heuristic and hunt and target strategy. We expect results of our AI to beat heuristics moves to win by 25 - 40%

### Qualitative Evaluation
We can visualize heatmaps of shot probabilities, game replays showing bad decision and failure cases where the agent performs poorly. These visualizations will be used to verify that the agent is indeed learning from its mistakes.

## 2.6 AI tool usage
The tools we will be using will stem from Claude, Gemini and ChatGPT in the form of asking advice and no direct implementation. Code and logic will be explicitly overlooked and validated by the project team to ensure no "hallucinations" are to occur.