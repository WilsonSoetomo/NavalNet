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