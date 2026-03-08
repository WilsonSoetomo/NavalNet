"""
Play tkinter minigame against a given DQN agent.
Usage: python test_dqn.py --model models/dqn.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from battleship import BattleshipEnv
from battleship.opponents import PlayerTKOpponent
from agents import DQNAgent

def play_game(
    agent: DQNAgent,
    env: BattleshipEnv,
    verbose: bool = False,
):
    """
    Run evaluation games. Returns dict with win_rate, avg_shots, shots_std, etc.
    """
    agent.epsilon = 0.0  # Greedy evaluation
    wins = 0
    shots_list: list[int] = []
    rewards_list: list[float] = []

    obs, info = env.reset()

    # Placement
    while info.get("phase") == "placement":
        placement_obs = env.get_placement_observation()
        ship_index = info.get("ship_index", 0)
        mask = env.get_valid_placement_mask()
        action = agent.select_placement_action(
            placement_obs, ship_index, mask, deterministic=True
        )
        obs, _, _, _, info = env.step(action)

    # Shooting
    shots = 0
    total_reward = 0.0
    while not (info.get("agent_won") or info.get("opponent_won")):
        mask = env.get_valid_shooting_mask()
        if mask.sum() == 0:
            break
        action = agent.select_shooting_action(obs, mask, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        shots += 1
        total_reward += reward

    won = bool(info.get("agent_won", False))
    if won:
        print("Agent won!")
    else:
        print("You won!")


def main():
    parser = argparse.ArgumentParser(description="Test DQN agent on Battleship")
    parser.add_argument("--model", type=str, default="models/dqn.pt", help="Path to saved model")
    parser.add_argument("--games", type=int, default=100, help="Number of evaluation games")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (for opponent)")
    parser.add_argument("--verbose", action="store_true", help="Print per-game results")
    parser.add_argument("--save-results", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {args.model}")
        print("Train first with: python train_dqn.py --episodes 2000")
        sys.exit(1)

    env = BattleshipEnv(opponent=PlayerTKOpponent(), seed=args.seed)
    agent = DQNAgent()
    agent.load(model_path)

    print(f"DQN (opponent: Human)")
    print(f"Model: {args.model}")
    print("-" * 50)

    play_game(agent, env, args.games)


if __name__ == "__main__":
    main()
