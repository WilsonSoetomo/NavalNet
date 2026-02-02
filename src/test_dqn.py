"""
Test / evaluate a trained DQN agent on Battleship.
Usage: python test_dqn.py --model models/dqn.pt [--games N] [--save-results]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from battleship import BattleshipEnv
from battleship.opponents import RandomOpponent
from agents import DQNAgent


def evaluate(
    agent: DQNAgent,
    env: BattleshipEnv,
    num_games: int,
    verbose: bool = False,
) -> dict:
    """
    Run evaluation games. Returns dict with win_rate, avg_shots, shots_std, etc.
    """
    agent.epsilon = 0.0  # Greedy evaluation
    wins = 0
    shots_list: list[int] = []
    rewards_list: list[float] = []

    for g in range(num_games):
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
        wins += int(won)
        shots_list.append(shots)
        rewards_list.append(total_reward)

        if verbose:
            outcome = "WIN" if won else "LOSS"
            print(f"  Game {g+1}: {outcome} in {shots} shots (reward={total_reward:.1f})")

    return {
        "win_rate": wins / num_games,
        "wins": wins,
        "total_games": num_games,
        "avg_shots": np.mean(shots_list),
        "std_shots": np.std(shots_list),
        "min_shots": int(np.min(shots_list)),
        "max_shots": int(np.max(shots_list)),
        "avg_reward": np.mean(rewards_list),
    }


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

    env = BattleshipEnv(opponent=RandomOpponent(), seed=args.seed)
    agent = DQNAgent()
    agent.load(model_path)

    print(f"Evaluating DQN on {args.games} games (opponent: Random)")
    print(f"Model: {args.model}")
    print("-" * 50)

    results = evaluate(agent, env, args.games, verbose=args.verbose)

    print(f"Win rate:     {100 * results['win_rate']:.1f}% ({results['wins']}/{results['total_games']})")
    print(f"Avg shots:    {results['avg_shots']:.1f} (+/- {results['std_shots']:.1f})")
    print(f"Shots range:  {results['min_shots']} - {results['max_shots']}")
    print(f"Avg reward:   {results['avg_reward']:.1f}")

    if args.save_results:
        import json
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
