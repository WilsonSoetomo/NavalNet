"""
Train DQN agent for Battleship.
Usage: python train_dqn.py [--episodes N] [--save-path PATH] [--seed N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from battleship import BattleshipEnv
from battleship.opponents import RandomOpponent
from agents import DQNAgent


def run_episode(
    env: BattleshipEnv,
    agent: DQNAgent,
    train: bool = True,
) -> tuple[float, int, bool]:
    """
    Run one episode. Returns (total_reward, num_shots, agent_won).
    Collects placement transitions during placement; at end of episode assigns
    episode return to each placement step and updates the placement head.
    """
    obs, info = env.reset()
    total_reward = 0.0
    num_shots = 0
    prev_obs: np.ndarray | None = None
    prev_action: int | None = None
    prev_mask: np.ndarray | None = None
    placement_transitions: list[tuple[np.ndarray, int, int]] = []

    # Placement phase: collect (placement_obs, ship_index, action) for each step
    while info.get("phase") == "placement":
        placement_obs = env.get_placement_observation()
        ship_index = info.get("ship_index", 0)
        mask = env.get_valid_placement_mask()
        action = agent.select_placement_action(
            placement_obs, ship_index, mask, deterministic=not train
        )
        placement_transitions.append((placement_obs.copy(), ship_index, action))
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward

    # Shooting phase
    while not (info.get("agent_won") or info.get("opponent_won")):
        mask = env.get_valid_shooting_mask()
        if mask.sum() == 0:
            break

        action = agent.select_shooting_action(obs, mask, deterministic=not train)
        prev_obs = obs.copy()
        prev_action = action
        prev_mask = mask.copy()

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        num_shots += 1

        # Store transition for DQN (only shooting steps)
        if train and prev_obs is not None and prev_action is not None:
            next_mask = env.get_valid_shooting_mask()
            agent.store_transition(
                prev_obs, prev_action, reward, obs, term or trunc, next_mask
            )
            agent.update()

    # Placement learning: assign episode return to each placement step (Monte Carlo)
    if train and placement_transitions:
        for placement_obs, ship_index, action in placement_transitions:
            agent.store_placement_transition(
                placement_obs, ship_index, action, total_reward
            )
        for _ in range(len(placement_transitions)):
            agent.update_placement()

    return total_reward, num_shots, bool(info.get("agent_won", False))


def main():
    parser = argparse.ArgumentParser(description="Train DQN on Battleship")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes")
    parser.add_argument("--save-path", type=str, default="models/dqn.pt", help="Model save path")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N episodes")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluate every N episodes")
    parser.add_argument("--eval-games", type=int, default=20, help="Games per evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    env = BattleshipEnv(opponent=RandomOpponent(), seed=args.seed)
    agent = DQNAgent(
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        buffer_size=50_000,
        batch_size=64,
        target_update_freq=500,
        seed=args.seed,
    )

    wins = 0
    total_shots = 0
    rewards_history: list[float] = []

    print(f"Training DQN for {args.episodes} episodes...")
    print(f"  Opponent: Random")
    print(f"  Save path: {args.save_path}")
    print("-" * 50)

    for ep in range(1, args.episodes + 1):
        reward, shots, won = run_episode(env, agent, train=True)
        wins += int(won)
        total_shots += shots
        rewards_history.append(reward)

        if ep % args.eval_every == 0:
            # Evaluate without exploration
            eval_wins = 0
            eval_shots: list[int] = []
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0
            for _ in range(args.eval_games):
                _, shots, won = run_episode(env, agent, train=False)
                eval_wins += int(won)
                eval_shots.append(shots)
            agent.epsilon = old_epsilon

            recent = rewards_history[-100:]
            print(
                f"Ep {ep:5d} | "
                f"Win% {100*eval_wins/args.eval_games:5.1f} | "
                f"AvgShots {np.mean(eval_shots):5.1f} | "
                f"eps {agent.epsilon:.3f} | "
                f"R_avg {np.mean(recent):.1f}"
            )

        if ep % args.save_every == 0:
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            agent.save(args.save_path)
            print(f"  Saved to {args.save_path}")

    agent.save(args.save_path)
    print("-" * 50)
    print(f"Training complete. Final model saved to {args.save_path}")


if __name__ == "__main__":
    main()
