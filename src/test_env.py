"""Quick smoke test of the Battleship environment."""

import gymnasium as gym

# Register and use local module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from battleship import BattleshipEnv, RandomOpponent

def main():
    env = BattleshipEnv(opponent=RandomOpponent(), seed=42)
    obs, info = env.reset()

    print("Observation shape:", obs.shape, "(10x10 attack board)")
    print("Phase:", info["phase"])
    print("Ship index:", info["ship_index"])

    # Placement phase: place 5 ships
    for i in range(5):
        mask = env.get_valid_placement_mask()
        valid_actions = mask.nonzero()[0]
        if len(valid_actions) == 0:
            print("No valid placement - bug")
            break
        action = int(valid_actions[0])  # Pick first valid
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Place ship {i+1}: reward={reward}, phase={info['phase']}")

    # Shooting phase
    turns = 0
    while not (info.get("agent_won") or info.get("opponent_won")) and turns < 200:
        mask = env.get_valid_shooting_mask()
        valid = mask.nonzero()[0]
        if len(valid) == 0:
            break
        action = int(valid[0])
        obs, reward, term, trunc, info = env.step(action)
        turns += 1
        if term or trunc:
            break

    print(f"\nGame over after {turns} shots. Agent won: {info.get('agent_won')}, Opponent won: {info.get('opponent_won')}")

if __name__ == "__main__":
    main()
