"""
Demo script to visualize a Battleship game.
Run this to see how the visualization works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from battleship import BattleshipEnv
from battleship.opponents import RandomOpponent
from battleship.visualization import print_board, print_action, print_game_summary
from agents import DQNAgent


def main():
    print("="*60)
    print("Battleship Visualization Demo")
    print("="*60)
    
    env = BattleshipEnv(opponent=RandomOpponent(), seed=42)
    agent = DQNAgent(epsilon=1.0)  # Fully random for demo
    
    obs, info = env.reset()
    
    print("\n=== PLACEMENT PHASE ===")
    placement_step = 0
    while info.get("phase") == "placement":
        placement_obs = env.get_placement_observation()
        ship_index = info.get("ship_index", 0)
        mask = env.get_valid_placement_mask()
        
        print(f"\nPlacing ship {ship_index + 1}/5")
        print_board(placement_obs.tolist(), "Agent Board (Placement)")
        print(f"Valid actions: {mask.sum()}/{len(mask)}")
        
        action = agent.select_placement_action(
            placement_obs, ship_index, mask, deterministic=False
        )
        print_action(action, "placement", ship_index)
        
        obs, reward, term, trunc, info = env.step(action)
        placement_step += 1
        
        if reward < 0:
            print(f"⚠ Invalid placement! Reward: {reward}")
    
    print("\n=== SHOOTING PHASE ===")
    shot_num = 0
    while not (info.get("agent_won") or info.get("opponent_won")):
        shot_num += 1
        mask = env.get_valid_shooting_mask()
        
        if shot_num <= 5 or shot_num % 10 == 0:
            print(f"\n--- Shot {shot_num} ---")
            print_board(obs.tolist(), "Attack Board")
            print_game_summary(obs.tolist(), shot_num, "shooting")
        
        action = agent.select_shooting_action(obs, mask, deterministic=False)
        
        if shot_num <= 5:
            print_action(action, "shooting")
        
        obs, reward, term, trunc, info = env.step(action)
        
        if shot_num <= 5:
            hit = reward > 0.5  # Hit gives reward > 0.5
            print(f"  Result: {'HIT!' if hit else 'MISS'} | Reward: {reward:.2f}")
    
    print("\n=== GAME OVER ===")
    print_board(obs.tolist(), "Final Attack Board")
    print(f"\nAgent {'WON' if info.get('agent_won') else 'LOST'} in {shot_num} shots!")


if __name__ == "__main__":
    main()
