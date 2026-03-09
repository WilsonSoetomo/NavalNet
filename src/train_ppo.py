"""
Train PPO agent for Battleship.
Usage: python train_ppo.py [--episodes N] [--save-path PATH] [--seed N]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))

from battleship import BattleshipEnv
from battleship.opponents import CurriculumOpponent, HuntTargetOpponent, RandomOpponent
from battleship.board_renderer import render_game_boards, figure_to_numpy
from agents import PPOAgent

OPPONENTS = {
    "random": RandomOpponent,
    "hunt_target": HuntTargetOpponent,
    "curriculum": None,
}


def run_episode(
    env: BattleshipEnv,
    agent: PPOAgent,
    train: bool = True,
    verbose: bool = False,
    train_mode: str = "full",
) -> tuple[float, int, bool, dict]:
    """
    Run one episode. Returns (total_reward, num_shots, agent_won, stats).
    stats contains loss values for TensorBoard logging.

    train_mode: "full"      – train both heads in a normal game
                "shooting"  – shooting head only (no placement, no opponent turn)
                "placement" – placement head only (reward = attacker shot count)
    """
    obs, info = env.reset()
    total_reward = 0.0
    num_shots = 0

    # ── Placement phase ──────────────────────────────────────────────
    placement_steps = 0
    max_placement_steps = 100
    skip_placement = (train_mode == "shooting")

    while info.get("phase") == "placement" and not skip_placement:
        if placement_steps >= max_placement_steps:
            print(f"WARNING: Placement exceeded {max_placement_steps} steps", flush=True)
            break
        placement_obs = env.get_placement_observation()
        ship_index = info.get("ship_index", 0)
        mask = env.get_valid_placement_mask()
        if mask.sum() == 0:
            print(f"WARNING: No valid placement actions at ship_index {ship_index}", flush=True)
            break

        action, log_prob = agent.select_placement_action(
            placement_obs, ship_index, mask, deterministic=not train
        )

        agent.placement_actor_critic.eval()
        with torch.no_grad():
            x = torch.tensor(placement_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            si = torch.tensor([ship_index], dtype=torch.long, device=agent.device)
            _, value = agent.placement_actor_critic(x, si)
            value = value.item()
        agent.placement_actor_critic.train()

        if train:
            agent.store_placement_transition(
                placement_obs, ship_index, action, 0.0, log_prob, value
            )

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        placement_steps += 1

        if term or trunc:
            break

    if verbose:
        print(f"  Placement done in {placement_steps} steps", flush=True)

    # ── Placement-only: episode ends after placement ─────────────────
    if train_mode == "placement":
        placement_stats: dict = {}
        if train:
            placement_stats = agent.update_placement(total_reward)
            agent.shooting_trajectory.clear()
        attacker_shots = info.get("attacker_shots", 0)
        stats = {
            **placement_stats,
            "avg_shots_to_sink": 0.0,
            "avg_sink_efficiency": 0.0,
            "ships_sunk": 0,
            "attacker_shots": attacker_shots,
        }
        return total_reward, 0, False, stats

    # ── Shooting phase ───────────────────────────────────────────────
    max_shooting_steps = 200
    shooting_steps = 0
    while not (info.get("agent_won") or info.get("opponent_won")):
        if shooting_steps >= max_shooting_steps:
            print(f"WARNING: Shooting exceeded {max_shooting_steps} steps", flush=True)
            break
        mask = env.get_valid_shooting_mask()
        if mask.sum() == 0:
            break

        action, log_prob = agent.select_shooting_action(obs, mask, deterministic=not train)

        agent.shooting_actor_critic.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=agent.device)
            _, value = agent.shooting_actor_critic(x)
            value = value.item()
        agent.shooting_actor_critic.train()

        prev_obs = obs.copy()

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        num_shots += 1
        shooting_steps += 1

        if verbose and shooting_steps % 20 == 0:
            print(f"    Shot {shooting_steps}: reward={reward:.2f}, total={total_reward:.2f}", flush=True)

        if train:
            agent.store_shooting_transition(
                prev_obs, action, reward, obs, term or trunc, log_prob, value
            )

    if verbose:
        print(f"  Shooting done in {shooting_steps} shots, won={info.get('agent_won', False)}", flush=True)

    # ── Update policies (on-policy: end of episode) ──────────────────
    placement_stats2: dict = {}
    shooting_stats: dict = {}
    if train:
        if train_mode == "full":
            placement_stats2 = agent.update_placement(total_reward)
        else:
            agent.placement_trajectory.clear()
        shooting_stats = agent.update_shooting()

    sink_stats = env.get_sink_stats()
    avg_sts = float(np.mean([s["shots_to_sink"] for s in sink_stats])) if sink_stats else 0.0
    avg_eff = float(np.mean([s["efficiency"] for s in sink_stats])) if sink_stats else 0.0
    stats = {
        **placement_stats2,
        **shooting_stats,
        "avg_shots_to_sink": avg_sts,
        "avg_sink_efficiency": avg_eff,
        "ships_sunk": len(sink_stats),
    }
    return total_reward, num_shots, bool(info.get("agent_won", False)), stats


def run_showcase_game(
    env: BattleshipEnv,
    agent: PPOAgent,
    writer: SummaryWriter,
    episode: int,
) -> None:
    """
    Play one full game with step-by-step TensorBoard image logging.
    Each shot produces a board snapshot so the game can be replayed
    in TensorBoard's image slider.
    """
    obs, info = env.reset()

    # Placement phase (deterministic)
    placement_steps = 0
    while info.get("phase") == "placement":
        if placement_steps >= 100:
            break
        placement_obs = env.get_placement_observation()
        ship_index = info.get("ship_index", 0)
        mask = env.get_valid_placement_mask()
        if mask.sum() == 0:
            break
        action, _ = agent.select_placement_action(
            placement_obs, ship_index, mask, deterministic=True
        )
        obs, _, _, _, info = env.step(action)
        placement_steps += 1

    # Log initial board state (before any shots)
    board_state = env.get_full_board_state()
    fig = render_game_boards(board_state, episode=episode, step=0, result="START")
    img = figure_to_numpy(fig)
    writer.add_image("replay/game_board", img, 0, dataformats="HWC")
    plt_close(fig)

    # Shooting phase: log after every shot
    shot_num = 0
    max_shots = 200
    while not (info.get("agent_won") or info.get("opponent_won")):
        if shot_num >= max_shots:
            break
        mask = env.get_valid_shooting_mask()
        if mask.sum() == 0:
            break

        action, _ = agent.select_shooting_action(obs, mask, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        shot_num += 1

        board_state = env.get_full_board_state()
        won = board_state.get("agent_won", False)
        lost = board_state.get("opponent_won", False)
        if won:
            result = "WIN"
        elif lost:
            result = "LOSS"
        else:
            result = "IN PROGRESS"

        fig = render_game_boards(board_state, episode=episode, step=shot_num, result=result)
        img = figure_to_numpy(fig)
        writer.add_image("replay/game_board", img, shot_num, dataformats="HWC")
        plt_close(fig)

    writer.flush()


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Battleship")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--save-path", type=str, default="models/ppo.pt")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-shot progress within episodes")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to pre-trained model to resume from")
    parser.add_argument("--opponent", type=str, default="random",
                        choices=list(OPPONENTS.keys()),
                        help="Opponent type (default: random)")
    parser.add_argument("--curriculum-start", type=float, default=0.0,
                        help="Starting hard-opponent ratio for curriculum (default 0.0)")
    parser.add_argument("--curriculum-end", type=float, default=0.8,
                        help="Ending hard-opponent ratio for curriculum (default 0.8)")
    parser.add_argument("--curriculum-ramp", type=int, default=5000,
                        help="Episodes over which to ramp curriculum difficulty (linear mode)")
    parser.add_argument("--curriculum-gate-wr", type=float, default=0.0,
                        help="Win-rate threshold for gated curriculum (0 = linear mode)")
    parser.add_argument("--reward-win", type=float, default=100.0)
    parser.add_argument("--reward-lose", type=float, default=-100.0)
    parser.add_argument("--reward-hit", type=float, default=1.0)
    parser.add_argument("--reward-sink", type=float, default=5.0)
    parser.add_argument("--reward-miss", type=float, default=-0.5)
    parser.add_argument("--reward-efficient-sink", type=float, default=2.0)
    parser.add_argument("--reward-adjacent-hit", type=float, default=0.3)
    parser.add_argument("--reward-per-turn", type=float, default=-0.05)
    parser.add_argument("--reward-shots-between-sinks", type=float, default=0.0)
    parser.add_argument("--train-mode", type=str, default="full",
                        choices=["full", "shooting", "placement"],
                        help="Training mode: full (both heads), shooting (only shooting, "
                             "no opponent turn), placement (only placement, rated by attacker)")
    parser.add_argument("--entropy-coef", type=float, default=0.05,
                        help="Entropy bonus coefficient (default: 0.05 for shooting mode)")
    parser.add_argument("--update-epochs", type=int, default=8,
                        help="PPO update epochs per batch (default: 8)")
    parser.add_argument("--logdir", type=str, default="runs/ppo",
                        help="TensorBoard log directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Opponent setup ───────────────────────────────────────────────
    if args.opponent == "curriculum":
        opponent = CurriculumOpponent(
            easy=RandomOpponent(),
            hard=HuntTargetOpponent(),
            start_hard_ratio=args.curriculum_start,
            end_hard_ratio=args.curriculum_end,
            ramp_episodes=args.curriculum_ramp,
            gate_wr=args.curriculum_gate_wr,
        )
    else:
        opponent = OPPONENTS[args.opponent]()

    env = BattleshipEnv(
        opponent=opponent,
        mode=args.train_mode,
        reward_hit=args.reward_hit,
        reward_sink=args.reward_sink,
        reward_miss=args.reward_miss,
        reward_win=args.reward_win,
        reward_lose=args.reward_lose,
        reward_per_turn=args.reward_per_turn,
        reward_shots_between_sinks=args.reward_shots_between_sinks,
        reward_efficient_sink=args.reward_efficient_sink,
        reward_adjacent_hit=args.reward_adjacent_hit,
        seed=args.seed,
    )

    agent = PPOAgent(
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=args.entropy_coef,
        max_grad_norm=0.5,
        update_epochs=args.update_epochs,
        seed=args.seed,
    )

    if args.load_model:
        agent.load(args.load_model, resume=False)
        print(f"Loaded weights from {args.load_model} (fresh optimizer)", flush=True)

    # ── TensorBoard ──────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=args.logdir)
    print(f"TensorBoard logs: {args.logdir}", flush=True)
    print(f"  Launch: tensorboard --bind_all --logdir {args.logdir}", flush=True)

    wins = 0
    total_shots = 0
    rewards_history: list[float] = []
    win_history: list[int] = []

    print(f"Training PPO for {args.episodes} episodes...", flush=True)
    print(f"  Device: {device}", flush=True)
    print(f"  Opponent: {args.opponent}", flush=True)
    if args.opponent == "curriculum":
        if args.curriculum_gate_wr > 0:
            print(f"  Curriculum: gated (wr>={args.curriculum_gate_wr:.0%} to ramp up, "
                  f"{args.curriculum_start:.0%} -> {args.curriculum_end:.0%})", flush=True)
        else:
            print(f"  Curriculum: linear {args.curriculum_start:.0%} -> {args.curriculum_end:.0%} "
                  f"over {args.curriculum_ramp} eps", flush=True)
    print(f"  Train mode: {args.train_mode}", flush=True)
    print(f"  Rewards: win={args.reward_win} lose={args.reward_lose} hit={args.reward_hit} "
          f"miss={args.reward_miss} sink={args.reward_sink} adj_hit={args.reward_adjacent_hit}", flush=True)
    print(f"  Save path: {args.save_path}", flush=True)
    print("-" * 50, flush=True)

    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        if ep == 1:
            print("Starting first episode...", flush=True)

        if hasattr(opponent, "set_episode"):
            opponent.set_episode(ep)

        episode_start = time.time()
        ep_verbose = args.verbose or ep == 1
        reward, shots, won, stats = run_episode(
            env, agent, train=True, verbose=ep_verbose,
            train_mode=args.train_mode,
        )
        episode_time = time.time() - episode_start
        wins += int(won)
        total_shots += shots
        rewards_history.append(reward)
        win_history.append(int(won))

        if hasattr(opponent, "report_result"):
            opponent.report_result(won)

        # ── TensorBoard scalars (every episode) ─────────────────────
        writer.add_scalar("episode/reward", reward, ep)
        writer.add_scalar("episode/shots", shots, ep)
        writer.add_scalar("episode/won", int(won), ep)
        writer.add_scalar("episode/time_s", episode_time, ep)
        if "policy_loss" in stats:
            writer.add_scalar("loss/policy", stats["policy_loss"], ep)
        if "value_loss" in stats:
            writer.add_scalar("loss/value", stats["value_loss"], ep)
        if "entropy" in stats:
            writer.add_scalar("loss/entropy", stats["entropy"], ep)
        if stats["avg_shots_to_sink"] > 0:
            writer.add_scalar("episode/avg_shots_to_sink", stats["avg_shots_to_sink"], ep)
            writer.add_scalar("episode/sink_efficiency", stats["avg_sink_efficiency"], ep)
        writer.add_scalar("episode/ships_sunk", stats["ships_sunk"], ep)
        if "attacker_shots" in stats and stats["attacker_shots"] > 0:
            writer.add_scalar("episode/attacker_shots", stats["attacker_shots"], ep)
        if hasattr(opponent, "hard_ratio"):
            writer.add_scalar("curriculum/hard_ratio", opponent.hard_ratio, ep)

        # Rolling averages
        recent_n = min(100, len(rewards_history))
        writer.add_scalar("rolling/reward_avg_100",
                          float(np.mean(rewards_history[-recent_n:])), ep)
        writer.add_scalar("rolling/win_rate_100",
                          float(np.mean(win_history[-recent_n:])), ep)

        if ep == 1:
            print(f"Episode 1 complete: {shots} shots, {episode_time:.2f}s, "
                  f"{'WIN' if won else 'LOSS'}", flush=True)

        # Console progress every 10 episodes
        if ep == 1 or ep % 10 == 0:
            elapsed = time.time() - start_time
            recent = rewards_history[-10:] if len(rewards_history) >= 10 else rewards_history
            eps_per_sec = ep / elapsed if elapsed > 0 else 0
            cur_tag = ""
            if hasattr(opponent, "hard_ratio"):
                cur_tag = f" | hard {opponent.hard_ratio:.0%}"
            print(
                f"Ep {ep:5d} | Shots {shots:3d} | Reward {reward:6.1f} | "
                f"Win {'Y' if won else 'N'} | "
                f"R_avg {np.mean(recent):.1f} | {eps_per_sec:.1f} ep/s{cur_tag}",
                flush=True,
            )

        # ── Evaluation + board snapshot ──────────────────────────────
        if ep % args.eval_every == 0:
            eval_wins = 0
            eval_shots_list: list[int] = []

            last_board_state = None
            for g in range(args.eval_games):
                _, shots_e, won_e, _ = run_episode(env, agent, train=False,
                                                    train_mode=args.train_mode)
                eval_wins += int(won_e)
                eval_shots_list.append(shots_e)
                if g == args.eval_games - 1:
                    last_board_state = env.get_full_board_state()

            eval_wr = 100 * eval_wins / args.eval_games
            eval_avg_shots = float(np.mean(eval_shots_list))
            writer.add_scalar("eval/win_rate", eval_wr, ep)
            writer.add_scalar("eval/avg_shots", eval_avg_shots, ep)

            if last_board_state is not None:
                result_str = "WIN" if last_board_state["agent_won"] else "LOSS"
                fig = render_game_boards(
                    last_board_state, episode=ep, result=f"Eval {result_str}"
                )
                img = figure_to_numpy(fig)
                writer.add_image("eval/game_board", img, ep, dataformats="HWC")
                plt_close(fig)

            run_showcase_game(env, agent, writer, episode=ep)

            recent = rewards_history[-100:]
            print(
                f"  EVAL  | Win% {eval_wr:5.1f} | AvgShots {eval_avg_shots:5.1f} | "
                f"R_avg {np.mean(recent):.1f}",
                flush=True,
            )

        if ep % args.save_every == 0:
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            agent.save(args.save_path)
            print(f"  Saved to {args.save_path}", flush=True)

    agent.save(args.save_path)
    writer.close()
    total_time = time.time() - start_time
    print("-" * 50, flush=True)
    print(f"Training complete in {total_time:.1f}s. Final model saved to {args.save_path}", flush=True)


def plt_close(fig):
    """Safely close a matplotlib figure."""
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
