#!/usr/bin/env python3
"""
Evolutionary reward tuning for Battleship shooting mode.

Runs a genetic algorithm over reward weights. Each individual is evaluated by
training for a short burst and measuring fitness (e.g. avg total shots).
Uses sbatch for GPU; run with: sbatch sbatch_evolve_rewards.sh

Usage:
  python src/evolve_rewards.py --generations 10 --population 8 --eval-episodes 3000
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))


def _eval_worker(args: tuple) -> tuple[dict, float]:
    """Worker for multiprocessing: returns (config, fitness). Suppresses per-episode output."""
    config, eval_episodes, agent_type, seed, worker_id = args
    fit = evaluate_fitness(
        config, eval_episodes, agent_type, seed,
        verbose=False,
        worker_id=worker_id,
    )
    return (config, fit)


def evaluate_fitness(
    reward_config: dict,
    eval_episodes: int = 3000,
    agent_type: str = "dqn",
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = True,
    worker_id: int | None = None,
) -> float:
    """
    Train with given reward config for eval_episodes, return fitness.
    Fitness = negative mean total shots (lower shots = higher fitness).
    """
    from battleship import BattleshipEnv
    from battleship.opponents import HuntTargetOpponent

    if agent_type == "dqn":
        from agents import DQNAgent
        from train_dqn import run_episode

        agent = DQNAgent(
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.99995,
            buffer_size=20_000,
            batch_size=64,
            seed=seed,
        )
    else:
        from agents import PPOAgent
        from train_ppo import run_episode

        agent = PPOAgent(
            lr=3e-4,
            gamma=0.99,
            seed=seed,
        )

    env = BattleshipEnv(
        opponent=HuntTargetOpponent(),
        mode="shooting",
        reward_miss=reward_config["miss"],
        reward_per_turn=reward_config["per_turn"],
        reward_efficient_sink=reward_config["efficient_sink"],
        reward_shots_between_sinks=reward_config["shots_between"],
        seed=seed,
    )

    shots_list: list[int] = []
    t0 = time.time()
    wid = f"[W{worker_id}] " if worker_id is not None else "    "
    for ep in range(eval_episodes):
        if agent_type == "dqn":
            _, shots, _, _ = run_episode(
                env, agent, train=True, update_every=4, train_mode="shooting",
            )
        else:
            _, shots, _, _ = run_episode(
                env, agent, train=True, train_mode="shooting",
            )
        shots_list.append(shots)
        if verbose and ((ep + 1) % 200 == 0 or ep == 0):
            recent = np.mean(shots_list[-min(200, len(shots_list)):])
            elapsed = time.time() - t0
            eps_per_sec = (ep + 1) / elapsed if elapsed > 0 else 0
            eps_left = eval_episodes - (ep + 1)
            eta = eps_left / eps_per_sec if eps_per_sec > 0 else 0
            eps_str = f"eps/s={eps_per_sec:.1f}" if agent_type == "dqn" else ""
            print(
                f"{wid}ep {ep+1:4d}/{eval_episodes}  avg_shots={recent:.1f}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  {eps_str}",
                flush=True,
            )

    elapsed = time.time() - t0
    fitness = -float(np.mean(shots_list[-500:]))
    print(f"{wid}-> fitness={fitness:.2f}  (total {elapsed:.0f}s)", flush=True)
    return fitness


def sample_config(bounds: dict) -> dict:
    """Sample a random config within bounds."""
    return {
        k: random.uniform(v[0], v[1]) for k, v in bounds.items()
    }


def mutate(config: dict, bounds: dict, sigma: float = 0.3) -> dict:
    """Gaussian mutation of config, clamped to bounds."""
    out = {}
    for k, (lo, hi) in bounds.items():
        v = config[k] + random.gauss(0, sigma * (hi - lo))
        out[k] = max(lo, min(hi, v))
    return out


def crossover(a: dict, b: dict) -> dict:
    """Uniform crossover."""
    return {k: (a[k] if random.random() < 0.5 else b[k]) for k in a}


def main():
    ap = argparse.ArgumentParser(description="Evolve reward weights for shooting mode")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=3000)
    ap.add_argument("--agent", choices=["dqn", "ppo"], default="dqn")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="evolve_rewards_results.json")
    ap.add_argument("--elite", type=int, default=2, help="Elites to carry over unchanged")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for fitness eval (1=sequential). GPU shared across workers.")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Bounds for shooting-mode rewards (env zeros win/lose/hit/sink/adjacent)
    BOUNDS = {
        "miss": (-3.0, -0.5),
        "per_turn": (-0.5, -0.02),
        "efficient_sink": (5.0, 20.0),
        "shots_between": (0.0, 0.3),
    }

    total_start = time.time()
    print("=" * 60, flush=True)
    print("Evolutionary Reward Tuning — Shooting Mode", flush=True)
    print("=" * 60, flush=True)
    print(f"  Generations: {args.generations}", flush=True)
    print(f"  Population: {args.population}", flush=True)
    print(f"  Eval episodes per individual: {args.eval_episodes}", flush=True)
    print(f"  Agent: {args.agent}", flush=True)
    print(f"  Elite carryover: {args.elite}", flush=True)
    print(f"  Workers (parallel): {args.workers}", flush=True)
    print(f"  Bounds: {BOUNDS}", flush=True)
    print("-" * 60, flush=True)

    def run_evals(configs: list[dict], gen: int, base_seed: int) -> list[tuple[dict, float]]:
        """Evaluate configs, optionally in parallel."""
        tasks = [
            (cfg, args.eval_episodes, args.agent, base_seed + i * 1000, i)
            for i, cfg in enumerate(configs)
        ]
        if args.workers <= 1:
            results = []
            for i, (cfg, _, _, seed, wid) in enumerate(tasks):
                print(f"\n  Individual {i+1}/{len(configs)}: {cfg}", flush=True)
                fit = evaluate_fitness(cfg, args.eval_episodes, args.agent, seed,
                                      verbose=True, worker_id=None)
                results.append((cfg, fit))
            return results
        with mp.Pool(args.workers) as pool:
            for i, (cfg, _, _, _, wid) in enumerate(tasks):
                print(f"  Individual {i+1}/{len(configs)}: {cfg}", flush=True)
            results = pool.map(_eval_worker, tasks)
        return results

    # Initial population
    configs_gen0 = [sample_config(BOUNDS) for _ in range(args.population)]
    print(f"\n[Gen 0] Evaluating {args.population} individuals...", flush=True)
    population = run_evals(configs_gen0, 0, args.seed)

    population.sort(key=lambda x: -x[1])  # higher fitness first
    results = [{"gen": 0, "best": population[0], "all": population}]

    best_cfg, best_fit = population[0]
    print("\n" + "-" * 60, flush=True)
    print(f"Gen 0 complete. Best: fitness={best_fit:.2f}  config={best_cfg}", flush=True)
    print(f"  Population ranked: {[f'{p[1]:.1f}' for p in population]}", flush=True)

    for gen in range(1, args.generations):
        gen_start = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"Generation {gen}/{args.generations - 1}", flush=True)
        print("=" * 60, flush=True)

        # Elites
        new_pop = population[: args.elite]
        print(f"  Carrying over {args.elite} elite(s) unchanged.", flush=True)

        # Fill rest with crossover + mutation
        configs_new = []
        while len(new_pop) + len(configs_new) < args.population:
            a, b = random.sample(population[: args.population // 2], 2)
            child = mutate(crossover(a[0], b[0]), BOUNDS)
            configs_new.append(child)

        print(f"  Evaluating {len(configs_new)} new individuals in parallel...", flush=True)
        evaled = run_evals(configs_new, gen, args.seed + gen * 10000)
        new_pop.extend(evaled)

        population = new_pop
        population.sort(key=lambda x: -x[1])
        results.append({"gen": gen, "best": population[0], "all": population})

        best_cfg, best_fit = population[0]
        gen_elapsed = time.time() - gen_start
        total_elapsed = time.time() - total_start
        print("\n" + "-" * 60, flush=True)
        print(f"Gen {gen} complete in {gen_elapsed:.0f}s. Best: fitness={best_fit:.2f}  config={best_cfg}", flush=True)
        print(f"  Population: {[f'{p[1]:.1f}' for p in population]}", flush=True)
        print(f"  Total elapsed: {total_elapsed/60:.1f} min", flush=True)

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            [
                {
                    "gen": r["gen"],
                    "best_config": r["best"][0],
                    "best_fitness": r["best"][1],
                }
                for r in results
            ],
            f,
            indent=2,
        )
    total_elapsed = time.time() - total_start
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Total runtime: {total_elapsed/60:.1f} min", flush=True)

    best = population[0]
    print("\n" + "=" * 60, flush=True)
    print("FINAL BEST CONFIG", flush=True)
    print("=" * 60, flush=True)
    for k, v in best[0].items():
        print(f"  {k}: {v}", flush=True)
    print(f"  Fitness (neg avg shots): {best[1]:.2f}", flush=True)
    print("\nAdd to sbatch scripts:", flush=True)
    print(f"  REWARD_MISS={best[0]['miss']:.3f}", flush=True)
    print(f"  REWARD_PER_TURN={best[0]['per_turn']:.3f}", flush=True)
    print(f"  REWARD_EFFICIENT_SINK={best[0]['efficient_sink']:.3f}", flush=True)
    print(f"  REWARD_SHOTS_BETWEEN_SINKS={best[0]['shots_between']:.3f}", flush=True)


if __name__ == "__main__":
    main()
