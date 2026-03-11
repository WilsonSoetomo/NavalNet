#!/usr/bin/env python3
"""Human evaluation tool — same setup as DQN shooting training.

- Opponent places ships randomly
- Human shoots until all ships sunk (no opponent turn)
- Record metrics: shots, avg_shots_to_sink, efficiency, reward
- 100 episodes, web-based

Run:  python src/play_human_eval.py --port 6061
Open:  http://localhost:6061
"""

import argparse
import json
import random
import sys
from typing import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

import numpy as np

from battleship.constants import (
    CELL_HIT,
    CELL_MISS,
    CELL_SUNK,
    CELL_UNKNOWN,
    GRID_SIZE,
    NUM_OBS_CHANNELS,
    SHIP_SIZES,
)
from battleship.game_engine import BattleshipGame, Board, CellState
from battleship.opponents import HuntTargetOpponent, RandomOpponent, _random_place_ships

SHIP_NAMES = {5: "Carrier", 4: "Battleship", 3: "Cruiser", 2: "Destroyer"}

# Reward params (match sbatch_train_dqn.sh shooting mode)
REWARD_PER_TURN = -0.15
REWARD_MISS = -1.5
REWARD_EFFICIENT_SINK = 12.0
REWARD_SHOTS_BETWEEN_SINKS = 0.1

TOTAL_EPISODES = 100


def _build_observation(board: Board) -> np.ndarray:
    """Build 7-channel observation for model (mirrors environment/play_web)."""
    raw = board.observation_matrix()
    obs = np.zeros((NUM_OBS_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            v = raw[r][c]
            if v == CELL_UNKNOWN:
                obs[0, r, c] = 1.0
            elif v == CELL_MISS:
                obs[1, r, c] = 1.0
            elif v == CELL_HIT:
                obs[2, r, c] = 1.0
            elif v == CELL_SUNK:
                obs[3, r, c] = 1.0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if obs[2, r, c] == 1.0:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if obs[0, nr, nc] == 1.0:
                            obs[4, nr, nc] = 1.0
    remaining = [s.length for s in board._ships if not s.is_sunk]
    if remaining:
        ok = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        unk = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                v = raw[r][c]
                if v == CELL_UNKNOWN:
                    ok[r, c] = True
                    unk[r, c] = True
                elif v == CELL_HIT:
                    ok[r, c] = True
        prob = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for ship_len in remaining:
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE - ship_len + 1):
                    if ok[r, c : c + ship_len].all():
                        for k in range(ship_len):
                            if unk[r, c + k]:
                                prob[r, c + k] += 1.0
            for c in range(GRID_SIZE):
                for r in range(GRID_SIZE - ship_len + 1):
                    if ok[r : r + ship_len, c].all():
                        for k in range(ship_len):
                            if unk[r + k, c]:
                                prob[r + k, c] += 1.0
        mx = prob.max()
        if mx > 0:
            prob /= mx
        obs[5] = prob
    return obs


def _valid_mask(board: Board) -> np.ndarray:
    mask = np.zeros(GRID_SIZE * GRID_SIZE, dtype=bool)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not board.is_shot(r, c):
                mask[r * GRID_SIZE + c] = True
    return mask


def _load_model_shooter(path: str, model_type: str):
    """Load DQN or PPO model for shooting. Returns callable get_shot(board) -> action."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if model_type == "dqn":
        sd = ckpt.get("policy_net", {})
    else:
        sd = ckpt.get("shooting_actor_critic", {})
    w = sd.get("conv.0.weight")
    in_channels = w.shape[1] if w is not None else NUM_OBS_CHANNELS

    if model_type == "dqn":
        from agents.dqn_agent import DQNAgent, DQNNetwork

        agent = DQNAgent()
        if in_channels != NUM_OBS_CHANNELS:
            agent.policy_net = DQNNetwork(in_channels=in_channels).to(agent.device)
            agent.target_net = DQNNetwork(in_channels=in_channels).to(agent.device)
        agent.load(path, resume=False)
        agent.epsilon = 0.0

        def get_shot(board: Board) -> int:
            obs = _build_observation(board)
            if in_channels < obs.shape[0]:
                obs = obs[:in_channels]
            mask = _valid_mask(board)
            return agent.select_shooting_action(obs, mask, deterministic=True)

    else:
        from agents.ppo_agent import ActorCriticNetwork, PPOAgent

        agent = PPOAgent()
        if in_channels != NUM_OBS_CHANNELS:
            agent.shooting_actor_critic = ActorCriticNetwork(
                in_channels=in_channels
            ).to(agent.device)
        agent.load(path, resume=False)

        def get_shot(board: Board) -> int:
            obs = _build_observation(board)
            if in_channels < obs.shape[0]:
                obs = obs[:in_channels]
            mask = _valid_mask(board)
            action, _ = agent.select_shooting_action(obs, mask, deterministic=True)
            return action

    return get_shot


class HumanEvalState:
    def __init__(self):
        self.game: BattleshipGame | None = None
        self.episode = 0
        self.shots = 0
        self._ship_first_hit_shot: dict[int, int] = {}
        self._sink_stats: list[dict] = []
        self._shots_since_last_sink = 0
        self.episode_reward = 0.0
        self.messages: list[str] = []

        # Aggregate over all completed episodes
        self.completed_episodes: list[dict] = []
        self.total_episodes = TOTAL_EPISODES

    def new_episode(self, seed: int | None = None, total_episodes: int | None = None):
        if total_episodes is not None:
            self.total_episodes = total_episodes
        self.game = BattleshipGame()
        if seed is not None:
            rng = random.Random(seed)
            for length in SHIP_SIZES:
                placed = False
                while not placed:
                    row = rng.randint(0, GRID_SIZE - 1)
                    col = rng.randint(0, GRID_SIZE - 1)
                    orient = rng.choice([0, 1])
                    placed = self.game.opponent_board.place_ship(
                        length, row, col, orient
                    )
        else:
            _random_place_ships(self.game.opponent_board)

        self.game._phase = "shooting"
        self.game._turn = "agent"
        self.shots = 0
        self._ship_first_hit_shot = {}
        self._sink_stats = []
        self._shots_since_last_sink = 0
        self.episode_reward = 0.0
        self.messages = []

    def shoot(self, row: int, col: int) -> dict:
        if self.game is None or self.game.opponent_board.is_shot(row, col):
            return {"error": "Invalid shot."}

        reward = REWARD_PER_TURN
        self.shots += 1

        hit, sunk = self.game.agent_shoot(row, col)

        if hit:
            ship = self.game.opponent_board.get_ship_at(row, col)
            ship_key = id(ship)
            if ship_key not in self._ship_first_hit_shot:
                self._ship_first_hit_shot[ship_key] = self.shots

            if sunk:
                first_hit = self._ship_first_hit_shot[ship_key]
                shots_to_sink = self.shots - first_hit + 1
                efficiency = ship.length / shots_to_sink
                eff_reward = efficiency ** 2
                reward += REWARD_EFFICIENT_SINK * eff_reward
                if REWARD_SHOTS_BETWEEN_SINKS > 0:
                    reward -= REWARD_SHOTS_BETWEEN_SINKS * self._shots_since_last_sink
                self._shots_since_last_sink = 0
                self._sink_stats.append({
                    "ship_length": ship.length,
                    "shots_to_sink": shots_to_sink,
                    "efficiency": efficiency,
                })
                sname = SHIP_NAMES.get(ship.length, "Ship")
                self.messages.append(f"SUNK {sname} ({shots_to_sink} shots)")
            else:
                self._shots_since_last_sink += 1
            self.messages.append(f"HIT at {chr(65+col)}{row}")
        else:
            reward += REWARD_MISS
            self._shots_since_last_sink += 1
            self.messages.append(f"Miss at {chr(65+col)}{row}")

        self.episode_reward += reward

        result = {"result": "sunk" if sunk else "hit" if hit else "miss"}
        if sunk:
            result["sunk_ship"] = SHIP_NAMES.get(ship.length, "Ship")

        if self.game.agent_won():
            avg_sts = (
                sum(s["shots_to_sink"] for s in self._sink_stats) / len(self._sink_stats)
                if self._sink_stats else 0
            )
            avg_eff = (
                sum(s["efficiency"] for s in self._sink_stats) / len(self._sink_stats)
                if self._sink_stats else 0
            )
            ep_data = {
                "episode": self.episode,
                "shots": self.shots,
                "avg_shots_to_sink": avg_sts,
                "avg_efficiency": avg_eff,
                "reward": self.episode_reward,
                "sink_stats": self._sink_stats,
            }
            self.completed_episodes.append(ep_data)
            result["episode_complete"] = True
            result["episode_data"] = ep_data
            result["all_done"] = len(self.completed_episodes) >= self.total_episodes

        result["state"] = self.to_dict()
        return result

    def to_dict(self) -> dict:
        if self.game is None:
            return {
                "phase": "menu",
                "episode": 0,
                "total_episodes": self.total_episodes,
                "attack": None,
                "shots": 0,
                "ships_sunk": 0,
                "episode_reward": 0,
                "messages": [],
                "completed_episodes": [],
                "summary": None,
            }

        attack = []
        obs = self.game.opponent_board.observation_matrix()
        for r in range(GRID_SIZE):
            row = []
            for c in range(GRID_SIZE):
                v = obs[r][c]
                if v == CELL_UNKNOWN:
                    row.append("unknown")
                elif v == CELL_MISS:
                    row.append("miss")
                elif v == CELL_HIT:
                    row.append("hit")
                elif v == CELL_SUNK:
                    row.append("sunk")
                else:
                    row.append("unknown")
            attack.append(row)

        ships_sunk = len(self._sink_stats)
        avg_sts = (
            sum(s["shots_to_sink"] for s in self._sink_stats) / ships_sunk
            if ships_sunk else 0
        )

        summary = None
        if self.completed_episodes:
            n = len(self.completed_episodes)
            total_shots = sum(e["shots"] for e in self.completed_episodes)
            total_reward = sum(e["reward"] for e in self.completed_episodes)
            all_sts = [e["avg_shots_to_sink"] for e in self.completed_episodes]
            all_eff = [e["avg_efficiency"] for e in self.completed_episodes]
            summary = {
                "completed": n,
                "total_episodes": self.total_episodes,
                "avg_shots": total_shots / n,
                "avg_shots_to_sink": sum(all_sts) / n,
                "avg_efficiency": sum(all_eff) / n,
                "avg_reward": total_reward / n,
                "min_shots": min(e["shots"] for e in self.completed_episodes),
                "max_shots": max(e["shots"] for e in self.completed_episodes),
            }

        return {
            "phase": "shooting" if not self.game.agent_won() else "gameover",
            "episode": self.episode,
            "total_episodes": self.total_episodes,
            "attack": attack,
            "shots": self.shots,
            "ships_sunk": ships_sunk,
            "avg_shots_to_sink": avg_sts,
            "episode_reward": self.episode_reward,
            "messages": self.messages[-20:],
            "completed_episodes": self.completed_episodes,
            "summary": summary,
            "won": self.game.agent_won(),
        }


state = HumanEvalState()


MAX_SHOTS_PER_EPISODE = 200  # safeguard against infinite loops


def _get_valid_actions(board: Board) -> list[int]:
    """Return list of unshot cell indices (0-99)."""
    return [
        r * GRID_SIZE + c
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if not board.is_shot(r, c)
    ]


def _run_shooter_eval(
    get_shot_fn,
    shooter_name: str,
    episodes: int,
    on_progress: Callable | None = None,
) -> dict:
    """Run any shooter (bot or model) against randomly placed ships for N episodes.

    on_progress(episode_num, total, shots, avg_sts, avg_eff, reward) called after each episode.
    """
    completed: list[dict] = []

    for ep in range(episodes):
        if on_progress:
            on_progress(ep + 1, episodes, 0, 0.0, 0.0, 0.0)  # signal episode start (shots=0)
        board = Board()
        _random_place_ships(board)

        shots = 0
        ship_first_hit: dict[int, int] = {}
        sink_stats: list[dict] = []
        shots_since_last_sink = 0
        episode_reward = 0.0
        invalid_retries = 0

        while not board.all_ships_sunk():
            if shots >= MAX_SHOTS_PER_EPISODE:
                break
            valid = _get_valid_actions(board)
            if not valid:
                break

            action = get_shot_fn(board)
            row, col = action // GRID_SIZE, action % GRID_SIZE

            if board.is_shot(row, col):
                invalid_retries += 1
                if invalid_retries > 10:
                    action = int(np.random.choice(valid))
                    row, col = action // GRID_SIZE, action % GRID_SIZE
                    invalid_retries = 0
                else:
                    continue
            else:
                invalid_retries = 0

            hit, sunk = board.shoot(row, col)
            shots += 1
            episode_reward += REWARD_PER_TURN

            if hit:
                ship = board.get_ship_at(row, col)
                ship_key = id(ship)
                if ship_key not in ship_first_hit:
                    ship_first_hit[ship_key] = shots

                if sunk:
                    first_hit = ship_first_hit[ship_key]
                    shots_to_sink = shots - first_hit + 1
                    efficiency = ship.length / shots_to_sink
                    episode_reward += REWARD_EFFICIENT_SINK * (efficiency ** 2)
                    if REWARD_SHOTS_BETWEEN_SINKS > 0:
                        episode_reward -= REWARD_SHOTS_BETWEEN_SINKS * shots_since_last_sink
                    shots_since_last_sink = 0
                    sink_stats.append({
                        "ship_length": ship.length,
                        "shots_to_sink": shots_to_sink,
                        "efficiency": efficiency,
                    })
                else:
                    shots_since_last_sink += 1
            else:
                episode_reward += REWARD_MISS
                shots_since_last_sink += 1

        avg_sts = (
            sum(s["shots_to_sink"] for s in sink_stats) / len(sink_stats)
            if sink_stats else 0
        )
        avg_eff = (
            sum(s["efficiency"] for s in sink_stats) / len(sink_stats)
            if sink_stats else 0
        )
        completed.append({
            "episode": ep + 1,
            "shots": shots,
            "avg_shots_to_sink": avg_sts,
            "avg_efficiency": avg_eff,
            "reward": episode_reward,
            "sink_stats": sink_stats,
        })
        if on_progress:
            on_progress(ep + 1, episodes, shots, avg_sts, avg_eff, episode_reward)

    n = len(completed)
    total_shots = sum(e["shots"] for e in completed)
    total_reward = sum(e["reward"] for e in completed)
    all_sts = [e["avg_shots_to_sink"] for e in completed]
    all_eff = [e["avg_efficiency"] for e in completed]

    return {
        "shooter": shooter_name,
        "episodes": n,
        "summary": {
            "completed": n,
            "total_episodes": n,
            "avg_shots": total_shots / n,
            "avg_shots_to_sink": sum(all_sts) / n,
            "avg_efficiency": sum(all_eff) / n,
            "avg_reward": total_reward / n,
            "min_shots": min(e["shots"] for e in completed),
            "max_shots": max(e["shots"] for e in completed),
        },
        "completed_episodes": completed,
    }


def run_bot_eval(
    opponent_type: str,
    episodes: int = 100,
    on_progress: Callable | None = None,
) -> dict:
    """Run bot (Random or HuntTarget) against randomly placed ships for N episodes."""
    opponent = (
        RandomOpponent() if opponent_type == "random" else HuntTargetOpponent()
    )

    def get_shot(board: Board):
        obs = board.observation_matrix()
        return opponent.get_shot(obs)

    return _run_shooter_eval(get_shot, opponent_type, episodes, on_progress)


def run_model_eval(
    model_path: str,
    model_type: str,
    episodes: int = 100,
    on_progress: Callable | None = None,
) -> dict:
    """Run DQN or PPO model against randomly placed ships for N episodes."""
    get_shot = _load_model_shooter(model_path, model_type)
    name = f"{model_type.upper()}: {Path(model_path).stem}"
    return _run_shooter_eval(get_shot, name, episodes, on_progress)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._serve_file(SRC_DIR / "play_human_eval_ui.html", "text/html")
        elif path == "/api/state":
            self._json(state.to_dict())
        elif path == "/api/models":
            models_dir = SRC_DIR.parent / "models"
            models = []
            if models_dir.exists():
                for f in sorted(models_dir.rglob("*.pt")):
                    rel = str(f.relative_to(SRC_DIR.parent))
                    mtype = "ppo" if "ppo" in f.name.lower() else "dqn"
                    models.append({"path": rel, "name": f.stem, "type": mtype})
            self._json({"models": models})
        elif path == "/api/eval-stream":
            self._handle_eval_stream()
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()

        if path == "/api/start":
            state.episode = 1
            state.completed_episodes = []
            state.new_episode(
                seed=body.get("seed"),
                total_episodes=body.get("episodes") or TOTAL_EPISODES,
            )
            self._json({"state": state.to_dict()})

        elif path == "/api/new-episode":
            state.episode += 1
            state.new_episode(seed=body.get("seed"))
            self._json({"state": state.to_dict()})

        elif path == "/api/shoot":
            r, c = body.get("row", 0), body.get("col", 0)
            res = state.shoot(r, c)
            self._json(res)

        elif path == "/api/bot-eval":
            opp = body.get("opponent", "random")
            n_ep = int(body.get("episodes", 100))
            if opp not in ("random", "hunt_target"):
                self._json({"error": "Invalid opponent"})
            else:
                result = run_bot_eval(opp, n_ep)
                self._json(result)

        elif path == "/api/model-eval":
            path_arg = body.get("model_path")
            model_type = body.get("model_type", "dqn")
            n_ep = int(body.get("episodes", 100))
            if not path_arg:
                self._json({"error": "model_path required"})
            elif model_type not in ("dqn", "ppo"):
                self._json({"error": "model_type must be dqn or ppo"})
            else:
                full_path = SRC_DIR.parent / path_arg
                if not full_path.exists():
                    self._json({"error": f"Model not found: {path_arg}"})
                else:
                    result = run_model_eval(str(full_path), model_type, n_ep)
                    self._json(result)

        else:
            self.send_error(404)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _json(self, data: dict):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_eval_stream(self):
        """Stream eval progress via Server-Sent Events."""
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        shooter = (params.get("shooter") or ["random"])[0]
        episodes = int((params.get("episodes") or ["10"])[0])
        model_path = (params.get("model_path") or [""])[0]
        model_type = (params.get("model_type") or ["dqn"])[0]

        def send_event(data: dict):
            line = "data: " + json.dumps(data) + "\n\n"
            self.wfile.write(line.encode())
            self.wfile.flush()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            if shooter == "model" and model_path:
                full_path = SRC_DIR.parent / model_path
                if not full_path.exists():
                    send_event({"error": f"Model not found: {model_path}"})
                    return
                send_event({"msg": f"Loading model {model_path}...", "type": "log"})
                result = run_model_eval(
                    str(full_path),
                    model_type,
                    episodes,
                    on_progress=lambda ep, tot, shots, sts, eff, rew: send_event({
                        "msg": f"Episode {ep}/{tot} complete: {shots} shots, "
                               f"{sts:.1f} avg shots/sink, {eff*100:.1f}% eff, reward {rew:.1f}"
                        if shots > 0 else f"Running episode {ep}/{tot}...",
                        "type": "progress",
                        "episode": ep,
                        "total": tot,
                        "shots": shots,
                        "avg_shots_to_sink": sts,
                        "avg_efficiency": eff,
                        "reward": rew,
                    }),
                )
            elif shooter in ("random", "hunt_target"):
                result = run_bot_eval(
                    shooter,
                    episodes,
                    on_progress=lambda ep, tot, shots, sts, eff, rew: send_event({
                        "msg": f"Episode {ep}/{tot} complete: {shots} shots, "
                               f"{sts:.1f} avg shots/sink, {eff*100:.1f}% eff, reward {rew:.1f}"
                        if shots > 0 else f"Running episode {ep}/{tot}...",
                        "type": "progress",
                        "episode": ep,
                        "total": tot,
                        "shots": shots,
                        "avg_shots_to_sink": sts,
                        "avg_efficiency": eff,
                        "reward": rew,
                    }),
                )
            else:
                send_event({"error": "Invalid shooter or missing model_path"})
                return
            send_event({"type": "done", "result": result})
        except Exception as e:
            send_event({"type": "error", "error": str(e)})


def main():
    ap = argparse.ArgumentParser(description="NavalNet Human Eval — 100 episodes")
    ap.add_argument("--port", type=int, default=6061)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    srv = HTTPServer((args.host, args.port), Handler)
    print(f"Human Eval running at http://localhost:{args.port}")
    print(f"Complete {TOTAL_EPISODES} episodes. Same setup as DQN shooting training.")
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        srv.shutdown()


if __name__ == "__main__":
    main()
