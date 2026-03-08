#!/usr/bin/env python3
"""Web-based interactive Battleship game server.

Run:  python src/play_web.py --port 6060
Open:  http://localhost:6060   (VSCode auto-forwards the port over SSH)
"""

import argparse
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from battleship.constants import (
    CELL_HIT,
    CELL_MISS,
    CELL_SUNK,
    CELL_UNKNOWN,
    GRID_SIZE,
    HORIZONTAL,
    NUM_CELLS,
    NUM_OBS_CHANNELS,
    SHIP_SIZES,
    VERTICAL,
)
from battleship.game_engine import BattleshipGame, Board, CellState
from battleship.opponents import HuntTargetOpponent, RandomOpponent, _random_place_ships

SHIP_NAMES = {5: "Carrier", 4: "Battleship", 3: "Cruiser", 2: "Destroyer"}


# ── Observation builder (mirrors environment.py, needed for model opponent) ──

def _build_observation(board: Board) -> np.ndarray:
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
    mask = np.zeros(NUM_CELLS, dtype=bool)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not board.is_shot(r, c):
                mask[r * GRID_SIZE + c] = True
    return mask


class ModelOpponent:
    def __init__(self, path: str, model_type: str = "dqn"):
        import torch

        self.model_type = model_type

        # Peek at the checkpoint to find the channel count the model was
        # trained with (handles old 5-ch models loaded into a 6-ch codebase).
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if model_type == "dqn":
            sd = ckpt.get("policy_net", {})
        else:
            sd = ckpt.get("shooting_actor_critic", {})
        w = sd.get("conv.0.weight")
        self.in_channels = w.shape[1] if w is not None else NUM_OBS_CHANNELS

        if model_type == "dqn":
            from agents.dqn_agent import DQNAgent, DQNNetwork

            self.agent = DQNAgent()
            if self.in_channels != NUM_OBS_CHANNELS:
                self.agent.policy_net = DQNNetwork(in_channels=self.in_channels).to(self.agent.device)
                self.agent.target_net = DQNNetwork(in_channels=self.in_channels).to(self.agent.device)
            self.agent.load(path, resume=False)
            self.agent.epsilon = 0.0
        else:
            from agents.ppo_agent import ActorCriticNetwork, PPOAgent

            self.agent = PPOAgent()
            if self.in_channels != NUM_OBS_CHANNELS:
                self.agent.shooting_actor_critic = ActorCriticNetwork(
                    in_channels=self.in_channels
                ).to(self.agent.device)
            self.agent.load(path, resume=False)

    def place_ships(self, board: Board) -> None:
        _random_place_ships(board)

    def get_shot(self, board: Board) -> int:
        obs = _build_observation(board)
        # Truncate to the channel count the model expects
        if self.in_channels < obs.shape[0]:
            obs = obs[: self.in_channels]
        mask = _valid_mask(board)
        if self.model_type == "dqn":
            return self.agent.select_shooting_action(obs, mask, deterministic=True)
        action, _ = self.agent.select_shooting_action(obs, mask, deterministic=True)
        return action


# ── Game state ──────────────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.game: BattleshipGame | None = None
        self.opponent = None
        self.opponent_name = ""
        self.is_model = False
        self.reveal = False
        self.phase = "menu"
        self.ship_idx = 0
        self.orientation = HORIZONTAL
        self.player_shots = 0
        self.opp_shots = 0
        self.messages: list[str] = []
        self.winner: str | None = None
        self.locked = False

    def new_game(self, opponent_type: str, model_type: str = "dqn"):
        self.game = BattleshipGame()
        self.phase = "placement"
        self.ship_idx = 0
        self.orientation = HORIZONTAL
        self.player_shots = 0
        self.opp_shots = 0
        self.messages = []
        self.winner = None
        self.locked = False

        if opponent_type == "random":
            self.opponent = RandomOpponent()
            self.opponent_name = "Random Bot"
            self.is_model = False
        elif opponent_type == "hunt_target":
            self.opponent = HuntTargetOpponent()
            self.opponent_name = "Hunt/Target Bot"
            self.is_model = False
        else:
            self.opponent = ModelOpponent(opponent_type, model_type)
            self.opponent_name = f"{model_type.upper()} Model"
            self.is_model = True

        self.opponent.place_ships(self.game.opponent_board)
        length = SHIP_SIZES[0]
        name = SHIP_NAMES.get(length, f"Ship({length})")
        self.messages.append(f"Game started vs {self.opponent_name}!")
        self.messages.append(f"Place your {name} (length {length}).")

    def place_ship(self, row: int, col: int):
        if self.phase != "placement" or self.ship_idx >= len(SHIP_SIZES):
            return False, "Not in placement phase."
        length = SHIP_SIZES[self.ship_idx]
        name = SHIP_NAMES.get(length, f"Ship({length})")
        ok = self.game.place_agent_ship(length, row, col, self.orientation)
        if not ok:
            return False, f"{name} doesn't fit there."
        ori = "H" if self.orientation == HORIZONTAL else "V"
        self.messages.append(f"Placed {name} at {chr(65+col)}{row} ({ori}).")
        self.ship_idx += 1
        if self.ship_idx >= len(SHIP_SIZES):
            self._begin_shooting()
        else:
            nl = SHIP_SIZES[self.ship_idx]
            nn = SHIP_NAMES.get(nl, f"Ship({nl})")
            self.messages.append(f"Place your {nn} (length {nl}).")
        return True, None

    def random_place(self):
        if self.phase != "placement":
            return
        for i in range(self.ship_idx, len(SHIP_SIZES)):
            ln = SHIP_SIZES[i]
            while True:
                r = random.randint(0, GRID_SIZE - 1)
                c = random.randint(0, GRID_SIZE - 1)
                o = random.choice([HORIZONTAL, VERTICAL])
                if self.game.place_agent_ship(ln, r, c, o):
                    break
        self.ship_idx = len(SHIP_SIZES)
        self.messages.append("Ships placed randomly.")
        self._begin_shooting()

    def _begin_shooting(self):
        self.phase = "shooting"
        self.game._phase = "shooting"
        self.game._turn = "agent"
        self.messages.append("All ships placed — open fire!")

    def shoot(self, row: int, col: int):
        if self.phase != "shooting" or self.locked:
            return {"error": "Not your turn."}
        if self.game.opponent_board.is_shot(row, col):
            return {"error": "Already shot there."}

        hit, sunk = self.game.agent_shoot(row, col)
        self.player_shots += 1
        coord = f"{chr(65+col)}{row}"
        result: dict = {}

        if sunk:
            ship = self.game.opponent_board.get_ship_at(row, col)
            sname = SHIP_NAMES.get(ship.length, "Ship") if ship else "Ship"
            result["result"] = "sunk"
            result["sunk_ship"] = sname
            self.messages.append(f"HIT & SUNK {sname} at {coord}!")
        elif hit:
            result["result"] = "hit"
            self.messages.append(f"HIT at {coord}!")
        else:
            result["result"] = "miss"
            self.messages.append(f"Miss at {coord}.")

        if self.game.game_over():
            self.phase = "gameover"
            self.winner = "player"
            self.messages.append("YOU WIN!")
            result["opponent_turn"] = []
            return result

        opp_actions: list[dict] = []
        if self.game._turn == "opponent":
            while self.game._turn == "opponent" and not self.game.game_over():
                if self.is_model:
                    action = self.opponent.get_shot(self.game.agent_board)
                else:
                    obs_mat = self.game.agent_board.observation_matrix()
                    action = self.opponent.get_shot(obs_mat)
                orow, ocol = action // GRID_SIZE, action % GRID_SIZE
                ohit, osunk = self.game.opponent_shoot(orow, ocol)
                self.opp_shots += 1
                oc = f"{chr(65+ocol)}{orow}"
                oa: dict = {"row": orow, "col": ocol}
                if osunk:
                    ship = self.game.agent_board.get_ship_at(orow, ocol)
                    sn = SHIP_NAMES.get(ship.length, "Ship") if ship else "Ship"
                    oa["result"] = "sunk"
                    oa["sunk_ship"] = sn
                    self.messages.append(f"Enemy SUNK your {sn} at {oc}!")
                elif ohit:
                    oa["result"] = "hit"
                    self.messages.append(f"Enemy HIT at {oc}!")
                else:
                    oa["result"] = "miss"
                    self.messages.append(f"Enemy miss at {oc}.")
                opp_actions.append(oa)
            if self.game.game_over():
                self.phase = "gameover"
                self.winner = "opponent"
                self.messages.append("YOU LOSE!")
            else:
                self.game._turn = "agent"

        result["opponent_turn"] = opp_actions
        return result

    def to_dict(self) -> dict:
        if self.game is None:
            return {"phase": "menu", "messages": [], "reveal": self.reveal}

        defense = []
        for r in range(GRID_SIZE):
            row = []
            for c in range(GRID_SIZE):
                s = self.game.agent_board.get_cell_state(r, c)
                row.append(s.name.lower())
            defense.append(row)

        attack = []
        opp_ships = []
        obs = self.game.opponent_board.observation_matrix()
        for r in range(GRID_SIZE):
            arow, srow = [], []
            for c in range(GRID_SIZE):
                v = obs[r][c]
                if v == CELL_UNKNOWN:
                    arow.append("unknown")
                elif v == CELL_MISS:
                    arow.append("miss")
                elif v == CELL_HIT:
                    arow.append("hit")
                elif v == CELL_SUNK:
                    arow.append("sunk")
                srow.append(
                    self.game.opponent_board.get_cell_state(r, c) == CellState.SHIP
                )
            attack.append(arow)
            opp_ships.append(srow)

        current_ship = None
        if self.phase == "placement" and self.ship_idx < len(SHIP_SIZES):
            ln = SHIP_SIZES[self.ship_idx]
            current_ship = {
                "name": SHIP_NAMES.get(ln, f"Ship({ln})"),
                "length": ln,
                "index": self.ship_idx,
            }

        opp_alive = sum(1 for s in self.game.opponent_board._ships if not s.is_sunk)
        my_alive = sum(1 for s in self.game.agent_board._ships if not s.is_sunk)

        return {
            "phase": self.phase,
            "defense": defense,
            "attack": attack,
            "opponent_ships": opp_ships if self.reveal else None,
            "current_ship": current_ship,
            "orientation": self.orientation,
            "player_shots": self.player_shots,
            "opp_shots": self.opp_shots,
            "my_alive": my_alive,
            "my_total": len(self.game.agent_board._ships),
            "opp_alive": opp_alive,
            "opp_total": len(self.game.opponent_board._ships),
            "reveal": self.reveal,
            "winner": self.winner,
            "messages": self.messages[-30:],
            "opponent_name": self.opponent_name,
        }


# ── HTTP handler ────────────────────────────────────────────────────

state = GameState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    # ── GET ──────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._serve_file(SRC_DIR / "play_web_ui.html", "text/html")
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
        else:
            self.send_error(404)

    # ── POST ─────────────────────────────────────────────────────────

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._body()

        if path == "/api/new":
            opp = body.get("opponent", "random")
            mt = body.get("model_type", "dqn")
            state.new_game(opp, mt)
            self._json(state.to_dict())

        elif path == "/api/place":
            ok, err = state.place_ship(body.get("row", 0), body.get("col", 0))
            self._json({"success": ok, "error": err, "state": state.to_dict()})

        elif path == "/api/place-random":
            state.random_place()
            self._json(state.to_dict())

        elif path == "/api/shoot":
            res = state.shoot(body.get("row", 0), body.get("col", 0))
            res["state"] = state.to_dict()
            self._json(res)

        elif path == "/api/toggle-reveal":
            state.reveal = not state.reveal
            self._json(state.to_dict())

        elif path == "/api/rotate":
            state.orientation = (
                VERTICAL if state.orientation == HORIZONTAL else HORIZONTAL
            )
            self._json({"orientation": state.orientation, "state": state.to_dict()})

        else:
            self.send_error(404)

    # ── Helpers ──────────────────────────────────────────────────────

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


def main():
    ap = argparse.ArgumentParser(description="NavalNet Battleship — web UI")
    ap.add_argument("--port", type=int, default=6060)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    srv = HTTPServer((args.host, args.port), Handler)
    print(f"NavalNet Battleship running at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        srv.shutdown()


if __name__ == "__main__":
    main()
