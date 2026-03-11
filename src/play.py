#!/usr/bin/env python3
"""Interactive Battleship -- play against a trained model or bot in the terminal."""

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

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

# ── ANSI helpers ────────────────────────────────────────────────────
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GRN = "\033[92m"
YEL = "\033[93m"
BLU = "\033[94m"
MAG = "\033[95m"
CYN = "\033[96m"
GRY = "\033[90m"

SHIP_NAMES = {5: "Carrier", 4: "Battleship", 3: "Cruiser", 2: "Destroyer"}


# ── Build the 6-channel observation any board (for model opponent) ──
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


# ── Model opponent wrapper ──────────────────────────────────────────
class ModelOpponent:
    def __init__(self, path: str, model_type: str = "dqn"):
        import torch  # noqa: delayed import so non-model games stay fast

        self.model_type = model_type

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
                self.agent.policy_net = DQNNetwork(in_channels=self.in_channels).to(
                    self.agent.device
                )
                self.agent.target_net = DQNNetwork(in_channels=self.in_channels).to(
                    self.agent.device
                )
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
        if self.in_channels < obs.shape[0]:
            obs = obs[: self.in_channels]
        mask = _valid_mask(board)
        if self.model_type == "dqn":
            return self.agent.select_shooting_action(obs, mask, deterministic=True)
        action, _ = self.agent.select_shooting_action(obs, mask, deterministic=True)
        return action


# ── Rendering ───────────────────────────────────────────────────────
def _cell_defense(board: Board, r: int, c: int) -> str:
    s = board.get_cell_state(r, c)
    if s == CellState.WATER:
        return f"{BLU}\u00b7{RST}"
    if s == CellState.SHIP:
        return f"{GRN}{BOLD}\u25a0{RST}"
    if s == CellState.HIT:
        return f"{RED}{BOLD}\u2588{RST}"
    if s == CellState.MISS:
        return f"{GRY}\u25cb{RST}"
    if s == CellState.SUNK:
        return f"{MAG}{BOLD}\u2573{RST}"
    return "?"


def _cell_attack(board: Board, r: int, c: int, reveal: bool = False) -> str:
    obs = board.observation_matrix()
    v = obs[r][c]
    if v == CELL_UNKNOWN:
        if reveal and board.get_cell_state(r, c) == CellState.SHIP:
            return f"{DIM}{GRN}\u25a1{RST}"
        return f"{BLU}\u00b7{RST}"
    if v == CELL_MISS:
        return f"{GRY}\u25cb{RST}"
    if v == CELL_HIT:
        return f"{RED}{BOLD}\u2588{RST}"
    if v == CELL_SUNK:
        return f"{MAG}{BOLD}\u2573{RST}"
    return "?"


def _render_boards(game: BattleshipGame, reveal: bool = False) -> None:
    hdr = "  A B C D E F G H I J"
    print()
    print(
        f"  {CYN}{BOLD}YOUR BOARD (Defense){RST}"
        f"          {YEL}{BOLD}ATTACK BOARD{RST}"
    )
    print(f"  {hdr}        {hdr}")
    for r in range(GRID_SIZE):
        left = f"  {r:2d}"
        for c in range(GRID_SIZE):
            left += f" {_cell_defense(game.agent_board, r, c)}"
        right = f"  {r:2d}"
        for c in range(GRID_SIZE):
            right += f" {_cell_attack(game.opponent_board, r, c, reveal)}"
        print(f"{left}      {right}")
    print()
    print(
        f"  {BOLD}Legend:{RST}  "
        f"{BLU}\u00b7{RST}=water  "
        f"{GRN}{BOLD}\u25a0{RST}=ship  "
        f"{RED}{BOLD}\u2588{RST}=hit  "
        f"{GRY}\u25cb{RST}=miss  "
        f"{MAG}{BOLD}\u2573{RST}=sunk"
    )


# ── Coordinate parsing ──────────────────────────────────────────────
def _parse_coord(text: str) -> tuple[int, int] | None:
    """Accept A5, a5, 3 7, 3,7 etc. Columns A-J, Rows 0-9."""
    text = text.strip().upper()
    if not text:
        return None
    if text[0].isalpha() and len(text) >= 2:
        col = ord(text[0]) - ord("A")
        try:
            row = int(text[1:].strip())
        except ValueError:
            return None
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return row, col
        return None
    parts = text.replace(",", " ").split()
    if len(parts) == 2:
        try:
            row, col = int(parts[0]), int(parts[1])
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                return row, col
        except ValueError:
            pass
    return None


def _coord_str(r: int, c: int) -> str:
    return f"{chr(ord('A') + c)}{r}"


# ── Ship placement ──────────────────────────────────────────────────
def _place_ships_random(game: BattleshipGame) -> None:
    for length in SHIP_SIZES:
        while True:
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            ori = random.choice([HORIZONTAL, VERTICAL])
            if game.place_agent_ship(length, row, col, ori):
                break


def _show_placement_board(game: BattleshipGame) -> None:
    hdr = "  A B C D E F G H I J"
    print(f"\n  {CYN}{BOLD}YOUR BOARD{RST}")
    print(f"  {hdr}")
    for r in range(GRID_SIZE):
        row_str = f"  {r:2d}"
        for c in range(GRID_SIZE):
            row_str += f" {_cell_defense(game.agent_board, r, c)}"
        print(row_str)


def _place_ships_interactive(game: BattleshipGame) -> None:
    print(f"\n{BOLD}=== SHIP PLACEMENT ==={RST}")
    print(f"  Enter: {CYN}COL ROW H/V{RST}  (e.g. {CYN}A 3 H{RST} or {CYN}B 5 V{RST})")
    print(f"  Type {YEL}random{RST} to auto-place remaining ships.\n")

    placed = 0
    for i, length in enumerate(SHIP_SIZES):
        name = SHIP_NAMES.get(length, f"Ship({length})")
        while True:
            _show_placement_board(game)
            inp = input(
                f"\n  Place {GRN}{BOLD}{name}{RST} (len {length}): "
            ).strip()

            if inp.upper() == "RANDOM":
                for j in range(i, len(SHIP_SIZES)):
                    ln = SHIP_SIZES[j]
                    while True:
                        rr = random.randint(0, GRID_SIZE - 1)
                        cc = random.randint(0, GRID_SIZE - 1)
                        ori = random.choice([HORIZONTAL, VERTICAL])
                        if game.place_agent_ship(ln, rr, cc, ori):
                            break
                print(f"  {GRN}Placed remaining ships randomly.{RST}")
                return

            parts = inp.upper().replace(",", " ").split()
            if len(parts) < 3:
                print(f"  {RED}Format: COL ROW H/V  (e.g. A 3 H){RST}")
                continue

            try:
                col_ch = parts[0]
                col = ord(col_ch) - ord("A") if col_ch.isalpha() else int(col_ch)
                row = int(parts[1])
                ori = HORIZONTAL if parts[2] in ("H", "0") else VERTICAL
            except (ValueError, IndexError):
                print(f"  {RED}Could not parse. Try again.{RST}")
                continue

            if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
                print(f"  {RED}Out of bounds (Row 0-9, Col A-J).{RST}")
                continue

            if game.place_agent_ship(length, row, col, ori):
                placed += 1
                print(f"  {GRN}Placed {name}!{RST}")
                break
            else:
                print(f"  {RED}Invalid -- ship doesn't fit or overlaps.{RST}")


# ── Main game loop ──────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Play Battleship against a bot or trained model."
    )
    ap.add_argument(
        "--opponent",
        default="random",
        help="'random', 'hunt_target', or path to a .pt model checkpoint",
    )
    ap.add_argument(
        "--model-type",
        default="dqn",
        choices=["dqn", "ppo"],
        help="Model architecture when --opponent is a .pt file (default: dqn)",
    )
    ap.add_argument(
        "--placement",
        default="interactive",
        choices=["interactive", "random"],
        help="How to place your ships (default: interactive)",
    )
    ap.add_argument(
        "--reveal",
        action="store_true",
        help="Show opponent ships on the attack board (cheat mode)",
    )
    args = ap.parse_args()

    # ── Opponent setup ──────────────────────────────────────────────
    if args.opponent == "random":
        opp = RandomOpponent()
        opp_name = "Random Bot"
        is_model = False
    elif args.opponent == "hunt_target":
        opp = HuntTargetOpponent()
        opp_name = "Hunt/Target Bot"
        is_model = False
    else:
        opp = ModelOpponent(args.opponent, args.model_type)
        opp_name = f"{args.model_type.upper()} Model ({os.path.basename(args.opponent)})"
        is_model = True

    # ── New game ────────────────────────────────────────────────────
    game = BattleshipGame()

    os.system("clear 2>/dev/null || cls 2>/dev/null")
    print(f"\n{BOLD}{CYN}{'=' * 52}")
    print(f"          BATTLESHIP  --  NavalNet")
    print(f"{'=' * 52}{RST}")
    print(f"  Opponent : {YEL}{BOLD}{opp_name}{RST}")
    print(f"  Controls : Column-letter + Row-number (e.g. A5)")
    print(f"             Type {YEL}q{RST} to quit any time.\n")

    # Place opponent ships
    opp.place_ships(game.opponent_board)

    # Place player ships
    if args.placement == "random":
        _place_ships_random(game)
        print(f"  {GRN}Your ships placed randomly.{RST}")
    else:
        _place_ships_interactive(game)

    _show_placement_board(game)
    input(f"\n  {DIM}Ships placed. Press Enter to start...{RST}")

    game._phase = "shooting"
    game._turn = "agent"

    player_shots = 0
    opp_shots = 0

    while not game.game_over():
        os.system("clear 2>/dev/null || cls 2>/dev/null")

        opp_alive = sum(1 for s in game.opponent_board._ships if not s.is_sunk)
        my_alive = sum(1 for s in game.agent_board._ships if not s.is_sunk)
        opp_total = len(game.opponent_board._ships)
        my_total = len(game.agent_board._ships)

        print(f"\n{BOLD}{CYN}{'=' * 52}")
        print(f"  BATTLESHIP  |  vs {opp_name}")
        print(f"{'=' * 52}{RST}")
        print(
            f"  Your ships: {GRN}{my_alive}{RST}/{my_total}  |  "
            f"Enemy ships: {RED}{opp_alive}{RST}/{opp_total}  |  "
            f"Shots: you {player_shots} / them {opp_shots}"
        )

        _render_boards(game, reveal=args.reveal)

        # ── Player turn(s) (hit = keep shooting) ───────────────────
        if game._turn == "agent":
            while game._turn == "agent" and not game.game_over():
                while True:
                    inp = input(f"\n  {BOLD}Your shot{RST} (e.g. A5): ").strip()
                    if inp.lower() in ("q", "quit", "exit"):
                        print(f"\n  {YEL}Game abandoned.{RST}\n")
                        return
                    coord = _parse_coord(inp)
                    if coord is None:
                        print(f"  {RED}Invalid. Use column+row like A5, B0, J9.{RST}")
                        continue
                    r, c = coord
                    if game.opponent_board.is_shot(r, c):
                        print(f"  {RED}Already shot there.{RST}")
                        continue
                    break

                hit, sunk = game.agent_shoot(r, c)
                player_shots += 1

                if sunk:
                    ship = game.opponent_board.get_ship_at(r, c)
                    sname = SHIP_NAMES.get(ship.length, "Ship") if ship else "Ship"
                    print(
                        f"  {RED}{BOLD}HIT & SUNK {sname} at "
                        f"{_coord_str(r, c)}!{RST}"
                    )
                elif hit:
                    print(f"  {RED}{BOLD}HIT at {_coord_str(r, c)}!{RST}")
                else:
                    print(f"  {GRY}Miss at {_coord_str(r, c)}.{RST}")

            if game.game_over():
                break

        # ── Opponent turn(s) ────────────────────────────────────────
        opp_msgs: list[str] = []
        while game._turn == "opponent" and not game.game_over():
            if is_model:
                action = opp.get_shot(game.agent_board)
            else:
                obs_matrix = game.agent_board.observation_matrix()
                action = opp.get_shot(obs_matrix)

            orow, ocol = action // GRID_SIZE, action % GRID_SIZE
            ohit, osunk = game.opponent_shoot(orow, ocol)
            opp_shots += 1

            cs = _coord_str(orow, ocol)
            if osunk:
                ship = game.agent_board.get_ship_at(orow, ocol)
                sname = SHIP_NAMES.get(ship.length, "Ship") if ship else "Ship"
                opp_msgs.append(
                    f"  {RED}Enemy HIT & SUNK your {sname} at {cs}!{RST}"
                )
            elif ohit:
                opp_msgs.append(f"  {YEL}Enemy HIT at {cs}!{RST}")
            else:
                opp_msgs.append(f"  {DIM}Enemy miss at {cs}.{RST}")

        if opp_msgs:
            print(f"\n  {BOLD}-- Opponent's turn --{RST}")
            for m in opp_msgs:
                print(m)

        if not game.game_over():
            game._turn = "agent"
            input(f"\n  {DIM}Press Enter to continue...{RST}")

    # ── Game over ───────────────────────────────────────────────────
    os.system("clear 2>/dev/null || cls 2>/dev/null")
    _render_boards(game, reveal=True)

    if game.agent_won():
        print(f"\n  {GRN}{BOLD}YOU WIN!{RST}")
        print(f"  Sunk all enemy ships in {player_shots} shots.")
    else:
        print(f"\n  {RED}{BOLD}YOU LOSE{RST}")
        print(f"  {opp_name} sunk your fleet in {opp_shots} shots.")

    print(
        f"\n  Final -- Your shots: {player_shots}  |  Enemy shots: {opp_shots}\n"
    )


if __name__ == "__main__":
    main()
