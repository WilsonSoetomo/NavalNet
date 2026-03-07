"""
Gymnasium Battleship environment for RL training.
Observation: 10x10 matrix (0=Unknown, 1=Miss, 2=Hit, 3=Sunk).
Actions: Placement (row, col, orientation) per ship; Shooting (0-99).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .constants import (
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
from .game_engine import BattleshipGame
from .opponents import Opponent, RandomOpponent


class BattleshipEnv(gym.Env):
    """
    Battleship environment for training RL agents.
    Agent places ships, then takes turns shooting at opponent's board.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(
        self,
        opponent: Opponent | None = None,
        reward_hit: float = 1.0,
        reward_sink: float = 5.0,
        reward_miss: float = -0.1,
        reward_win: float = 100.0,
        reward_lose: float = -100.0,
        reward_per_turn: float = -0.05,
        reward_efficient_sink: float = 2.0,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.opponent = opponent or RandomOpponent()
        self.reward_hit = reward_hit
        self.reward_sink = reward_sink
        self.reward_miss = reward_miss
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        self.reward_per_turn = reward_per_turn
        self.reward_efficient_sink = reward_efficient_sink
        self.render_mode = render_mode

        # Observation: (C, 10, 10) multi-channel binary feature planes
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(NUM_OBS_CHANNELS, GRID_SIZE, GRID_SIZE),
            dtype=np.float32,
        )

        # Action: Placement = MultiDiscrete([10, 10, 2]); Shooting = Discrete(100)
        # We use a single flat action for compatibility; env interprets by phase
        self._placement_space = spaces.MultiDiscrete([GRID_SIZE, GRID_SIZE, 2], seed=seed)
        self._shooting_space = spaces.Discrete(NUM_CELLS, seed=seed)
        self.action_space = spaces.Discrete(
            max(GRID_SIZE * GRID_SIZE * 2, NUM_CELLS),
            seed=seed,
        )

        self._game = BattleshipGame()
        self._rng = np.random.default_rng(seed)
        self._total_turns = 0

        # Shots-to-sink tracking
        self._agent_shot_attempts = 0
        self._ship_first_hit_shot: dict[int, int] = {}  # id(ship) -> shot #
        self._sink_stats: list[dict] = []

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._game.reset()
        self._total_turns = 0
        self._agent_shot_attempts = 0
        self._ship_first_hit_shot = {}
        self._sink_stats = []

        # Opponent places ships first
        self.opponent.place_ships(self._game.opponent_board)

        obs = self._get_observation()
        info = {
            "phase": "placement",
            "ship_index": 0,
            "ships_remaining": list(SHIP_SIZES),
        }
        return obs, info

    def step(
        self, action: int | tuple
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._game.phase == "placement":
            return self._step_placement(action)
        return self._step_shooting(action)

    def _step_placement(self, action: int | tuple) -> tuple[np.ndarray, float, bool, bool, dict]:
        ship_index = self._game.agent_ships_placed
        if ship_index >= len(SHIP_SIZES):
            self._game._phase = "shooting"
            return self._get_observation(), 0.0, False, False, {"phase": "shooting"}

        length = SHIP_SIZES[ship_index]
        if isinstance(action, (tuple, list)):
            row, col, orient = int(action[0]), int(action[1]), int(action[2])
        else:
            # Decode flat: row*10*2 + col*2 + orient
            flat = int(action) % (GRID_SIZE * GRID_SIZE * 2)
            orient = flat % 2
            flat //= 2
            col = flat % GRID_SIZE
            row = flat // GRID_SIZE

        orient = HORIZONTAL if orient == 0 else VERTICAL
        ok = self._game.place_agent_ship(length, row, col, orient)

        reward = 0.0 if ok else -1.0  # Invalid placement penalty
        ship_index = self._game.agent_ships_placed
        if ship_index >= len(SHIP_SIZES):
            self._game._phase = "shooting"
            self._game._turn = "agent"

        obs = self._get_observation()
        terminated = False
        truncated = False
        info = {
            "phase": self._game.phase,
            "ship_index": ship_index,
            "placement_valid": ok,
        }
        return obs, reward, terminated, truncated, info

    def _step_shooting(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action) % NUM_CELLS
        row, col = action // GRID_SIZE, action % GRID_SIZE

        reward = self.reward_per_turn
        terminated = False
        truncated = False
        info: dict = {}
        self._agent_shot_attempts += 1

        # Agent's turn
        if self._game.turn == "agent":
            if self._game.opponent_board.is_shot(row, col):
                reward += -0.5  # Repeat shot penalty
                obs = self._get_observation()
                info = {"phase": "shooting", "repeat_shot": True, "turns": self._total_turns}
                return obs, reward, terminated, truncated, info

            hit, sunk = self._game.agent_shoot(row, col)
            if hit:
                reward += self.reward_hit

                ship = self._game.opponent_board.get_ship_at(row, col)
                ship_key = id(ship)
                if ship_key not in self._ship_first_hit_shot:
                    self._ship_first_hit_shot[ship_key] = self._agent_shot_attempts

                if sunk:
                    reward += self.reward_sink
                    first_hit = self._ship_first_hit_shot[ship_key]
                    shots_to_sink = self._agent_shot_attempts - first_hit + 1
                    efficiency = ship.length / shots_to_sink
                    reward += self.reward_efficient_sink * efficiency
                    self._sink_stats.append({
                        "ship_length": ship.length,
                        "shots_to_sink": shots_to_sink,
                        "efficiency": efficiency,
                    })
            else:
                reward += self.reward_miss

            if self._game.agent_won():
                reward += self.reward_win
                terminated = True
            elif not hit:
                # Opponent's turn
                self._run_opponent_turn()
                if self._game.opponent_won():
                    reward += self.reward_lose
                    terminated = True

        obs = self._get_observation()
        self._total_turns += 1
        info = {
            "phase": "shooting",
            "turns": self._total_turns,
            "agent_won": self._game.agent_won(),
            "opponent_won": self._game.opponent_won(),
        }
        return obs, reward, terminated, truncated, info

    def _run_opponent_turn(self) -> None:
        """Run opponent shots until they miss (hit grants another turn)."""
        while self._game.turn == "opponent" and not self._game.game_over():
            obs_matrix = self._game.agent_board.observation_matrix()
            cell = self.opponent.get_shot(obs_matrix)
            row, col = cell // GRID_SIZE, cell % GRID_SIZE
            self._game.opponent_shoot(row, col)

    def _get_observation(self) -> np.ndarray:
        """Multi-channel observation: (C, 10, 10) float32 binary planes.
        Ch0: unknown, Ch1: miss, Ch2: hit (unsunk), Ch3: sunk,
        Ch4: unshot cells adjacent to any unsunk hit (targeting hint)."""
        raw = self._game.opponent_board.observation_matrix()
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
        # Channel 4: unshot neighbours of unsunk hits
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if obs[2, r, c] == 1.0:  # unsunk hit
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                            if obs[0, nr, nc] == 1.0:  # still unknown
                                obs[4, nr, nc] = 1.0
        return obs

    def get_placement_observation(self) -> np.ndarray:
        """
        Return 10x10 agent board for placement phase: 0 = empty, 1 = ship.
        Use with info['ship_index'] so the placement head knows which ship to place.
        """
        matrix = self._game.agent_board.placement_matrix()
        return np.array(matrix, dtype=np.int8)

    def get_valid_placement_mask(self) -> np.ndarray:
        """Returns a mask of valid (row, col, orient) for current ship (for masking invalid actions)."""
        ship_index = self._game.agent_ships_placed
        if ship_index >= len(SHIP_SIZES):
            return np.zeros(GRID_SIZE * GRID_SIZE * 2, dtype=bool)
        length = SHIP_SIZES[ship_index]
        mask = np.zeros(GRID_SIZE * GRID_SIZE * 2, dtype=bool)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                for orient in (0, 1):
                    o = HORIZONTAL if orient == 0 else VERTICAL
                    if self._game.agent_board.can_place_ship(length, row, col, o):
                        idx = (row * GRID_SIZE + col) * 2 + orient
                        mask[idx] = True
        return mask

    def get_valid_shooting_mask(self) -> np.ndarray:
        """Returns a mask of valid cells to shoot (0 = unshot)."""
        mask = np.zeros(NUM_CELLS, dtype=bool)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if not self._game.opponent_board.is_shot(r, c):
                    mask[r * GRID_SIZE + c] = True
        return mask

    def get_sink_stats(self) -> list[dict]:
        """
        Per-ship sink efficiency for this episode.
        Each entry: {ship_length, shots_to_sink, efficiency}.
        shots_to_sink = total agent shots from first hit to sinking shot.
        efficiency   = ship_length / shots_to_sink  (1.0 = perfect focus).
        """
        return list(self._sink_stats)

    def get_full_board_state(self) -> dict:
        """
        Return full board state for visualization.
        Returns dict with raw grid data for both boards + ship positions.
        """
        from .game_engine import CellState

        def grid_to_array(board) -> np.ndarray:
            """Convert board grid to numpy array with cell states."""
            arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    arr[r, c] = int(board._grid[r][c])
            return arr

        def ships_to_list(board) -> list[dict]:
            return [
                {
                    "length": s.length,
                    "row": s.row,
                    "col": s.col,
                    "orientation": s.orientation,
                    "hits": s.hits,
                    "sunk": s.is_sunk,
                }
                for s in board._ships
            ]

        return {
            "agent_grid": grid_to_array(self._game.agent_board),
            "opponent_grid": grid_to_array(self._game.opponent_board),
            "agent_ships": ships_to_list(self._game.agent_board),
            "opponent_ships": ships_to_list(self._game.opponent_board),
            "phase": self._game.phase,
            "turn": self._game.turn,
            "agent_won": self._game.agent_won(),
            "opponent_won": self._game.opponent_won(),
            "total_turns": self._total_turns,
        }
