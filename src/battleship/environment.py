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
        self.render_mode = render_mode

        # Observation: 10x10 matrix (0=Unknown, 1=Miss, 2=Hit, 3=Sunk)
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8
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

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._game.reset()
        self._total_turns = 0

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
                if sunk:
                    reward += self.reward_sink
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
        matrix = self._game.opponent_board.observation_matrix()
        return np.array(matrix, dtype=np.int8)

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
