"""
Battleship game engine - core logic for NavalNet RL training.
Separate from RL logic; handles board state, placement, and shooting.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

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


class CellState(IntEnum):
    """Internal board cell representation."""

    WATER = 0
    SHIP = 1
    HIT = 2
    MISS = 3
    SUNK = 4


@dataclass
class Ship:
    """A ship on the board."""

    length: int
    row: int
    col: int
    orientation: int  # HORIZONTAL or VERTICAL
    hits: int = 0

    def cells(self) -> Iterator[tuple[int, int]]:
        """Yield (row, col) for each cell the ship occupies."""
        for i in range(self.length):
            if self.orientation == HORIZONTAL:
                yield self.row, self.col + i
            else:
                yield self.row + i, self.col

    @property
    def is_sunk(self) -> bool:
        return self.hits >= self.length


class Board:
    """
    10x10 Battleship board. Tracks ship placement and shot results.
    """

    def __init__(self):
        self._grid: list[list[CellState]] = [
            [CellState.WATER] * GRID_SIZE for _ in range(GRID_SIZE)
        ]
        self._ships: list[Ship] = []
        self._ship_at: dict[tuple[int, int], Ship] = {}  # (r,c) -> Ship

    def reset(self) -> None:
        """Clear the board for a new game."""
        self._grid = [[CellState.WATER] * GRID_SIZE for _ in range(GRID_SIZE)]
        self._ships = []
        self._ship_at = {}

    def place_ship(self, length: int, row: int, col: int, orientation: int) -> bool:
        """
        Place a ship. Returns True if placement is valid, False otherwise.
        """
        cells = list(
            Ship(length=length, row=row, col=col, orientation=orientation).cells()
        )
        if not self._valid_placement(cells):
            return False
        ship = Ship(length=length, row=row, col=col, orientation=orientation)
        self._ships.append(ship)
        for r, c in cells:
            self._grid[r][c] = CellState.SHIP
            self._ship_at[(r, c)] = ship
        return True

    def _valid_placement(self, cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
                return False
            if self._grid[r][c] != CellState.WATER:
                return False
        return True

    def can_place_ship(self, length: int, row: int, col: int, orientation: int) -> bool:
        """Check if a ship can be placed without actually placing it."""
        cells = list(
            Ship(length=length, row=row, col=col, orientation=orientation).cells()
        )
        return self._valid_placement(cells)

    def shoot(self, row: int, col: int) -> tuple[bool, bool]:
        """
        Shoot at (row, col). Returns (hit, sunk) where:
        - hit: whether we hit a ship
        - sunk: whether that shot sank a ship (implies hit)
        """
        if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
            raise ValueError(f"Invalid cell ({row}, {col})")

        if self._grid[row][col] in (CellState.HIT, CellState.MISS, CellState.SUNK):
            raise ValueError(f"Already shot at ({row}, {col})")

        if self._grid[row][col] == CellState.WATER:
            self._grid[row][col] = CellState.MISS
            return False, False

        # Hit a ship
        ship = self._ship_at[(row, col)]
        ship.hits += 1
        if ship.is_sunk:
            for r, c in ship.cells():
                self._grid[r][c] = CellState.SUNK
            return True, True
        self._grid[row][col] = CellState.HIT
        return True, False

    def get_cell_state(self, row: int, col: int) -> CellState:
        return self._grid[row][col]

    def is_shot(self, row: int, col: int) -> bool:
        """True if this cell has been shot at."""
        s = self._grid[row][col]
        return s in (CellState.HIT, CellState.MISS, CellState.SUNK)

    def all_ships_sunk(self) -> bool:
        return all(s.is_sunk for s in self._ships)

    def observation_matrix(self) -> list[list[int]]:
        """
        Return 10x10 matrix for RL observation space:
        0 = Unknown (not shot), 1 = Miss, 2 = Hit, 3 = Sunk
        """
        obs = [[CELL_UNKNOWN] * GRID_SIZE for _ in range(GRID_SIZE)]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                s = self._grid[r][c]
                if s == CellState.MISS:
                    obs[r][c] = CELL_MISS
                elif s == CellState.HIT:
                    obs[r][c] = CELL_HIT
                elif s == CellState.SUNK:
                    obs[r][c] = CELL_SUNK
        return obs

    def placement_matrix(self) -> list[list[int]]:
        """
        Return 10x10 matrix for placement-phase observation: 0 = empty, 1 = ship.
        Used by the placement head to see current board state.
        """
        return [
            [1 if self._grid[r][c] == CellState.SHIP else 0 for c in range(GRID_SIZE)]
            for r in range(GRID_SIZE)
        ]


class BattleshipGame:
    """
    Full game: two boards (agent vs opponent). Manages turns and game over.
    """

    def __init__(self):
        self.agent_board = Board()
        self.opponent_board = Board()
        self.agent_ships_placed = 0
        self.opponent_ships_placed = 0
        self._phase: str = "placement"  # "placement" | "shooting"
        self._turn: str = "agent"  # "agent" | "opponent"

    def reset(self) -> None:
        self.agent_board.reset()
        self.opponent_board.reset()
        self.agent_ships_placed = 0
        self.opponent_ships_placed = 0
        self._phase = "placement"
        self._turn = "agent"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def turn(self) -> str:
        return self._turn

    def place_agent_ship(self, length: int, row: int, col: int, orientation: int) -> bool:
        ok = self.agent_board.place_ship(length, row, col, orientation)
        if ok:
            self.agent_ships_placed += 1
            if self.agent_ships_placed >= len(SHIP_SIZES):
                self._phase = "shooting"
        return ok

    def place_opponent_ship(self, length: int, row: int, col: int, orientation: int) -> bool:
        ok = self.opponent_board.place_ship(length, row, col, orientation)
        if ok:
            self.opponent_ships_placed += 1
        return ok

    def agent_shoot(self, row: int, col: int) -> tuple[bool, bool]:
        """Agent shoots at opponent's board. Returns (hit, sunk)."""
        hit, sunk = self.opponent_board.shoot(row, col)
        if not hit:
            self._turn = "opponent"
        return hit, sunk

    def opponent_shoot(self, row: int, col: int) -> tuple[bool, bool]:
        """Opponent shoots at agent's board. Returns (hit, sunk)."""
        hit, sunk = self.agent_board.shoot(row, col)
        if not hit:
            self._turn = "agent"
        return hit, sunk

    def agent_won(self) -> bool:
        return self.opponent_board.all_ships_sunk()

    def opponent_won(self) -> bool:
        return self.agent_board.all_ships_sunk()

    def game_over(self) -> bool:
        return self.agent_won() or self.opponent_won()
