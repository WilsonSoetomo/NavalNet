"""Opponent policies for Battleship - used by the environment."""

import random
from typing import Protocol

from .constants import GRID_SIZE, HORIZONTAL, SHIP_SIZES, VERTICAL
from .game_engine import Board


class Opponent(Protocol):
    """Interface for opponent placement and shooting."""

    def place_ships(self, board: Board) -> None:
        """Place all ships on the given board."""
        ...

    def get_shot(self, observation: list[list[int]]) -> int:
        """
        Choose a cell to shoot (0-99). observation is 10x10 attack board
        (0=unknown, 1=miss, 2=hit, 3=sunk). Must not shoot unknown cells.
        """
        ...


class RandomOpponent:
    """Places ships randomly and shoots at random valid (unshot) cells."""

    def place_ships(self, board: Board) -> None:
        for length in SHIP_SIZES:
            placed = False
            while not placed:
                row = random.randint(0, GRID_SIZE - 1)
                col = random.randint(0, GRID_SIZE - 1)
                orientation = random.choice([HORIZONTAL, VERTICAL])
                placed = board.place_ship(length, row, col, orientation)

    def get_shot(self, observation: list[list[int]]) -> int:
        unshot = [
            r * GRID_SIZE + c
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if observation[r][c] == 0
        ]
        return random.choice(unshot) if unshot else 0
