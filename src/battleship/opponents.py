"""Opponent policies for Battleship - used by the environment."""

import random
from typing import Protocol

from .constants import (
    CELL_HIT,
    CELL_UNKNOWN,
    GRID_SIZE,
    HORIZONTAL,
    SHIP_SIZES,
    VERTICAL,
)
from .game_engine import Board

_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


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


def _random_place_ships(board: Board) -> None:
    """Standard random ship placement shared by all opponents."""
    for length in SHIP_SIZES:
        placed = False
        while not placed:
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            orientation = random.choice([HORIZONTAL, VERTICAL])
            placed = board.place_ship(length, row, col, orientation)


class RandomOpponent:
    """Places ships randomly and shoots at random valid (unshot) cells."""

    def place_ships(self, board: Board) -> None:
        _random_place_ships(board)

    def get_shot(self, observation: list[list[int]]) -> int:
        unshot = [
            r * GRID_SIZE + c
            for r in range(GRID_SIZE)
            for c in range(GRID_SIZE)
            if observation[r][c] == CELL_UNKNOWN
        ]
        return random.choice(unshot) if unshot else 0


class HuntTargetOpponent:
    """
    Hunt / Target strategy:
      Hunt  – checkerboard parity search (smallest ship = 2, so every ship
              must occupy at least one parity cell).
      Target – on an unsunk hit, extend the line; try ends of contiguous
              hit segments first, then all four neighbours of isolated hits.
    Switches back to hunt automatically once no unsunk hits remain.
    """

    def __init__(self, parity: int | None = None):
        self._parity = parity if parity is not None else random.randint(0, 1)

    def place_ships(self, board: Board) -> None:
        _random_place_ships(board)

    def get_shot(self, observation: list[list[int]]) -> int:
        targets = self._build_target_list(observation)
        if targets:
            r, c = targets[0]
            return r * GRID_SIZE + c
        return self._hunt(observation)

    # ── Target mode ──────────────────────────────────────────────────

    def _build_target_list(self, obs: list[list[int]]) -> list[tuple[int, int]]:
        """Score candidate cells around unsunk hits; return best-first."""
        unsunk = set()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if obs[r][c] == CELL_HIT:
                    unsunk.add((r, c))

        if not unsunk:
            return []

        scored: dict[tuple[int, int], int] = {}

        for r, c in unsunk:
            h_seg = self._contiguous(unsunk, r, c, horizontal=True)
            v_seg = self._contiguous(unsunk, r, c, horizontal=False)

            if len(h_seg) > 1:
                min_c = min(cc for _, cc in h_seg)
                max_c = max(cc for _, cc in h_seg)
                if min_c > 0 and obs[r][min_c - 1] == CELL_UNKNOWN:
                    scored[(r, min_c - 1)] = max(scored.get((r, min_c - 1), 0), 10)
                if max_c < GRID_SIZE - 1 and obs[r][max_c + 1] == CELL_UNKNOWN:
                    scored[(r, max_c + 1)] = max(scored.get((r, max_c + 1), 0), 10)

            if len(v_seg) > 1:
                min_r = min(rr for rr, _ in v_seg)
                max_r = max(rr for rr, _ in v_seg)
                if min_r > 0 and obs[min_r - 1][c] == CELL_UNKNOWN:
                    scored[(min_r - 1, c)] = max(scored.get((min_r - 1, c), 0), 10)
                if max_r < GRID_SIZE - 1 and obs[max_r + 1][c] == CELL_UNKNOWN:
                    scored[(max_r + 1, c)] = max(scored.get((max_r + 1, c), 0), 10)

            if len(h_seg) <= 1 and len(v_seg) <= 1:
                for dr, dc in _DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if obs[nr][nc] == CELL_UNKNOWN:
                            scored[(nr, nc)] = max(scored.get((nr, nc), 0), 1)

        items = list(scored.items())
        random.shuffle(items)
        items.sort(key=lambda x: -x[1])
        return [cell for cell, _ in items]

    @staticmethod
    def _contiguous(
        hit_set: set[tuple[int, int]], r: int, c: int, horizontal: bool
    ) -> list[tuple[int, int]]:
        """Return contiguous segment of hits through (r, c) along one axis."""
        seg = [(r, c)]
        for sign in (1, -1):
            for step in range(1, GRID_SIZE):
                nr = r if horizontal else r + sign * step
                nc = c + sign * step if horizontal else c
                if (nr, nc) in hit_set:
                    seg.append((nr, nc))
                else:
                    break
        return seg

    # ── Hunt mode ────────────────────────────────────────────────────

    def _hunt(self, obs: list[list[int]]) -> int:
        """Checkerboard parity search with slight centre bias."""
        unshot = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if obs[r][c] == CELL_UNKNOWN:
                    unshot.append((r, c))

        parity_cells = [(r, c) for r, c in unshot if (r + c) % 2 == self._parity]
        pool = parity_cells if parity_cells else unshot
        if not pool:
            return 0

        mid = (GRID_SIZE - 1) / 2.0
        pool.sort(key=lambda rc: -(abs(rc[0] - mid) + abs(rc[1] - mid)))
        top = max(1, len(pool) // 3)
        cell = random.choice(pool[:top]) if random.random() < 0.6 else random.choice(pool)
        return cell[0] * GRID_SIZE + cell[1]
