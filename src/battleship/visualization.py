"""
Visualization utilities for Battleship game state.
"""

from .constants import CELL_HIT, CELL_MISS, CELL_SUNK, CELL_UNKNOWN, GRID_SIZE


def print_board(observation: list[list[int]], title: str = "Board") -> None:
    """
    Print a battleship board observation in a readable format.
    
    Args:
        observation: 10x10 matrix where:
            0 = Unknown (not shot)
            1 = Miss
            2 = Hit
            3 = Sunk
        title: Title to display above the board
    """
    print(f"\n{title}:")
    print("   " + " ".join(str(i) for i in range(10)))
    
    for r in range(GRID_SIZE):
        row_str = f"{r:2d} "
        for c in range(GRID_SIZE):
            cell = observation[r][c]
            if cell == CELL_UNKNOWN:
                row_str += "· "  # Unknown
            elif cell == CELL_MISS:
                row_str += "O "  # Miss
            elif cell == CELL_HIT:
                row_str += "X "  # Hit
            elif cell == CELL_SUNK:
                row_str += "# "  # Sunk
            else:
                row_str += "? "  # Unknown state
        print(row_str)
    print()


def print_placement_board(placement_matrix: list[list[int]], title: str = "Placement Board") -> None:
    """
    Print a placement board showing where ships are placed.
    
    Args:
        placement_matrix: 10x10 matrix where 0 = empty, 1 = ship
        title: Title to display above the board
    """
    print(f"\n{title}:")
    print("   " + " ".join(str(i) for i in range(10)))
    
    for r in range(GRID_SIZE):
        row_str = f"{r:2d} "
        for c in range(GRID_SIZE):
            cell = placement_matrix[r][c]
            if cell == 0:
                row_str += ". "  # Empty
            else:
                row_str += "S "  # Ship
        print(row_str)
    print()


def print_game_summary(
    observation: list[list[int]],
    shots_taken: int,
    phase: str,
    ship_index: int | None = None,
) -> None:
    """
    Print a summary of the current game state.
    
    Args:
        observation: Current observation matrix
        shots_taken: Number of shots taken so far
        phase: Current phase ("placement" or "shooting")
        ship_index: Current ship index (for placement phase)
    """
    hits = sum(1 for r in observation for c in observation[r] if observation[r][c] == CELL_HIT)
    misses = sum(1 for r in observation for c in observation[r] if observation[r][c] == CELL_MISS)
    sunk = sum(1 for r in observation for c in observation[r] if observation[r][c] == CELL_SUNK)
    
    print(f"Phase: {phase}")
    if ship_index is not None:
        print(f"Placing ship {ship_index + 1}/5")
    print(f"Shots: {shots_taken} | Hits: {hits} | Misses: {misses} | Sunk: {sunk}")


def print_action(action: int, phase: str, ship_index: int | None = None) -> None:
    """
    Print a human-readable action.
    
    Args:
        action: Action taken (flat action index)
        phase: Current phase
        ship_index: Ship index (for placement)
    """
    if phase == "placement":
        # Decode: row*10*2 + col*2 + orient
        flat = action % (GRID_SIZE * GRID_SIZE * 2)
        orient = flat % 2
        flat //= 2
        col = flat % GRID_SIZE
        row = flat // GRID_SIZE
        orient_str = "H" if orient == 0 else "V"
        print(f"  Action: Place ship at ({row}, {col}) orientation {orient_str}")
    else:
        # Shooting: action is 0-99
        row = action // GRID_SIZE
        col = action % GRID_SIZE
        print(f"  Action: Shoot at ({row}, {col})")
