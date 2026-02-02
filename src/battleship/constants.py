"""Battleship game constants for NavalNet RL training."""

# Grid
GRID_SIZE = 10
NUM_CELLS = GRID_SIZE * GRID_SIZE

# Ship configurations: (length, count) - Standard Battleship fleet
SHIP_SPECS = [
    (5, 1),  # Carrier
    (4, 1),  # Battleship
    (3, 2),  # Cruiser x2
    (2, 1),  # Destroyer
]
SHIP_SIZES = [5, 4, 3, 3, 2]  # Flat list for iteration

# Cell states for observation space (Gymnasium)
CELL_UNKNOWN = 0
CELL_MISS = 1
CELL_HIT = 2
CELL_SUNK = 3

# Orientation
HORIZONTAL = 0
VERTICAL = 1
