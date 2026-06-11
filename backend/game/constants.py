"""Game constants."""

# Board dimensions
BOARD_SIZE = 8

# Cell states
EMPTY = 0
BLACK = 1
WHITE = 2

# Directions for checking flips (row_delta, col_delta)
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
