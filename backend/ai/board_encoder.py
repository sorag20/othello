"""Board state to tensor encoder.

Provides encoding of Othello board states into 3x8x8 tensors for neural
network input, and decoding tensors back to board states.

Channels:
  - Channel 0: Current player's stones (1.0/0.0)
  - Channel 1: Opponent's stones (1.0/0.0)
  - Channel 2: Turn information (all 1.0 for black, all 0.0 for white)
"""

import numpy as np
import torch

from game.board import Board
from game.constants import BLACK, BOARD_SIZE, EMPTY, WHITE


def encode_board(board: Board, color: int) -> torch.Tensor:
    """盤面を3×8×8テンソルにエンコード

    Args:
        board: Board instance
        color: current player's color (BLACK or WHITE)

    Returns:
        float32 tensor of shape (3, 8, 8)
    """
    state = board.get_state()
    opponent = WHITE if color == BLACK else BLACK

    tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if state[r][c] == color:
                tensor[0][r][c] = 1.0
            elif state[r][c] == opponent:
                tensor[1][r][c] = 1.0

    # Channel 2: turn info (all 1.0 if black's turn, all 0.0 if white's turn)
    if color == BLACK:
        tensor[2] = 1.0

    return torch.from_numpy(tensor)


def decode_board(tensor: torch.Tensor, color: int) -> Board:
    """テンソルから盤面を復元

    Args:
        tensor: float32 tensor of shape (3, 8, 8)
        color: the current player's color that was used during encoding

    Returns:
        Board instance with the decoded state
    """
    # Convert to numpy for easier manipulation
    data = tensor.numpy()

    # Determine current player's color from channel 2
    # All 1.0 means BLACK's turn, all 0.0 means WHITE's turn
    turn_channel_mean = data[2].mean()
    if turn_channel_mean > 0.5:
        current_color = BLACK
    else:
        current_color = WHITE

    opponent = WHITE if current_color == BLACK else BLACK

    # Reconstruct board state
    state = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if data[0][r][c] > 0.5:
                state[r][c] = current_color
            elif data[1][r][c] > 0.5:
                state[r][c] = opponent

    return Board(state=state)
