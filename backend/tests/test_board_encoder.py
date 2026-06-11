"""Tests for ai/board_encoder.py - board-to-tensor encoding and decoding."""

import torch

from ai.board_encoder import decode_board, encode_board
from game.board import Board
from game.constants import BLACK, BOARD_SIZE, EMPTY, WHITE


class TestEncodeBoard:
    """Tests for encode_board function."""

    def test_initial_board_black_turn(self):
        """Initial board encoded for black's turn."""
        board = Board()
        tensor = encode_board(board, BLACK)

        assert tensor.shape == (3, BOARD_SIZE, BOARD_SIZE)
        assert tensor.dtype == torch.float32

        # Channel 0: black stones (current player = black)
        # Initial: (3,4) and (4,3) are black
        assert tensor[0][3][4] == 1.0
        assert tensor[0][4][3] == 1.0
        # White positions should be 0 in channel 0
        assert tensor[0][3][3] == 0.0
        assert tensor[0][4][4] == 0.0

        # Channel 1: white stones (opponent)
        assert tensor[1][3][3] == 1.0
        assert tensor[1][4][4] == 1.0
        # Black positions should be 0 in channel 1
        assert tensor[1][3][4] == 0.0
        assert tensor[1][4][3] == 0.0

        # Channel 2: all 1.0 for black's turn
        assert tensor[2].sum().item() == BOARD_SIZE * BOARD_SIZE

    def test_initial_board_white_turn(self):
        """Initial board encoded for white's turn."""
        board = Board()
        tensor = encode_board(board, WHITE)

        # Channel 0: white stones (current player = white)
        assert tensor[0][3][3] == 1.0
        assert tensor[0][4][4] == 1.0

        # Channel 1: black stones (opponent)
        assert tensor[1][3][4] == 1.0
        assert tensor[1][4][3] == 1.0

        # Channel 2: all 0.0 for white's turn
        assert tensor[2].sum().item() == 0.0

    def test_empty_cells_are_zero(self):
        """Empty cells should be 0 in both channel 0 and channel 1."""
        board = Board()
        tensor = encode_board(board, BLACK)

        # (0, 0) is empty
        assert tensor[0][0][0] == 0.0
        assert tensor[1][0][0] == 0.0

    def test_tensor_dtype_is_float32(self):
        """Tensor should be float32."""
        board = Board()
        tensor = encode_board(board, BLACK)
        assert tensor.dtype == torch.float32


class TestDecodeBoard:
    """Tests for decode_board function."""

    def test_round_trip_initial_board_black(self):
        """Encode then decode should give back the same board (black turn)."""
        board = Board()
        tensor = encode_board(board, BLACK)
        decoded = decode_board(tensor, BLACK)

        assert decoded.get_state() == board.get_state()

    def test_round_trip_initial_board_white(self):
        """Encode then decode should give back the same board (white turn)."""
        board = Board()
        tensor = encode_board(board, WHITE)
        decoded = decode_board(tensor, WHITE)

        assert decoded.get_state() == board.get_state()

    def test_round_trip_after_moves(self):
        """Round-trip after some moves are played."""
        board = Board()
        board.place_stone(2, 3, BLACK)
        board.place_stone(2, 2, WHITE)

        for color in [BLACK, WHITE]:
            tensor = encode_board(board, color)
            decoded = decode_board(tensor, color)
            assert decoded.get_state() == board.get_state()

    def test_round_trip_custom_board(self):
        """Round-trip with a custom board state."""
        state = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        state[0][0] = BLACK
        state[0][1] = WHITE
        state[7][7] = BLACK

        board = Board(state=state)

        tensor = encode_board(board, BLACK)
        decoded = decode_board(tensor, BLACK)
        assert decoded.get_state() == board.get_state()

    def test_decode_determines_color_from_channel2(self):
        """Decode correctly infers turn from channel 2."""
        board = Board()

        # Encode as black
        tensor_black = encode_board(board, BLACK)
        decoded_black = decode_board(tensor_black, BLACK)
        assert decoded_black.get_state() == board.get_state()

        # Encode as white
        tensor_white = encode_board(board, WHITE)
        decoded_white = decode_board(tensor_white, WHITE)
        assert decoded_white.get_state() == board.get_state()
