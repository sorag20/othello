"""Tests for board state and logic.

Unit tests for the Board class including initial state, stone placement,
legal move calculation, stone counting, and copying.
"""

import pytest

from game.board import Board
from game.constants import BLACK, BOARD_SIZE, EMPTY, WHITE


class TestBoardInitialization:
    """Test Board initialization."""

    def test_default_initialization_creates_initial_board(self):
        """Default Board() creates the standard Othello starting position."""
        board = Board()
        state = board.get_state()

        # Center stones are placed correctly
        assert state[3][3] == WHITE
        assert state[3][4] == BLACK
        assert state[4][3] == BLACK
        assert state[4][4] == WHITE

    def test_default_initialization_empty_cells(self):
        """All non-center cells are empty on initial board."""
        board = Board()
        state = board.get_state()

        empty_count = sum(
            1
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if state[r][c] == EMPTY
        )
        assert empty_count == 60  # 64 - 4 initial stones

    def test_custom_state_initialization(self):
        """Board can be initialized with a custom state."""
        custom_state = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        custom_state[0][0] = BLACK
        custom_state[7][7] = WHITE

        board = Board(state=custom_state)
        state = board.get_state()

        assert state[0][0] == BLACK
        assert state[7][7] == WHITE
        assert state[3][3] == EMPTY  # No default initialization

    def test_custom_state_is_copied(self):
        """Modifying the original state list doesn't affect the board."""
        custom_state = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        custom_state[0][0] = BLACK

        board = Board(state=custom_state)
        custom_state[0][0] = WHITE  # Modify original

        assert board.get_state()[0][0] == BLACK  # Board unaffected


class TestGetState:
    """Test Board.get_state method."""

    def test_get_state_returns_copy(self):
        """get_state returns a copy, not a reference."""
        board = Board()
        state = board.get_state()
        state[0][0] = BLACK  # Modify the returned copy

        assert board.get_state()[0][0] == EMPTY  # Board unaffected


class TestPlaceStone:
    """Test Board.place_stone method."""

    def test_place_stone_on_initial_board(self):
        """Black can place at (2,3) on initial board, flipping white at (3,3)."""
        board = Board()
        flipped = board.place_stone(2, 3, BLACK)

        assert (3, 3) in flipped
        assert len(flipped) == 1
        assert board.get_state()[2][3] == BLACK
        assert board.get_state()[3][3] == BLACK  # Flipped from white

    def test_place_stone_returns_empty_for_occupied_cell(self):
        """Cannot place on an occupied cell."""
        board = Board()
        flipped = board.place_stone(3, 3, BLACK)

        assert flipped == []

    def test_place_stone_returns_empty_for_no_flips(self):
        """Cannot place where no flips would occur."""
        board = Board()
        flipped = board.place_stone(0, 0, BLACK)

        assert flipped == []
        assert board.get_state()[0][0] == EMPTY  # State unchanged

    def test_place_stone_out_of_bounds(self):
        """Out of bounds placement returns empty list."""
        board = Board()
        assert board.place_stone(-1, 0, BLACK) == []
        assert board.place_stone(0, 8, BLACK) == []
        assert board.place_stone(8, 8, BLACK) == []

    def test_place_stone_multiple_directions(self):
        """Place a stone that flips in multiple directions."""
        # Set up a board where placing at (2,2) as BLACK flips in multiple directions
        state = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        # Black stone at (2,4) - to create a horizontal flip
        state[2][4] = BLACK
        # White stones between
        state[2][3] = WHITE
        # Black stone at (4,2) - vertical flip
        state[4][2] = BLACK
        state[3][2] = WHITE
        # Black stone at (4,4) - diagonal flip
        state[3][3] = WHITE
        state[4][4] = BLACK

        board = Board(state=state)
        flipped = board.place_stone(2, 2, BLACK)

        assert (2, 3) in flipped  # horizontal
        assert (3, 2) in flipped  # vertical
        assert (3, 3) in flipped  # diagonal
        assert len(flipped) == 3


class TestGetLegalMoves:
    """Test Board.get_legal_moves method."""

    def test_initial_board_black_has_4_legal_moves(self):
        """Black has exactly 4 legal moves on the initial board."""
        board = Board()
        moves = board.get_legal_moves(BLACK)

        assert len(moves) == 4
        expected = [(2, 3), (3, 2), (4, 5), (5, 4)]
        assert sorted(moves) == sorted(expected)

    def test_initial_board_white_has_4_legal_moves(self):
        """White has exactly 4 legal moves on the initial board."""
        board = Board()
        moves = board.get_legal_moves(WHITE)

        assert len(moves) == 4
        expected = [(2, 4), (3, 5), (4, 2), (5, 3)]
        assert sorted(moves) == sorted(expected)

    def test_no_legal_moves_on_empty_board(self):
        """No legal moves when the board is empty."""
        state = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board = Board(state=state)

        assert board.get_legal_moves(BLACK) == []
        assert board.get_legal_moves(WHITE) == []


class TestCountStones:
    """Test Board.count_stones method."""

    def test_initial_board_count(self):
        """Initial board has 2 black and 2 white stones."""
        board = Board()
        count = board.count_stones()

        assert count == {"black": 2, "white": 2}

    def test_count_after_move(self):
        """After black places at (2,3), counts update correctly."""
        board = Board()
        board.place_stone(2, 3, BLACK)
        count = board.count_stones()

        # 2 original black + 1 placed + 1 flipped = 4 black, 2-1 = 1 white
        assert count == {"black": 4, "white": 1}


class TestIsFull:
    """Test Board.is_full method."""

    def test_initial_board_not_full(self):
        """Initial board is not full."""
        board = Board()
        assert board.is_full() is False

    def test_full_board(self):
        """A completely filled board returns True."""
        state = [[BLACK] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board = Board(state=state)
        assert board.is_full() is True

    def test_one_empty_cell_not_full(self):
        """A board with one empty cell is not full."""
        state = [[BLACK] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        state[7][7] = EMPTY
        board = Board(state=state)
        assert board.is_full() is False


class TestCopy:
    """Test Board.copy method."""

    def test_copy_creates_independent_board(self):
        """copy() creates a new board independent of the original."""
        board = Board()
        board_copy = board.copy()

        # Place a stone on the copy
        board_copy.place_stone(2, 3, BLACK)

        # Original should be unchanged
        assert board.get_state()[2][3] == EMPTY
        assert board.get_state()[3][3] == WHITE  # Not flipped on original

    def test_copy_preserves_state(self):
        """copy() preserves the current board state."""
        board = Board()
        board.place_stone(2, 3, BLACK)

        board_copy = board.copy()

        assert board_copy.get_state() == board.get_state()
