"""Tests for the bitboard minimax engine.

Cross-checks bitboard move generation and flip computation against the
already-trusted Board/OthelloRules implementation across many random
self-play games, then tests MinimaxEngine's public behavior (legal
move selection, time budget, endgame solver correctness).
"""

import random
import time

import pytest

from ai.minimax_engine import (
    InvalidBoardStateError,
    MinimaxEngine,
    _apply_move,
    _frontier_discs,
    _get_moves,
    _popcount,
    _stable_discs,
    bit_to_rc,
    board_to_bitboards,
    rc_to_bit,
)
from game.board import Board
from game.constants import BLACK, WHITE


def _play_random_game(seed: int):
    """Play one full random-legal-move game, yielding (board, color) each ply."""
    rng = random.Random(seed)
    board = Board()
    color = BLACK
    passes_in_a_row = 0

    while passes_in_a_row < 2:
        legal_moves = board.get_legal_moves(color)
        if not legal_moves:
            passes_in_a_row += 1
            color = WHITE if color == BLACK else BLACK
            continue

        passes_in_a_row = 0
        yield board, color, legal_moves

        row, col = rng.choice(legal_moves)
        board.place_stone(row, col, color)
        color = WHITE if color == BLACK else BLACK


class TestBitboardConversion:
    """board_to_bitboards / bit_to_rc / rc_to_bit のテスト"""

    def test_round_trip_initial_board(self):
        board = Board()
        state = board.get_state()

        own, opp = board_to_bitboards(state, BLACK)
        assert bin(own).count("1") == 2
        assert bin(opp).count("1") == 2

        for row in range(8):
            for col in range(8):
                bit = rc_to_bit(row, col)
                if state[row][col] == BLACK:
                    assert own & bit
                elif state[row][col] == WHITE:
                    assert opp & bit
                else:
                    assert not (own & bit) and not (opp & bit)

    def test_bit_to_rc_and_back(self):
        for row in range(8):
            for col in range(8):
                bit = rc_to_bit(row, col)
                assert bit_to_rc(bit) == (row, col)


class TestMoveGenerationMatchesBoard:
    """ビットボードのMove生成/反転計算がBoard実装と一致することを確認"""

    def test_legal_moves_match_across_random_games(self):
        mismatches = []
        for seed in range(30):
            for board, color, expected_moves in _play_random_game(seed):
                own, opp = board_to_bitboards(board.get_state(), color)
                bitboard_moves = {
                    bit_to_rc(bit) for bit in _iter_bits(_get_moves(own, opp))
                }
                if bitboard_moves != set(expected_moves):
                    mismatches.append((seed, color, expected_moves, bitboard_moves))

        assert not mismatches, f"Move generation mismatches: {mismatches[:3]}"

    def test_flip_results_match_across_random_games(self):
        mismatches = []
        for seed in range(30):
            for board, color, legal_moves in _play_random_game(seed):
                own, opp = board_to_bitboards(board.get_state(), color)
                for row, col in legal_moves:
                    expected_flipped = set(
                        board.copy().place_stone(row, col, color)
                    )
                    move_bit = rc_to_bit(row, col)
                    new_own, new_opp = _apply_move(own, opp, move_bit)
                    flipped_bits = new_own & ~own & ~move_bit & ((1 << 64) - 1)
                    actual_flipped = {bit_to_rc(b) for b in _iter_bits(flipped_bits)}
                    if actual_flipped != expected_flipped:
                        mismatches.append(
                            (seed, color, (row, col), expected_flipped, actual_flipped)
                        )

        assert not mismatches, f"Flip mismatches: {mismatches[:3]}"


def _iter_bits(bits):
    while bits:
        lsb = bits & (-bits)
        yield lsb
        bits ^= lsb


class _TooBig(Exception):
    """Raised to abandon a brute-force check once its game tree is too large."""


def _enumerate_continuations_check(own, opp, turn_is_own, stable_own, stable_opp, counter, node_budget):
    """Walk every possible continuation from (own, opp), failing if a disc
    marked stable in `stable_own`/`stable_opp` ever changes color at a leaf.
    """
    counter[0] += 1
    if counter[0] > node_budget:
        raise _TooBig()

    moves = _get_moves(own, opp) if turn_is_own else _get_moves(opp, own)
    if moves == 0:
        other_moves = _get_moves(opp, own) if turn_is_own else _get_moves(own, opp)
        if other_moves == 0:
            assert not (stable_own & opp), "a disc marked stable for 'own' was flipped to 'opp'"
            assert not (stable_opp & own), "a disc marked stable for 'opp' was flipped to 'own'"
            return
        _enumerate_continuations_check(
            own, opp, not turn_is_own, stable_own, stable_opp, counter, node_budget
        )
        return

    for move in _iter_bits(moves):
        if turn_is_own:
            new_own, new_opp = _apply_move(own, opp, move)
        else:
            new_opp, new_own = _apply_move(opp, own, move)
        _enumerate_continuations_check(
            new_own, new_opp, not turn_is_own, stable_own, stable_opp, counter, node_budget
        )


def _random_position_with_empties(seed, target_empty):
    """Play random legal moves until at most `target_empty` squares remain,
    returning bitboards relative to whoever actually moves next."""
    rng = random.Random(seed)
    board = Board()
    color = BLACK
    passes_in_a_row = 0
    while passes_in_a_row < 2:
        empties = sum(row.count(0) for row in board.get_state())
        if empties <= target_empty:
            break
        legal_moves = board.get_legal_moves(color)
        if not legal_moves:
            passes_in_a_row += 1
            color = WHITE if color == BLACK else BLACK
            continue
        passes_in_a_row = 0
        row, col = rng.choice(legal_moves)
        board.place_stone(row, col, color)
        color = WHITE if color == BLACK else BLACK
    own, opp = board_to_bitboards(board.get_state(), color)
    return own, opp


class TestStableDiscs:
    """_stable_discs のテスト: 「安定石」と判定された石が、以降どんな手順でも
    絶対にひっくり返されないことを全探索で検証する(過大判定=危険なバグ)。
    """

    def test_corner_anchored_run_is_stable(self):
        # a1, b1, c1 (row0, col0-2): anchored to the top-left corner and the
        # top edge, so all three should be unconditionally stable.
        own = 0b111
        stable = _stable_discs(own)
        assert stable == 0b111

    def test_isolated_disc_is_not_stable(self):
        # A single disc in the middle of an otherwise empty board touches no
        # edge and has no stable neighbor, so it must not be marked stable.
        own = rc_to_bit(3, 3)
        assert _stable_discs(own) == 0

    def test_marked_stable_discs_never_flip_in_any_continuation(self):
        """Exhaustively verify (not just sample) that every disc _stable_discs
        marks stable for either side stays that color through every possible
        sequence of future legal moves, for many small random endgame
        positions. This is the critical soundness check: a false positive
        here would make the evaluation function trust a disc that can
        actually still be captured.
        """
        tested = 0
        skipped = 0
        for seed in range(120):
            own, opp = _random_position_with_empties(seed, target_empty=8)
            if _popcount(own | opp) == 64:
                continue

            stable_own = _stable_discs(own)
            stable_opp = _stable_discs(opp)

            counter = [0]
            try:
                _enumerate_continuations_check(
                    own, opp, True, stable_own, stable_opp, counter, node_budget=40000
                )
            except _TooBig:
                skipped += 1
                continue
            tested += 1

        # Make sure the test is actually exercising a meaningful number of
        # positions and not silently skipping everything.
        assert tested >= 60, f"only {tested} positions were small enough to check exhaustively"

    def test_stable_discs_is_subset_of_color_bits(self):
        for seed in range(20):
            own, opp = _random_position_with_empties(seed, target_empty=20)
            assert _stable_discs(own) & ~own == 0
            assert _stable_discs(opp) & ~opp == 0


class TestFrontierDiscs:
    """_frontier_discs のテスト"""

    def test_no_frontier_when_board_full(self):
        own = (1 << 64) - 1
        assert _frontier_discs(own, empty=0) == 0

    def test_disc_adjacent_to_empty_is_frontier(self):
        own = rc_to_bit(3, 3)
        empty = rc_to_bit(3, 4)  # directly east of the disc
        assert _frontier_discs(own, empty) == own

    def test_disc_not_adjacent_to_empty_is_not_frontier(self):
        own = rc_to_bit(0, 0)
        empty = rc_to_bit(7, 7)  # far away, not adjacent
        assert _frontier_discs(own, empty) == 0


class TestMinimaxEngineComputeMove:
    """MinimaxEngine.compute_move のテスト"""

    def test_returns_legal_move_on_initial_board(self):
        engine = MinimaxEngine(time_limit=0.5)
        board = Board()

        move = engine.compute_move(board, BLACK)

        assert move in board.get_legal_moves(BLACK)

    def test_raises_when_no_legal_moves(self):
        engine = MinimaxEngine(time_limit=0.5)
        # Fabricate a board state with no legal moves for either color.
        full_state = [[BLACK] * 8 for _ in range(8)]
        board = Board(state=full_state)

        with pytest.raises(InvalidBoardStateError):
            engine.compute_move(board, WHITE)

    def test_single_legal_move_returned_immediately(self):
        engine = MinimaxEngine(time_limit=5.0)
        board = Board()
        legal_moves = board.get_legal_moves(BLACK)
        board.place_stone(*legal_moves[0], BLACK)

        white_moves = board.get_legal_moves(WHITE)

        start = time.monotonic()
        move = engine.compute_move(board, WHITE)
        elapsed = time.monotonic() - start

        assert move in white_moves
        if len(white_moves) == 1:
            # Should short-circuit well under the 5s budget.
            assert elapsed < 1.0

    def test_respects_time_budget(self):
        engine = MinimaxEngine(time_limit=0.3)
        board = Board()

        start = time.monotonic()
        move = engine.compute_move(board, BLACK)
        elapsed = time.monotonic() - start

        assert move in board.get_legal_moves(BLACK)
        # Generous slack for scheduling jitter; search must not run away.
        assert elapsed < 2.0

    def test_deterministic_depth_one_prefers_corner(self):
        # Construct a position where BLACK can either take a corner or a
        # mediocre square; even a shallow search should take the corner.
        state = [[0] * 8 for _ in range(8)]
        state[0][1] = WHITE
        state[0][2] = WHITE
        state[0][3] = BLACK
        board = Board(state=state)

        engine = MinimaxEngine(time_limit=1.0)
        move = engine.compute_move(board, BLACK)

        assert move == (0, 0)


class TestMinimaxEngineEndgameSolver:
    """終盤の完全読みソルバーが最適手を返すことを確認"""

    def test_matches_brute_force_on_near_full_board(self):
        rng = random.Random(42)
        board = Board()
        color = BLACK
        passes_in_a_row = 0

        # Play random moves until 8 empty squares remain (small enough to
        # brute-force exhaustively but non-trivial).
        while passes_in_a_row < 2:
            empties = sum(row.count(0) for row in board.get_state())
            if empties <= 8:
                break
            legal_moves = board.get_legal_moves(color)
            if not legal_moves:
                passes_in_a_row += 1
                color = WHITE if color == BLACK else BLACK
                continue
            passes_in_a_row = 0
            row, col = rng.choice(legal_moves)
            board.place_stone(row, col, color)
            color = WHITE if color == BLACK else BLACK

        legal_moves = board.get_legal_moves(color)
        if not legal_moves:
            pytest.skip("Random game ended before reaching a small endgame position")

        # Reference: exact final disc-difference for each candidate move via
        # a straightforward (unpruned) minimax over the small remaining tree.
        def exact_value(b: Board, turn: int, root_color: int) -> int:
            moves = b.get_legal_moves(turn)
            if not moves:
                other = WHITE if turn == BLACK else BLACK
                if not b.get_legal_moves(other):
                    counts = b.count_stones()
                    diff = counts["black"] - counts["white"]
                    return diff if root_color == BLACK else -diff
                return exact_value(b, other, root_color)

            values = []
            for row, col in moves:
                child = b.copy()
                child.place_stone(row, col, turn)
                values.append(exact_value(child, WHITE if turn == BLACK else BLACK, root_color))
            return max(values) if turn == root_color else min(values)

        best_reference_value = None
        reference_best_moves = []
        for row, col in legal_moves:
            child = board.copy()
            child.place_stone(row, col, color)
            value = exact_value(child, WHITE if color == BLACK else BLACK, color)
            if best_reference_value is None or value > best_reference_value:
                best_reference_value = value
                reference_best_moves = [(row, col)]
            elif value == best_reference_value:
                reference_best_moves.append((row, col))

        engine = MinimaxEngine(time_limit=5.0, endgame_empty_threshold=10)
        chosen = engine.compute_move(board, color)

        assert chosen in reference_best_moves, (
            f"Engine chose {chosen} with non-optimal outcome; "
            f"optimal moves were {reference_best_moves} (value={best_reference_value})"
        )
