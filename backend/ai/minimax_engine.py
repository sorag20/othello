"""Minimax-based Othello engine using bitboards and alpha-beta pruning.

Provides MinimaxEngine, a classical search-based alternative to the
AlphaZero-style (MCTS + neural network) engine. Board state is converted
to a pair of 64-bit bitboards (own/opponent) for fast legal-move
generation and flip computation, then searched with iterative-deepening
negamax + alpha-beta pruning, a transposition table, and an exact
disc-difference solver for the endgame.
"""

import time
from typing import Dict, List, Optional, Tuple

from game.board import Board
from game.constants import BLACK, BOARD_SIZE, WHITE

_FULL_MASK = (1 << 64) - 1
_FILE_A = 0x0101010101010101
_FILE_H = 0x8080808080808080
_NOT_FILE_A = _FULL_MASK ^ _FILE_A
_NOT_FILE_H = _FULL_MASK ^ _FILE_H

# (shift amount, premask applied to source bits before shifting).
# Positive shift = left shift (<<), negative = right shift (>>).
# East/West/diagonals mask out the file that would wrap around the board
# edge; North/South need no mask since 64-bit truncation already discards
# whatever would fall off the top/bottom row.
_DIRECTIONS = (
    (1, _NOT_FILE_H),   # East
    (-1, _NOT_FILE_A),  # West
    (8, _FULL_MASK),    # South
    (-8, _FULL_MASK),   # North
    (9, _NOT_FILE_H),   # South-East
    (7, _NOT_FILE_A),   # South-West
    (-7, _NOT_FILE_H),  # North-East
    (-9, _NOT_FILE_A),  # North-West
)

_ROW0 = 0x00000000000000FF
_ROW7 = 0xFF00000000000000

# Stability is computed per axis (a pair of opposite directions). A disc has
# an unbroken same-color chain to the edge in a given direction if either it
# IS that edge already, or its backward neighbor (one step closer to the
# disc, i.e. the opposite direction) is same-colored and itself has such a
# chain. For each direction we store (edge_mask, back_amount, back_premask):
# `edge_mask` seeds the fixed point (squares already at that edge), and
# `back_amount`/`back_premask` extend the chain one square inward using the
# OPPOSITE direction's own shift/premask (see _chain_to_edge).
_STABILITY_AXES = (
    (  # horizontal: East, West
        (_FILE_H, -1, _NOT_FILE_A),
        (_FILE_A, 1, _NOT_FILE_H),
    ),
    (  # vertical: South, North
        (_ROW7, -8, _FULL_MASK),
        (_ROW0, 8, _FULL_MASK),
    ),
    (  # diagonal "\": South-East, North-West
        (_FILE_H | _ROW7, -9, _NOT_FILE_A),
        (_FILE_A | _ROW0, 9, _NOT_FILE_H),
    ),
    (  # diagonal "/": South-West, North-East
        (_FILE_A | _ROW7, -7, _NOT_FILE_H),
        (_FILE_H | _ROW0, 7, _NOT_FILE_A),
    ),
)

# Classic Othello static positional weight table (row-major, row 0 = top).
_WEIGHTS = (
    100, -20, 10, 5, 5, 10, -20, 100,
    -20, -50, -2, -2, -2, -2, -50, -20,
    10, -2, 1, 1, 1, 1, -2, 10,
    5, -2, 1, 1, 1, 1, -2, 5,
    5, -2, 1, 1, 1, 1, -2, 5,
    10, -2, 1, 1, 1, 1, -2, 10,
    -20, -50, -2, -2, -2, -2, -50, -20,
    100, -20, 10, 5, 5, 10, -20, 100,
)

# Below this many empty squares, switch to an exact disc-difference solver.
_ENDGAME_EMPTY_THRESHOLD = 10

# Large sentinel bounding the search window; must comfortably exceed any
# possible heuristic or terminal score magnitude produced below.
_INF = 1_000_000

_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2


def _shift(bits: int, amount: int, premask: int) -> int:
    """Shift bits by amount, masking wraparound edges before shifting."""
    bits &= premask
    if amount > 0:
        return (bits << amount) & _FULL_MASK
    return bits >> (-amount)


def _get_moves(own: int, opp: int) -> int:
    """Return a bitboard of all legal move squares for `own` against `opp`."""
    empty = _FULL_MASK ^ (own | opp)
    moves = 0
    for amount, premask in _DIRECTIONS:
        candidates = _shift(own, amount, premask) & opp
        for _ in range(5):
            candidates |= _shift(candidates, amount, premask) & opp
        moves |= _shift(candidates, amount, premask) & empty
    return moves


def _flip_mask(own: int, opp: int, move_bit: int) -> int:
    """Return the bitboard of opponent stones flipped by placing at move_bit."""
    flips = 0
    for amount, premask in _DIRECTIONS:
        candidates = _shift(move_bit, amount, premask) & opp
        for _ in range(5):
            candidates |= _shift(candidates, amount, premask) & opp
        if _shift(candidates, amount, premask) & own:
            flips |= candidates
    return flips


def _stable_discs(color_bits: int) -> int:
    """Fully stable discs of `color_bits`: can never be flipped again.

    A disc is stable once, for each of the 4 axes through it (horizontal,
    vertical, and both diagonals), at least one of the two directions either
    reaches the board edge directly, or its immediate neighbor in that
    direction is same-colored AND is itself already (fully, all 4 axes)
    stable. A disc must clear all 4 axes to be considered stable overall.

    This must be one coupled fixed point over the whole stable set, not 4
    independent per-axis chains: an "anchor" disc further down a line is
    only a permanent anchor if it is itself immune to being flipped by some
    OTHER axis. Treating "same color" alone as good enough (ignoring
    whether that neighbor could later flip via a different axis) silently
    permits chains to lean on a neighbor that isn't actually permanent.
    """
    # Trivial seed: discs where every axis already touches its edge
    # directly in at least one direction (no neighbor to vouch for).
    stable = color_bits
    for edge_a, edge_b in ((axis[0][0], axis[1][0]) for axis in _STABILITY_AXES):
        stable &= edge_a | edge_b

    while True:
        axis_ok_all = _FULL_MASK
        for (edge_a, back_amount_a, back_premask_a), (edge_b, back_amount_b, back_premask_b) in _STABILITY_AXES:
            reach_a = _shift(stable, back_amount_a, back_premask_a)
            reach_b = _shift(stable, back_amount_b, back_premask_b)
            axis_ok_all &= edge_a | edge_b | reach_a | reach_b
        grown = color_bits & axis_ok_all
        if grown == stable:
            return stable
        stable = grown


def _frontier_discs(color_bits: int, empty: int) -> int:
    """Discs in `color_bits` adjacent to at least one empty square.

    Frontier discs are more exposed (closer to being flippable soon), so
    fewer of them is generally better -- a cheap, well-known proxy for
    "potential mobility" / positional safety.
    """
    adjacent_to_empty = 0
    for amount, premask in _DIRECTIONS:
        adjacent_to_empty |= _shift(empty, amount, premask)
    return color_bits & adjacent_to_empty


def _apply_move(own: int, opp: int, move_bit: int) -> Tuple[int, int]:
    """Apply a move for `own`, returning the new (own, opp) bitboards."""
    flips = _flip_mask(own, opp, move_bit)
    new_own = own | move_bit | flips
    new_opp = opp & ~flips & _FULL_MASK
    return new_own, new_opp


def _iter_bits(bits: int):
    """Yield each set bit of `bits` as a single-bit mask."""
    while bits:
        lsb = bits & (-bits)
        yield lsb
        bits ^= lsb


def _popcount(bits: int) -> int:
    return bin(bits).count("1")


def board_to_bitboards(state: List[List[int]], color: int) -> Tuple[int, int]:
    """Convert a Board 2D state into (own, opp) bitboards for `color`."""
    opponent = WHITE if color == BLACK else BLACK
    own = 0
    opp = 0
    for row in range(BOARD_SIZE):
        row_state = state[row]
        base = row * BOARD_SIZE
        for col in range(BOARD_SIZE):
            cell = row_state[col]
            if cell == color:
                own |= 1 << (base + col)
            elif cell == opponent:
                opp |= 1 << (base + col)
    return own, opp


def bit_to_rc(bit: int) -> Tuple[int, int]:
    """Convert a single-bit bitboard mask to a (row, col) tuple."""
    index = bit.bit_length() - 1
    return index // BOARD_SIZE, index % BOARD_SIZE


def rc_to_bit(row: int, col: int) -> int:
    """Convert a (row, col) tuple to a single-bit bitboard mask."""
    return 1 << (row * BOARD_SIZE + col)


class MinimaxEngine:
    """Alpha-beta minimax Othello engine (bitboard-based).

    Runs iterative-deepening negamax with alpha-beta pruning and a
    transposition table for the midgame, switching to an exact
    disc-difference solver once few empty squares remain.
    """

    def __init__(
        self,
        time_limit: float = 3.0,
        endgame_empty_threshold: int = _ENDGAME_EMPTY_THRESHOLD,
        max_depth: int = 60,
    ):
        """Configure search limits.

        Args:
            time_limit: Wall-clock search budget per move, in seconds.
            endgame_empty_threshold: Empty-square count at or below which
                the engine switches to the exact endgame solver.
            max_depth: Hard cap on iterative-deepening depth (safety net;
                the board only has 64 squares so this is rarely reached).
        """
        self._time_limit = time_limit
        self._endgame_empty_threshold = endgame_empty_threshold
        self._max_depth = max_depth
        self._deadline = 0.0
        # key: (own, opp) -> (depth, flag, value, best_move)
        self._tt: Dict[Tuple[int, int], Tuple[int, int, int, Optional[int]]] = {}
        # key: (own, opp) -> (flag, value, best_move); exact, so no depth needed.
        self._endgame_tt: Dict[Tuple[int, int], Tuple[int, int, Optional[int]]] = {}
        # key: remaining depth -> up to 2 moves that recently caused a beta
        # cutoff at that depth (killer-move heuristic for move ordering).
        self._killers: Dict[int, List[int]] = {}

    def compute_move(self, board: Board, color: int) -> Tuple[int, int]:
        """Compute the best move for `color` on `board` within the time budget.

        Args:
            board: Current board state.
            color: Color to move (BLACK=1 or WHITE=2).

        Returns:
            (row, col) of the chosen move.

        Raises:
            InvalidBoardStateError: If no legal moves exist.
        """
        legal_moves = board.get_legal_moves(color)
        if not legal_moves:
            raise InvalidBoardStateError(
                "合法手が存在しない盤面でcompute_moveが呼ばれました"
            )
        if len(legal_moves) == 1:
            return legal_moves[0]

        own, opp = board_to_bitboards(board.get_state(), color)
        empty_count = BOARD_SIZE * BOARD_SIZE - _popcount(own | opp)

        self._deadline = time.monotonic() + self._time_limit
        self._tt.clear()
        self._endgame_tt.clear()
        self._killers.clear()

        if empty_count <= self._endgame_empty_threshold:
            move_bit = self._solve_endgame(own, opp)
        else:
            move_bit = self._search_iterative_deepening(own, opp, empty_count)

        if move_bit is None:
            return legal_moves[0]

        move = bit_to_rc(move_bit)
        if move not in legal_moves:
            return legal_moves[0]
        return move

    # -- move ordering (shared by midgame and endgame search) ---------------------

    def _order_moves(
        self,
        own: int,
        opp: int,
        moves: List[int],
        tt_move: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> List[int]:
        if len(moves) <= 1:
            return moves

        killers = self._killers.get(depth, ()) if depth is not None else ()

        def sort_key(move: int) -> int:
            if move == tt_move:
                return -_INF
            if move in killers:
                return -_INF + 1 + killers.index(move)
            new_own, new_opp = _apply_move(own, opp, move)
            # Prefer moves that leave the opponent with fewer replies (a
            # cheap but effective classic move-ordering heuristic).
            return _popcount(_get_moves(new_opp, new_own))

        return sorted(moves, key=sort_key)

    def _record_killer(self, depth: int, move: int) -> None:
        """Remember a move that caused a beta cutoff at this remaining depth."""
        killers = self._killers.setdefault(depth, [])
        if move in killers:
            return
        killers.insert(0, move)
        del killers[2:]

    # -- midgame heuristic search ---------------------------------------------------

    def _search_iterative_deepening(self, own: int, opp: int, empty_count: int) -> Optional[int]:
        best_move = None
        tt_move_hint = None
        max_depth = min(self._max_depth, empty_count)
        for depth in range(1, max_depth + 1):
            if time.monotonic() >= self._deadline:
                break
            try:
                _, move = self._negamax_root(own, opp, depth, tt_move_hint)
            except _TimeUp:
                break
            if move is not None:
                best_move = move
                tt_move_hint = move
            if time.monotonic() >= self._deadline:
                break
        return best_move

    def _negamax_root(
        self, own: int, opp: int, depth: int, tt_move_hint: Optional[int]
    ) -> Tuple[int, Optional[int]]:
        moves = self._order_moves(
            own, opp, list(_iter_bits(_get_moves(own, opp))), tt_move_hint, depth
        )

        best_value = -_INF
        best_move = moves[0]
        alpha, beta = -_INF, _INF
        for i, move in enumerate(moves):
            new_own, new_opp = _apply_move(own, opp, move)
            if i == 0:
                value = -self._negamax(new_opp, new_own, depth - 1, -beta, -alpha)
            else:
                # NegaScout/PVS: cheaply verify the move isn't better than
                # alpha with a null (zero-width) window first; only pay for a
                # full-window re-search if it actually looks like it might be.
                value = -self._negamax(new_opp, new_own, depth - 1, -alpha - 1, -alpha)
                if alpha < value < beta:
                    value = -self._negamax(new_opp, new_own, depth - 1, -beta, -value)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
        return best_value, best_move

    def _negamax(self, own: int, opp: int, depth: int, alpha: int, beta: int) -> int:
        if time.monotonic() >= self._deadline:
            raise _TimeUp()

        own_moves = _get_moves(own, opp)
        if depth <= 0:
            return self._evaluate(own, opp, own_moves)

        if own_moves == 0:
            opp_moves = _get_moves(opp, own)
            if opp_moves == 0:
                return self._final_score(own, opp)
            # Pass: opponent moves again with roles swapped.
            return -self._negamax(opp, own, depth - 1, -beta, -alpha)

        key = (own, opp)
        tt_entry = self._tt.get(key)
        tt_move = None
        if tt_entry is not None:
            tt_depth, tt_flag, tt_value, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == _TT_EXACT:
                    return tt_value
                if tt_flag == _TT_LOWER:
                    alpha = max(alpha, tt_value)
                elif tt_flag == _TT_UPPER:
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    return tt_value

        moves = self._order_moves(own, opp, list(_iter_bits(own_moves)), tt_move, depth)
        original_alpha = alpha
        best_value = -_INF
        best_move = moves[0]
        for i, move in enumerate(moves):
            new_own, new_opp = _apply_move(own, opp, move)
            if i == 0:
                value = -self._negamax(new_opp, new_own, depth - 1, -beta, -alpha)
            else:
                value = -self._negamax(new_opp, new_own, depth - 1, -alpha - 1, -alpha)
                if alpha < value < beta:
                    value = -self._negamax(new_opp, new_own, depth - 1, -beta, -value)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                self._record_killer(depth, move)
                break

        flag = _TT_EXACT
        if best_value <= original_alpha:
            flag = _TT_UPPER
        elif best_value >= beta:
            flag = _TT_LOWER

        existing = self._tt.get(key)
        if existing is None or existing[0] <= depth:
            self._tt[key] = (depth, flag, best_value, best_move)
        return best_value

    def _evaluate(self, own: int, opp: int, own_moves: int) -> int:
        """Phase-aware heuristic evaluation from `own`'s perspective."""
        opp_moves = _get_moves(opp, own)
        own_move_count = _popcount(own_moves)
        opp_move_count = _popcount(opp_moves)

        empty = _FULL_MASK ^ (own | opp)
        empty_count = _popcount(empty)
        own_discs = _popcount(own)
        opp_discs = _popcount(opp)

        static_score = sum(
            _WEIGHTS[i] for i in range(64) if (own >> i) & 1
        ) - sum(
            _WEIGHTS[i] for i in range(64) if (opp >> i) & 1
        )

        mobility_total = own_move_count + opp_move_count
        mobility_score = (
            100 * (own_move_count - opp_move_count) / mobility_total
            if mobility_total > 0
            else 0
        )

        # Stable discs (can never be flipped again) are a far more accurate
        # safety signal than the static corner-adjacency proxy alone.
        own_stable = _popcount(_stable_discs(own))
        opp_stable = _popcount(_stable_discs(opp))
        stability_score = own_stable - opp_stable

        # Frontier discs (adjacent to an empty square) are more exposed;
        # fewer of our own is better (a cheap potential-mobility proxy).
        own_frontier = _popcount(_frontier_discs(own, empty))
        opp_frontier = _popcount(_frontier_discs(opp, empty))
        frontier_score = opp_frontier - own_frontier

        parity_weight = max(0, 30 - empty_count)
        parity_score = parity_weight * (own_discs - opp_discs)

        return int(
            static_score
            + 1.5 * mobility_score
            + 20 * stability_score
            + 2 * frontier_score
            + parity_score
        )

    def _final_score(self, own: int, opp: int) -> int:
        """Exact terminal score: disc-difference amplified to dominate heuristics."""
        diff = _popcount(own) - _popcount(opp)
        return diff * 1_000

    # -- exact endgame solver ------------------------------------------------------

    def _solve_endgame(self, own: int, opp: int) -> Optional[int]:
        own_moves = _get_moves(own, opp)
        moves = list(_iter_bits(own_moves))
        if not moves:
            return None

        tt_entry = self._endgame_tt.get((own, opp))
        tt_move = tt_entry[2] if tt_entry is not None else None
        moves = self._order_moves(own, opp, moves, tt_move)

        best_value = -65
        best_move = moves[0]
        alpha, beta = -65, 65
        for i, move in enumerate(moves):
            new_own, new_opp = _apply_move(own, opp, move)
            try:
                if i == 0:
                    value = -self._solve(new_opp, new_own, -beta, -alpha)
                else:
                    value = -self._solve(new_opp, new_own, -alpha - 1, -alpha)
                    if alpha < value < beta:
                        value = -self._solve(new_opp, new_own, -beta, -value)
            except _TimeUp:
                return best_move
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
        return best_move

    def _solve(self, own: int, opp: int, alpha: int, beta: int) -> int:
        if time.monotonic() >= self._deadline:
            raise _TimeUp()

        own_moves = _get_moves(own, opp)
        if own_moves == 0:
            opp_moves = _get_moves(opp, own)
            if opp_moves == 0:
                return _popcount(own) - _popcount(opp)
            return -self._solve(opp, own, -beta, -alpha)

        key = (own, opp)
        tt_entry = self._endgame_tt.get(key)
        tt_move = None
        if tt_entry is not None:
            tt_flag, tt_value, tt_move = tt_entry
            if tt_flag == _TT_EXACT:
                return tt_value
            if tt_flag == _TT_LOWER:
                alpha = max(alpha, tt_value)
            elif tt_flag == _TT_UPPER:
                beta = min(beta, tt_value)
            if alpha >= beta:
                return tt_value

        moves = self._order_moves(own, opp, list(_iter_bits(own_moves)), tt_move)
        original_alpha = alpha
        best_value = -65
        best_move = moves[0]
        for i, move in enumerate(moves):
            new_own, new_opp = _apply_move(own, opp, move)
            if i == 0:
                value = -self._solve(new_opp, new_own, -beta, -alpha)
            else:
                value = -self._solve(new_opp, new_own, -alpha - 1, -alpha)
                if alpha < value < beta:
                    value = -self._solve(new_opp, new_own, -beta, -value)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        flag = _TT_EXACT
        if best_value <= original_alpha:
            flag = _TT_UPPER
        elif best_value >= beta:
            flag = _TT_LOWER
        self._endgame_tt[key] = (flag, best_value, best_move)
        return best_value


class _TimeUp(Exception):
    """Internal signal that the search deadline was reached mid-search."""


class InvalidBoardStateError(Exception):
    """無効な盤面状態の入力（合法手が存在しない）"""
