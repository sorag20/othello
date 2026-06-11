"""Board state and logic.

Provides the Board class for managing the 8x8 Othello board state,
including initial setup, stone placement, legal move calculation,
stone counting, and deep copying.
"""

import copy
from typing import Dict, List, Optional, Tuple

from game.constants import BLACK, BOARD_SIZE, DIRECTIONS, EMPTY, WHITE


class Board:
    """8×8オセロ盤の状態管理"""

    EMPTY = EMPTY
    BLACK = BLACK
    WHITE = WHITE

    def __init__(self, state: Optional[List[List[int]]] = None):
        """初期盤面またはstate指定で生成

        Args:
            state: 8×8の2次元リスト。Noneの場合は初期盤面を生成。
        """
        if state is not None:
            self._state = [row[:] for row in state]
        else:
            self._state = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
            # 初期配置: 中央4マスに石を配置
            self._state[3][3] = WHITE
            self._state[3][4] = BLACK
            self._state[4][3] = BLACK
            self._state[4][4] = WHITE

    def get_state(self) -> List[List[int]]:
        """現在の盤面状態を返す

        Returns:
            8×8の2次元リスト（コピー）
        """
        return [row[:] for row in self._state]

    def place_stone(self, row: int, col: int, color: int) -> List[Tuple[int, int]]:
        """石を配置し、反転した石の座標リストを返す

        Args:
            row: 行 (0-7)
            col: 列 (0-7)
            color: 石の色 (BLACK=1 or WHITE=2)

        Returns:
            反転した石の座標リスト。不正な手の場合は空リスト。
        """
        if not self._is_within_bounds(row, col):
            return []
        if self._state[row][col] != EMPTY:
            return []

        opponent = WHITE if color == BLACK else BLACK
        flipped: List[Tuple[int, int]] = []

        for dr, dc in DIRECTIONS:
            stones_to_flip = self._get_flipped_in_direction(row, col, dr, dc, color, opponent)
            flipped.extend(stones_to_flip)

        if not flipped:
            return []

        # 石を配置
        self._state[row][col] = color
        # 反転を実行
        for r, c in flipped:
            self._state[r][c] = color

        return flipped

    def get_legal_moves(self, color: int) -> List[Tuple[int, int]]:
        """指定色の合法手リストを返す

        Args:
            color: 石の色 (BLACK=1 or WHITE=2)

        Returns:
            合法手の座標リスト
        """
        opponent = WHITE if color == BLACK else BLACK
        legal_moves: List[Tuple[int, int]] = []

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self._state[row][col] != EMPTY:
                    continue
                if self._is_legal_move(row, col, color, opponent):
                    legal_moves.append((row, col))

        return legal_moves

    def count_stones(self) -> Dict[str, int]:
        """黒白の石数を返す

        Returns:
            {"black": 黒の石数, "white": 白の石数}
        """
        black_count = 0
        white_count = 0
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self._state[row][col] == BLACK:
                    black_count += 1
                elif self._state[row][col] == WHITE:
                    white_count += 1
        return {"black": black_count, "white": white_count}

    def is_full(self) -> bool:
        """盤面が全て埋まっているか

        Returns:
            全セルが埋まっていればTrue
        """
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self._state[row][col] == EMPTY:
                    return False
        return True

    def copy(self) -> "Board":
        """盤面のディープコピーを返す

        Returns:
            新しいBoardインスタンス
        """
        return Board(state=self._state)

    def _is_within_bounds(self, row: int, col: int) -> bool:
        """座標が盤面内かどうか判定"""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def _get_flipped_in_direction(
        self, row: int, col: int, dr: int, dc: int, color: int, opponent: int
    ) -> List[Tuple[int, int]]:
        """特定方向に反転される石のリストを返す"""
        stones: List[Tuple[int, int]] = []
        r, c = row + dr, col + dc

        # 相手の石が続く間、リストに追加
        while self._is_within_bounds(r, c) and self._state[r][c] == opponent:
            stones.append((r, c))
            r += dr
            c += dc

        # 自分の石で挟めていれば反転対象
        if stones and self._is_within_bounds(r, c) and self._state[r][c] == color:
            return stones

        return []

    def _is_legal_move(self, row: int, col: int, color: int, opponent: int) -> bool:
        """指定位置が合法手かどうか判定"""
        for dr, dc in DIRECTIONS:
            if self._get_flipped_in_direction(row, col, dr, dc, color, opponent):
                return True
        return False
