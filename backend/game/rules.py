"""Othello rules engine.

Provides the OthelloRules class with static methods for move validation,
stone flipping calculation, game-over detection, and winner determination.
"""

from typing import List, Optional, Tuple

from game.board import Board
from game.constants import BLACK, BOARD_SIZE, DIRECTIONS, EMPTY, WHITE


class OthelloRules:
    """オセロルールエンジン"""

    @staticmethod
    def is_valid_move(board: Board, row: int, col: int, color: int) -> bool:
        """指定位置が合法手かどうか判定

        Args:
            board: 現在の盤面
            row: 行 (0-7)
            col: 列 (0-7)
            color: 石の色 (BLACK=1 or WHITE=2)

        Returns:
            合法手であればTrue
        """
        # 盤面外チェック
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return False

        # 空セルチェック
        if board.get_state()[row][col] != EMPTY:
            return False

        # 少なくとも1つの石を反転できるかチェック
        flipped = OthelloRules.get_flipped_stones(board, row, col, color)
        return len(flipped) > 0

    @staticmethod
    def get_flipped_stones(
        board: Board, row: int, col: int, color: int
    ) -> List[Tuple[int, int]]:
        """配置時に反転される石の座標を返す

        8方向全てをチェックし、挟める相手石の座標をまとめて返す。

        Args:
            board: 現在の盤面
            row: 行 (0-7)
            col: 列 (0-7)
            color: 石の色 (BLACK=1 or WHITE=2)

        Returns:
            反転される石の座標リスト。反転不可の場合は空リスト。
        """
        # 盤面外チェック
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return []

        state = board.get_state()

        # 空セルでなければ反転不可
        if state[row][col] != EMPTY:
            return []

        opponent = WHITE if color == BLACK else BLACK
        all_flipped: List[Tuple[int, int]] = []

        for dr, dc in DIRECTIONS:
            flipped_in_dir: List[Tuple[int, int]] = []
            r, c = row + dr, col + dc

            # 相手の石が続く間、リストに追加
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and state[r][c] == opponent:
                flipped_in_dir.append((r, c))
                r += dr
                c += dc

            # 自分の石で挟めていれば反転対象
            if flipped_in_dir and 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and state[r][c] == color:
                all_flipped.extend(flipped_in_dir)

        return all_flipped

    @staticmethod
    def is_game_over(board: Board, current_color: int) -> bool:
        """ゲーム終了条件を判定

        ゲームが終了するのは以下のいずれかの場合:
        - 盤面が全て埋まっている（64セル全て使用済み）
        - 両プレイヤーとも合法手がない

        Args:
            board: 現在の盤面
            current_color: 現在のプレイヤーの色

        Returns:
            ゲーム終了ならTrue
        """
        # 盤面が全て埋まっていればゲーム終了
        if board.is_full():
            return True

        # 現在のプレイヤーに合法手があればゲーム続行
        if board.get_legal_moves(current_color):
            return False

        # 相手プレイヤーに合法手があればゲーム続行（パス）
        opponent = WHITE if current_color == BLACK else BLACK
        if board.get_legal_moves(opponent):
            return False

        # 両プレイヤーとも合法手なし → ゲーム終了
        return True

    @staticmethod
    def determine_winner(board: Board) -> Optional[int]:
        """勝者を判定

        石数を比較して勝者を決定する。

        Args:
            board: 現在の盤面

        Returns:
            BLACK(1): 黒の勝ち
            WHITE(2): 白の勝ち
            None: 引き分け
        """
        counts = board.count_stones()
        black_count = counts["black"]
        white_count = counts["white"]

        if black_count > white_count:
            return BLACK
        elif white_count > black_count:
            return WHITE
        else:
            return None
