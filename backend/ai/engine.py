"""AI Engine orchestrator.

Provides the AIEngine class that coordinates MCTS search with
neural network evaluation to compute the best move for a given
board state, with timeout handling and error management.
"""

import threading
from typing import Tuple

from game.board import Board
from ai.mcts import MCTS
from ai.neural_network import ModelLoadError, NeuralNetwork


class AITimeoutError(Exception):
    """AI計算のタイムアウト"""
    pass


class InvalidBoardStateError(Exception):
    """無効な盤面状態の入力"""
    pass


class AIEngine:
    """AlphaZero AIエンジン

    MCTSとニューラルネットワークを組み合わせてオセロの最善手を計算する。
    タイムアウト制御と合法手バリデーションを含む。
    """

    def __init__(self, model_path: str, simulations: int = 800, timeout: float = 5.0):
        """モデルをロードしてエンジンを初期化

        Args:
            model_path: 事前学習済みモデルの重みファイルパス
            simulations: MCTSシミュレーション回数（デフォルト800）
            timeout: 手計算のタイムアウト秒数（デフォルト5.0秒）

        Raises:
            ModelLoadError: モデルファイルのロードに失敗した場合
        """
        self._timeout = timeout
        self._simulations = simulations

        try:
            self._neural_network = NeuralNetwork(model_path=model_path)
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"モデルの初期化に失敗しました: {e}")

        self._mcts = MCTS(
            neural_network=self._neural_network,
            simulations=self._simulations,
        )

    def compute_move(self, board: Board, color: int) -> Tuple[int, int]:
        """MCTSを実行し最善手を返す（タイムアウト付き）

        タイムアウト制限付きでMCTS探索を実行し、最善手を返す。
        返される手が合法手であることを防御的にチェックする。

        Args:
            board: 現在の盤面
            color: 現在の手番色 (BLACK=1 or WHITE=2)

        Returns:
            最善手の座標 (row, col)

        Raises:
            AITimeoutError: 計算がタイムアウトした場合
            InvalidBoardStateError: 合法手が存在しない場合
        """
        # 合法手の存在チェック
        legal_moves = board.get_legal_moves(color)
        if not legal_moves:
            raise InvalidBoardStateError(
                "合法手が存在しない盤面でcompute_moveが呼ばれました"
            )

        # タイムアウト付きでMCTS探索を実行
        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = self._mcts.search(board, color)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self._timeout)

        if thread.is_alive():
            raise AITimeoutError(
                f"AI computation exceeded {self._timeout}s timeout"
            )

        if exception[0]:
            raise exception[0]

        move = result[0]

        # 防御的チェック: 返された手が合法手であることを確認
        if move is None or move not in legal_moves:
            # MCTSが不正な手を返した場合、合法手から最初のものを選択
            # (通常はここに到達しないが安全策として)
            move = legal_moves[0]

        return move

    def load_model(self, model_path: str) -> None:
        """事前学習済みモデルの重みをロード

        新しいモデル重みをロードし、MCTSインスタンスを再構築する。

        Args:
            model_path: モデル重みファイルパス

        Raises:
            ModelLoadError: モデルファイルのロードに失敗した場合
        """
        self._neural_network.load_weights(model_path)
        # MCTSを再構築（新しいネットワーク重みを反映）
        self._mcts = MCTS(
            neural_network=self._neural_network,
            simulations=self._simulations,
        )
