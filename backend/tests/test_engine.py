"""Tests for AI Engine orchestrator.

Unit tests for AIEngine class including model loading, timeout handling,
move computation, and error handling.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai.engine import AIEngine, AITimeoutError, InvalidBoardStateError
from ai.neural_network import ModelLoadError
from game.board import Board
from game.constants import BLACK, BOARD_SIZE, WHITE


class TestAIEngineInitialization:
    """AIEngineの初期化テスト"""

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_successful_initialization(self, mock_mcts_cls, mock_nn_cls):
        """正常にモデルをロードしてエンジンが初期化されること"""
        mock_nn = MagicMock()
        mock_nn_cls.return_value = mock_nn

        engine = AIEngine(model_path="test_model.pth", simulations=100, timeout=3.0)

        mock_nn_cls.assert_called_once_with(model_path="test_model.pth")
        mock_mcts_cls.assert_called_once_with(
            neural_network=mock_nn, simulations=100
        )
        assert engine._timeout == 3.0
        assert engine._simulations == 100

    @patch("ai.engine.NeuralNetwork")
    def test_model_load_error_on_invalid_path(self, mock_nn_cls):
        """存在しないモデルパスでModelLoadErrorが発生すること"""
        mock_nn_cls.side_effect = ModelLoadError("モデルファイルが見つかりません")

        with pytest.raises(ModelLoadError):
            AIEngine(model_path="nonexistent.pth")

    @patch("ai.engine.NeuralNetwork")
    def test_generic_exception_wrapped_as_model_load_error(self, mock_nn_cls):
        """NeuralNetwork初期化時の一般例外がModelLoadErrorにラップされること"""
        mock_nn_cls.side_effect = RuntimeError("unexpected error")

        with pytest.raises(ModelLoadError, match="モデルの初期化に失敗しました"):
            AIEngine(model_path="bad_model.pth")

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_default_parameters(self, mock_mcts_cls, mock_nn_cls):
        """デフォルトパラメータが正しく設定されること"""
        engine = AIEngine(model_path="model.pth")

        assert engine._timeout == 5.0
        assert engine._simulations == 800


class TestAIEngineComputeMove:
    """AIEngine.compute_moveのテスト"""

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_returns_legal_move(self, mock_mcts_cls, mock_nn_cls):
        """compute_moveが合法手を返すこと"""
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = (2, 3)
        mock_mcts_cls.return_value = mock_mcts

        engine = AIEngine(model_path="model.pth")
        board = Board()

        move = engine.compute_move(board, BLACK)

        assert move == (2, 3)
        assert move in board.get_legal_moves(BLACK)

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_raises_invalid_board_state_when_no_legal_moves(
        self, mock_mcts_cls, mock_nn_cls
    ):
        """合法手がない場合にInvalidBoardStateErrorが発生すること"""
        engine = AIEngine(model_path="model.pth")

        # 合法手がないボードをモック
        board = MagicMock()
        board.get_legal_moves.return_value = []

        with pytest.raises(InvalidBoardStateError):
            engine.compute_move(board, BLACK)

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_timeout_raises_ai_timeout_error(self, mock_mcts_cls, mock_nn_cls):
        """タイムアウト時にAITimeoutErrorが発生すること"""
        mock_mcts = MagicMock()

        def slow_search(board, color):
            time.sleep(3.0)  # タイムアウトより長い
            return (2, 3)

        mock_mcts.search.side_effect = slow_search
        mock_mcts_cls.return_value = mock_mcts

        engine = AIEngine(model_path="model.pth", timeout=0.1)
        board = Board()

        with pytest.raises(AITimeoutError, match="exceeded.*timeout"):
            engine.compute_move(board, BLACK)

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_mcts_exception_propagated(self, mock_mcts_cls, mock_nn_cls):
        """MCTS内部で発生した例外が伝播すること"""
        mock_mcts = MagicMock()
        mock_mcts.search.side_effect = RuntimeError("MCTS internal error")
        mock_mcts_cls.return_value = mock_mcts

        engine = AIEngine(model_path="model.pth")
        board = Board()

        with pytest.raises(RuntimeError, match="MCTS internal error"):
            engine.compute_move(board, BLACK)

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_fallback_to_first_legal_move_on_invalid_mcts_result(
        self, mock_mcts_cls, mock_nn_cls
    ):
        """MCTSが不正な手を返した場合に最初の合法手にフォールバックすること"""
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = (7, 7)  # 初期盤面では不正な手
        mock_mcts_cls.return_value = mock_mcts

        engine = AIEngine(model_path="model.pth")
        board = Board()
        legal_moves = board.get_legal_moves(BLACK)

        move = engine.compute_move(board, BLACK)

        assert move == legal_moves[0]
        assert move in legal_moves

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_fallback_on_none_result(self, mock_mcts_cls, mock_nn_cls):
        """MCTSがNoneを返した場合にフォールバックすること"""
        mock_mcts = MagicMock()
        mock_mcts.search.return_value = None
        mock_mcts_cls.return_value = mock_mcts

        engine = AIEngine(model_path="model.pth")
        board = Board()
        legal_moves = board.get_legal_moves(BLACK)

        move = engine.compute_move(board, BLACK)

        assert move == legal_moves[0]


class TestAIEngineLoadModel:
    """AIEngine.load_modelのテスト"""

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_load_model_updates_weights(self, mock_mcts_cls, mock_nn_cls):
        """load_modelがモデル重みを更新し、MCTSを再構築すること"""
        mock_nn = MagicMock()
        mock_nn_cls.return_value = mock_nn

        engine = AIEngine(model_path="model.pth", simulations=200)

        # load_modelを呼び出す
        engine.load_model("new_model.pth")

        mock_nn.load_weights.assert_called_once_with("new_model.pth")
        # MCTSが再構築されることを確認（合計2回呼ばれる: init + load_model）
        assert mock_mcts_cls.call_count == 2

    @patch("ai.engine.NeuralNetwork")
    @patch("ai.engine.MCTS")
    def test_load_model_raises_on_failure(self, mock_mcts_cls, mock_nn_cls):
        """load_modelでロード失敗時にModelLoadErrorが発生すること"""
        mock_nn = MagicMock()
        mock_nn.load_weights.side_effect = ModelLoadError("ロード失敗")
        mock_nn_cls.return_value = mock_nn

        engine = AIEngine(model_path="model.pth")

        with pytest.raises(ModelLoadError):
            engine.load_model("bad_model.pth")
