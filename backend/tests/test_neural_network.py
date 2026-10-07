"""Tests for neural network model.

Unit tests and property-based tests for OthelloNet and NeuralNetwork
including forward pass, output shapes, and weight management.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from ai.neural_network import ModelLoadError, NeuralNetwork, OthelloNet, ResidualBlock
from game.board import Board
from game.constants import BLACK, BOARD_SIZE, WHITE


class TestOthelloNet:
    """OthelloNet モデルのユニットテスト"""

    def test_default_initialization(self):
        """デフォルトパラメータで初期化できる"""
        model = OthelloNet()
        assert model is not None

    def test_custom_initialization(self):
        """カスタムパラメータで初期化できる"""
        model = OthelloNet(num_residual_blocks=5, num_filters=64)
        assert model is not None

    def test_forward_output_shapes(self):
        """forward passが正しい出力形状を返す"""
        model = OthelloNet()
        model.eval()
        batch_size = 4
        x = torch.randn(batch_size, 3, 8, 8)
        policy, value = model(x)

        assert policy.shape == (batch_size, 64)
        assert value.shape == (batch_size, 1)

    def test_forward_single_input(self):
        """バッチサイズ1の入力で動作する"""
        model = OthelloNet()
        model.eval()
        x = torch.randn(1, 3, 8, 8)
        policy, value = model(x)

        assert policy.shape == (1, 64)
        assert value.shape == (1, 1)

    def test_policy_is_probability_distribution(self):
        """ポリシー出力がsoftmax正規化されている（合計≈1.0）"""
        model = OthelloNet()
        model.eval()
        x = torch.randn(1, 3, 8, 8)
        policy, _ = model(x)

        policy_sum = policy.sum(dim=1).item()
        assert abs(policy_sum - 1.0) < 1e-5

    def test_policy_non_negative(self):
        """ポリシー出力が全て非負"""
        model = OthelloNet()
        model.eval()
        x = torch.randn(1, 3, 8, 8)
        policy, _ = model(x)

        assert (policy >= 0).all()

    def test_value_in_range(self):
        """バリュー出力が[-1, 1]の範囲内"""
        model = OthelloNet()
        model.eval()
        x = torch.randn(8, 3, 8, 8)
        _, value = model(x)

        assert (value >= -1.0).all()
        assert (value <= 1.0).all()


class TestResidualBlock:
    """ResidualBlock のユニットテスト"""

    def test_output_shape_preserved(self):
        """入力と出力の形状が同じ"""
        block = ResidualBlock(128)
        x = torch.randn(2, 128, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_skip_connection(self):
        """スキップ接続が機能している（出力がゼロでない）"""
        block = ResidualBlock(128)
        x = torch.randn(1, 128, 8, 8)
        out = block(x)
        assert not torch.allclose(out, torch.zeros_like(out))


class TestNeuralNetwork:
    """NeuralNetwork ラッパーのユニットテスト"""

    def test_initialization_without_model_path(self):
        """モデルパスなしで初期化できる"""
        nn = NeuralNetwork()
        assert nn.model is not None

    def test_initialization_with_invalid_path(self):
        """存在しないパスで初期化するとModelLoadErrorが発生"""
        with pytest.raises(ModelLoadError):
            NeuralNetwork(model_path="/nonexistent/model.pt")

    def test_encode_board_shape(self):
        """encode_boardが正しい形状のテンソルを返す"""
        nn = NeuralNetwork()
        board = Board()
        tensor = nn.encode_board(board, BLACK)

        assert tensor.shape == (3, 8, 8)
        assert tensor.dtype == torch.float32

    def test_encode_board_initial_position_black(self):
        """初期盤面の黒番エンコーディングが正しい"""
        nn = NeuralNetwork()
        board = Board()
        tensor = nn.encode_board(board, BLACK)

        # Channel 0: black stones (current player)
        assert tensor[0][3][4] == 1.0  # black stone
        assert tensor[0][4][3] == 1.0  # black stone
        assert tensor[0][3][3] == 0.0  # white stone position

        # Channel 1: white stones (opponent)
        assert tensor[1][3][3] == 1.0  # white stone
        assert tensor[1][4][4] == 1.0  # white stone
        assert tensor[1][3][4] == 0.0  # black stone position

        # Channel 2: turn info (black's turn = all 1.0)
        assert tensor[2].sum().item() == 64.0

    def test_encode_board_initial_position_white(self):
        """初期盤面の白番エンコーディングが正しい"""
        nn = NeuralNetwork()
        board = Board()
        tensor = nn.encode_board(board, WHITE)

        # Channel 0: white stones (current player)
        assert tensor[0][3][3] == 1.0  # white stone
        assert tensor[0][4][4] == 1.0  # white stone

        # Channel 1: black stones (opponent)
        assert tensor[1][3][4] == 1.0  # black stone
        assert tensor[1][4][3] == 1.0  # black stone

        # Channel 2: turn info (white's turn = all 0.0)
        assert tensor[2].sum().item() == 0.0

    def test_predict_output_types(self):
        """predictが正しい型の出力を返す"""
        nn = NeuralNetwork()
        board = Board()
        policy, value = nn.predict(board, BLACK)

        assert isinstance(policy, np.ndarray)
        assert isinstance(value, float)

    def test_predict_policy_shape(self):
        """predictのポリシー出力が正しい形状"""
        nn = NeuralNetwork()
        board = Board()
        policy, _ = nn.predict(board, BLACK)

        assert policy.shape == (64,)

    def test_predict_policy_sums_to_one(self):
        """predictのポリシーが合法手で正規化されている（合計≈1.0）"""
        nn = NeuralNetwork()
        board = Board()
        policy, _ = nn.predict(board, BLACK)

        assert abs(policy.sum() - 1.0) < 1e-6

    def test_predict_illegal_moves_are_zero(self):
        """predictのポリシーで非合法手が0"""
        nn = NeuralNetwork()
        board = Board()
        policy, _ = nn.predict(board, BLACK)

        legal_moves = board.get_legal_moves(BLACK)
        legal_positions = {row * BOARD_SIZE + col for row, col in legal_moves}

        for i in range(64):
            if i not in legal_positions:
                assert policy[i] == 0.0

    def test_predict_value_in_range(self):
        """predictのバリューが[-1, 1]の範囲内"""
        nn = NeuralNetwork()
        board = Board()
        _, value = nn.predict(board, BLACK)

        assert -1.0 <= value <= 1.0

    def test_predict_batch_empty(self):
        """predict_batchが空リストで空リストを返す"""
        nn = NeuralNetwork()
        assert nn.predict_batch([], []) == []

    def test_predict_batch_matches_predict(self):
        """predict_batchがpredictをN回呼ぶのと同じ結果を返す"""
        nn = NeuralNetwork()

        board_empty = Board()
        board_one_move = Board()
        board_one_move.place_stone(2, 3, BLACK)

        boards = [board_empty, board_one_move, Board()]
        colors = [BLACK, WHITE, WHITE]

        individual = [nn.predict(b, c) for b, c in zip(boards, colors)]
        batched = nn.predict_batch(boards, colors)

        assert len(batched) == len(individual)
        for (policy1, value1), (policy2, value2) in zip(individual, batched):
            assert np.allclose(policy1, policy2, atol=1e-5)
            assert abs(value1 - value2) < 1e-5

    def test_predict_batch_policy_shape_and_legal_mask(self):
        """predict_batchのポリシーが正しい形状で、非合法手が0"""
        nn = NeuralNetwork()
        board = Board()
        results = nn.predict_batch([board], [BLACK])

        assert len(results) == 1
        policy, value = results[0]
        assert policy.shape == (64,)
        assert isinstance(value, float)
        assert abs(policy.sum() - 1.0) < 1e-6

        legal_moves = board.get_legal_moves(BLACK)
        legal_positions = {row * BOARD_SIZE + col for row, col in legal_moves}
        for i in range(64):
            if i not in legal_positions:
                assert policy[i] == 0.0

    def test_save_and_load_weights(self):
        """重みの保存とロードが正しく動作する"""
        nn1 = NeuralNetwork()
        board = Board()

        # Get prediction before save
        policy1, value1 = nn1.predict(board, BLACK)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name

        try:
            # Save weights
            nn1.save_weights(temp_path)

            # Load into a new instance
            nn2 = NeuralNetwork(model_path=temp_path)
            policy2, value2 = nn2.predict(board, BLACK)

            # Should produce same results
            np.testing.assert_array_almost_equal(policy1, policy2)
            assert abs(value1 - value2) < 1e-6
        finally:
            os.unlink(temp_path)

    def test_load_weights_invalid_file(self):
        """不正なファイルをロードするとModelLoadErrorが発生"""
        nn = NeuralNetwork()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False, mode="w") as f:
            f.write("not a valid model file")
            temp_path = f.name

        try:
            with pytest.raises(ModelLoadError):
                nn.load_weights(temp_path)
        finally:
            os.unlink(temp_path)
