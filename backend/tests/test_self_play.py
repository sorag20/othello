"""Tests for self-play module.

Unit tests for SelfPlay class, SelfPlayConfig validation,
TrainingSample generation, and interruption handling.
"""

import os
import signal
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ai.self_play import SelfPlay, SelfPlayConfig, TrainingSample, TrainingSummary
from game.board import Board
from game.constants import BLACK, BOARD_SIZE, WHITE


class TestSelfPlayConfig:
    """SelfPlayConfigのバリデーションテスト"""

    def test_valid_default_config(self):
        """デフォルト設定が有効であることを確認"""
        config = SelfPlayConfig()
        assert config.num_games == 100
        assert config.simulations == 800
        assert config.learning_rate == 0.001
        assert config.batch_size == 64
        assert config.epochs == 10

    def test_valid_custom_config(self):
        """カスタム設定が有効であることを確認"""
        config = SelfPlayConfig(
            num_games=50,
            simulations=400,
            learning_rate=0.01,
            batch_size=128,
            epochs=5,
        )
        assert config.num_games == 50
        assert config.simulations == 400

    def test_valid_boundary_min(self):
        """最小境界値が受理されることを確認"""
        config = SelfPlayConfig(
            num_games=1,
            simulations=1,
            learning_rate=0.0001,
            batch_size=1,
            epochs=1,
        )
        assert config.num_games == 1

    def test_valid_boundary_max(self):
        """最大境界値が受理されることを確認"""
        config = SelfPlayConfig(
            num_games=10000,
            simulations=10000,
            learning_rate=0.1,
            batch_size=1024,
            epochs=1000,
        )
        assert config.num_games == 10000

    def test_invalid_num_games_zero(self):
        """num_games=0が拒否されることを確認"""
        with pytest.raises(ValueError, match="num_games"):
            SelfPlayConfig(num_games=0)

    def test_invalid_num_games_negative(self):
        """負のnum_gamesが拒否されることを確認"""
        with pytest.raises(ValueError, match="num_games"):
            SelfPlayConfig(num_games=-1)

    def test_invalid_num_games_too_large(self):
        """num_games>10000が拒否されることを確認"""
        with pytest.raises(ValueError, match="num_games"):
            SelfPlayConfig(num_games=10001)

    def test_invalid_simulations_zero(self):
        """simulations=0が拒否されることを確認"""
        with pytest.raises(ValueError, match="simulations"):
            SelfPlayConfig(simulations=0)

    def test_invalid_simulations_too_large(self):
        """simulations>10000が拒否されることを確認"""
        with pytest.raises(ValueError, match="simulations"):
            SelfPlayConfig(simulations=10001)

    def test_invalid_learning_rate_too_small(self):
        """learning_rate<0.0001が拒否されることを確認"""
        with pytest.raises(ValueError, match="learning_rate"):
            SelfPlayConfig(learning_rate=0.00001)

    def test_invalid_learning_rate_too_large(self):
        """learning_rate>0.1が拒否されることを確認"""
        with pytest.raises(ValueError, match="learning_rate"):
            SelfPlayConfig(learning_rate=0.2)

    def test_invalid_batch_size_zero(self):
        """batch_size=0が拒否されることを確認"""
        with pytest.raises(ValueError, match="batch_size"):
            SelfPlayConfig(batch_size=0)

    def test_invalid_batch_size_too_large(self):
        """batch_size>1024が拒否されることを確認"""
        with pytest.raises(ValueError, match="batch_size"):
            SelfPlayConfig(batch_size=1025)

    def test_invalid_epochs_zero(self):
        """epochs=0が拒否されることを確認"""
        with pytest.raises(ValueError, match="epochs"):
            SelfPlayConfig(epochs=0)

    def test_invalid_epochs_too_large(self):
        """epochs>1000が拒否されることを確認"""
        with pytest.raises(ValueError, match="epochs"):
            SelfPlayConfig(epochs=1001)


class TestSelfPlayInit:
    """SelfPlayの初期化テスト"""

    def test_initialization(self):
        """SelfPlayが正しく初期化されることを確認"""
        nn = MagicMock()
        config = SelfPlayConfig(num_games=10, simulations=50)
        sp = SelfPlay(nn, config)

        assert sp.neural_network is nn
        assert sp.config is config
        assert sp.mcts is not None
        assert sp.mcts.simulations == 50

    def test_initialization_with_invalid_config(self):
        """無効なconfigでの初期化が拒否されることを確認"""
        nn = MagicMock()
        # SelfPlayConfig.__post_init__で検証済みだが、
        # SelfPlay.__init__でも validate() を呼ぶ
        with pytest.raises(ValueError):
            SelfPlayConfig(num_games=0)


class TestSelfPlayGenerateGames:
    """SelfPlay.generate_gamesのテスト"""

    def _create_mock_nn(self):
        """テスト用のモックニューラルネットワークを作成"""
        nn = MagicMock()

        def mock_predict(board, color):
            legal_moves = board.get_legal_moves(color)
            policy = np.zeros(64, dtype=np.float32)
            if legal_moves:
                prob = 1.0 / len(legal_moves)
                for r, c in legal_moves:
                    policy[r * BOARD_SIZE + c] = prob
            return policy, 0.0

        def mock_encode_board(board, color):
            return torch.zeros(3, 8, 8)

        nn.predict.side_effect = mock_predict
        nn.encode_board.side_effect = mock_encode_board
        return nn

    def test_generate_single_game(self):
        """1ゲーム生成でサンプルが生成されることを確認"""
        nn = self._create_mock_nn()
        config = SelfPlayConfig(num_games=1, simulations=10)
        sp = SelfPlay(nn, config)

        samples = sp.generate_games(1)

        assert len(samples) > 0
        assert all(isinstance(s, TrainingSample) for s in samples)

    def test_sample_board_tensor_shape(self):
        """サンプルの盤面テンソルが正しい形状であることを確認"""
        nn = self._create_mock_nn()
        config = SelfPlayConfig(num_games=1, simulations=10)
        sp = SelfPlay(nn, config)

        samples = sp.generate_games(1)

        for sample in samples:
            assert sample.board_tensor.shape == (3, 8, 8)

    def test_sample_policy_shape_and_sum(self):
        """サンプルのポリシーが正しい形状で合計1.0であることを確認"""
        nn = self._create_mock_nn()
        config = SelfPlayConfig(num_games=1, simulations=10)
        sp = SelfPlay(nn, config)

        samples = sp.generate_games(1)

        for sample in samples:
            assert sample.policy.shape == (64,)
            assert abs(sample.policy.sum() - 1.0) < 1e-5

    def test_sample_value_range(self):
        """サンプルのvalueが{-1.0, 0.0, 1.0}のいずれかであることを確認"""
        nn = self._create_mock_nn()
        config = SelfPlayConfig(num_games=1, simulations=10)
        sp = SelfPlay(nn, config)

        samples = sp.generate_games(1)

        for sample in samples:
            assert sample.value in [-1.0, 0.0, 1.0]

    def test_generate_multiple_games(self):
        """複数ゲーム生成が期待通り動作することを確認"""
        nn = self._create_mock_nn()
        config = SelfPlayConfig(num_games=2, simulations=10)
        sp = SelfPlay(nn, config)

        samples = sp.generate_games(2)

        # 2ゲーム分のサンプルがあるはず（各ゲームは複数手）
        assert len(samples) > 2
        assert sp._games_completed == 2

    def test_interruption_keeps_completed_games(self):
        """中断時に完了済みゲームのサンプルが保持されることを確認"""
        nn = self._create_mock_nn()
        config = SelfPlayConfig(num_games=5, simulations=10)
        sp = SelfPlay(nn, config)

        game_count = [0]
        original_play = sp._play_single_game

        def play_with_interrupt():
            game_count[0] += 1
            if game_count[0] == 3:
                raise KeyboardInterrupt()
            return original_play()

        sp._play_single_game = play_with_interrupt

        samples = sp.generate_games(5)

        # 2ゲームが完了し、3ゲーム目で中断
        assert sp._games_completed == 2
        assert len(samples) > 0

    def test_games_completed_tracking(self):
        """_games_completedが正しく追跡されることを確認"""
        nn = self._create_mock_nn()
        config = SelfPlayConfig(num_games=3, simulations=10)
        sp = SelfPlay(nn, config)

        sp.generate_games(3)

        assert sp._games_completed == 3


class TestSelfPlayTrain:
    """SelfPlay.trainのテスト"""

    def _create_real_nn(self):
        """テスト用の実NeuralNetworkインスタンスを作成"""
        from ai.neural_network import NeuralNetwork
        return NeuralNetwork(model_path=None)

    def test_train_returns_summary(self):
        """trainがTrainingSummaryを返すことを確認"""
        nn = self._create_real_nn()
        config = SelfPlayConfig(num_games=1, simulations=10, epochs=1, batch_size=2)
        sp = SelfPlay(nn, config)
        sp._games_completed = 1

        samples = [
            TrainingSample(
                board_tensor=torch.randn(3, 8, 8),
                policy=np.ones(64, dtype=np.float32) / 64,
                value=1.0,
            ),
            TrainingSample(
                board_tensor=torch.randn(3, 8, 8),
                policy=np.ones(64, dtype=np.float32) / 64,
                value=-1.0,
            ),
        ]

        summary = sp.train(samples)

        assert isinstance(summary, TrainingSummary)
        assert summary.total_samples == 2
        assert summary.games_played == 1
        assert summary.final_loss > 0.0

    def test_train_empty_samples(self):
        """空のサンプルリストで学習した場合、loss=0.0で即座に返ることを確認"""
        nn = self._create_real_nn()
        config = SelfPlayConfig(num_games=1, simulations=10)
        sp = SelfPlay(nn, config)
        sp._games_completed = 0

        summary = sp.train([])

        assert isinstance(summary, TrainingSummary)
        assert summary.total_samples == 0
        assert summary.final_loss == 0.0

    def test_train_saves_weights(self, tmp_path, monkeypatch):
        """学習完了後にモデル重みが保存されることを確認"""
        import ai.self_play as self_play_module

        model_path = str(tmp_path / "model.pt")
        monkeypatch.setattr(self_play_module, "DEFAULT_MODEL_PATH", model_path)

        nn = self._create_real_nn()
        config = SelfPlayConfig(num_games=1, simulations=10, epochs=1, batch_size=2)
        sp = SelfPlay(nn, config)
        sp._games_completed = 1

        samples = [
            TrainingSample(
                board_tensor=torch.randn(3, 8, 8),
                policy=np.ones(64, dtype=np.float32) / 64,
                value=1.0,
            ),
        ]

        sp.train(samples)

        assert os.path.exists(model_path)

    def test_train_model_returns_to_eval_mode(self):
        """学習後にモデルがevalモードに戻ることを確認"""
        nn = self._create_real_nn()
        config = SelfPlayConfig(num_games=1, simulations=10, epochs=1, batch_size=2)
        sp = SelfPlay(nn, config)
        sp._games_completed = 1

        samples = [
            TrainingSample(
                board_tensor=torch.randn(3, 8, 8),
                policy=np.ones(64, dtype=np.float32) / 64,
                value=1.0,
            ),
        ]

        sp.train(samples)

        assert not nn.model.training

    def test_train_loss_decreases_with_epochs(self):
        """エポック数を増やすとlossが減少する傾向にあることを確認"""
        nn = self._create_real_nn()

        # Create consistent samples
        torch.manual_seed(42)
        samples = [
            TrainingSample(
                board_tensor=torch.randn(3, 8, 8),
                policy=np.ones(64, dtype=np.float32) / 64,
                value=1.0,
            )
            for _ in range(10)
        ]

        # Train with 1 epoch
        config1 = SelfPlayConfig(num_games=1, simulations=10, epochs=1, batch_size=10)
        sp1 = SelfPlay(nn, config1)
        sp1._games_completed = 1
        summary1 = sp1.train(samples)

        # Re-initialize and train with more epochs
        nn2 = self._create_real_nn()
        config2 = SelfPlayConfig(num_games=1, simulations=10, epochs=10, batch_size=10)
        sp2 = SelfPlay(nn2, config2)
        sp2._games_completed = 1
        summary2 = sp2.train(samples)

        # More epochs should lead to lower or equal loss
        assert summary2.final_loss <= summary1.final_loss


class TestSelfPlayRun:
    """SelfPlay.runのテスト"""

    def _create_mock_nn(self):
        """テスト用のモックニューラルネットワークを作成（generate_games用）"""
        from ai.neural_network import NeuralNetwork

        nn = NeuralNetwork(model_path=None)

        # Override predict for faster self-play
        original_predict = nn.predict

        def fast_predict(board, color):
            legal_moves = board.get_legal_moves(color)
            policy = np.zeros(64, dtype=np.float32)
            if legal_moves:
                prob = 1.0 / len(legal_moves)
                for r, c in legal_moves:
                    policy[r * BOARD_SIZE + c] = prob
            return policy, 0.0

        nn.predict = fast_predict
        return nn

    def test_run_returns_summary(self):
        """runがTrainingSummaryを返すことを確認"""
        nn = self._create_mock_nn()
        config = SelfPlayConfig(num_games=1, simulations=10, epochs=1, batch_size=32)
        sp = SelfPlay(nn, config)

        summary = sp.run()

        assert isinstance(summary, TrainingSummary)
        assert summary.games_played == 1
        assert summary.total_samples > 0
        assert summary.final_loss > 0.0


class TestTrainingSample:
    """TrainingSampleデータクラスのテスト"""

    def test_creation(self):
        """TrainingSampleが正しく作成されることを確認"""
        tensor = torch.randn(3, 8, 8)
        policy = np.ones(64, dtype=np.float32) / 64
        value = 1.0

        sample = TrainingSample(board_tensor=tensor, policy=policy, value=value)

        assert torch.equal(sample.board_tensor, tensor)
        assert np.array_equal(sample.policy, policy)
        assert sample.value == 1.0
