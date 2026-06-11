"""Self-play training module.

Provides the SelfPlay class for generating training data through
self-play games and training the neural network with the collected samples.
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ai.mcts import MCTS
from ai.neural_network import NeuralNetwork
from game.board import Board
from game.constants import BLACK, BOARD_SIZE, WHITE

DEFAULT_MODEL_PATH = "models/weights/model.pt"


@dataclass
class TrainingSample:
    """学習サンプル: 盤面テンソル、MCTSポリシー、ゲーム結果"""

    board_tensor: torch.Tensor  # shape: (3, 8, 8)
    policy: np.ndarray  # shape: (64,), sum=1.0
    value: float  # -1.0 to 1.0 (game result from current player's perspective)


@dataclass
class SelfPlayConfig:
    """セルフプレイ設定パラメータ"""

    num_games: int = 100  # 1-10000
    simulations: int = 800  # 1-10000
    learning_rate: float = 0.001  # 0.0001-0.1
    batch_size: int = 64  # 1-1024
    epochs: int = 10  # 1-1000

    def __post_init__(self) -> None:
        """パラメータ範囲を検証"""
        self.validate()

    def validate(self) -> None:
        """パラメータが有効範囲内か検証し、範囲外ならValueErrorを送出"""
        if not (1 <= self.num_games <= 10000):
            raise ValueError(
                f"num_games must be between 1 and 10000, got {self.num_games}"
            )
        if not (1 <= self.simulations <= 10000):
            raise ValueError(
                f"simulations must be between 1 and 10000, got {self.simulations}"
            )
        if not (0.0001 <= self.learning_rate <= 0.1):
            raise ValueError(
                f"learning_rate must be between 0.0001 and 0.1, got {self.learning_rate}"
            )
        if not (1 <= self.batch_size <= 1024):
            raise ValueError(
                f"batch_size must be between 1 and 1024, got {self.batch_size}"
            )
        if not (1 <= self.epochs <= 1000):
            raise ValueError(
                f"epochs must be between 1 and 1000, got {self.epochs}"
            )


@dataclass
class TrainingSummary:
    """学習結果サマリー"""

    games_played: int
    total_samples: int
    final_loss: float


class SelfPlay:
    """セルフプレイによる学習データ生成・学習"""

    def __init__(self, neural_network: NeuralNetwork, config: SelfPlayConfig):
        """セルフプレイモジュールを初期化

        Args:
            neural_network: 評価に使用するNeuralNetworkインスタンス
            config: セルフプレイ設定

        Raises:
            ValueError: configのパラメータが範囲外の場合
        """
        config.validate()
        self.neural_network = neural_network
        self.config = config
        self.mcts = MCTS(
            neural_network=neural_network,
            simulations=config.simulations,
        )

    def generate_games(self, num_games: int) -> List[TrainingSample]:
        """指定数のゲームをセルフプレイで生成

        同一モデルの重みを使って両プレイヤーとして対局し、
        各手で(盤面テンソル, MCTSポリシー, ゲーム結果)を記録する。

        中断時（KeyboardInterrupt）は完了済みゲームのサンプルを保持し、
        未完了ゲームのサンプルは破棄する。

        Args:
            num_games: 生成するゲーム数

        Returns:
            全完了ゲームから収集したTrainingSampleのリスト
        """
        all_samples: List[TrainingSample] = []
        self._games_completed = 0

        try:
            for _ in range(num_games):
                game_samples = self._play_single_game()
                all_samples.extend(game_samples)
                self._games_completed += 1
        except KeyboardInterrupt:
            # 中断時: 完了済みゲームのサンプルは保持、未完了は破棄
            pass

        return all_samples

    def train(self, samples: List[TrainingSample]) -> TrainingSummary:
        """収集したデータでモデルを学習

        ポリシークロスエントロピーとバリューMSEの結合損失関数で学習を行う。

        Args:
            samples: 学習に使用するTrainingSampleのリスト

        Returns:
            学習結果のTrainingSummary
        """
        # Edge case: empty samples
        if not samples:
            return TrainingSummary(
                games_played=self._games_completed if hasattr(self, "_games_completed") else 0,
                total_samples=0,
                final_loss=0.0,
            )

        # 1. Prepare data: convert samples to tensors
        board_tensors = torch.stack([s.board_tensor for s in samples])  # (N, 3, 8, 8)
        target_policies = torch.stack(
            [torch.from_numpy(s.policy).float() for s in samples]
        )  # (N, 64)
        target_values = torch.tensor(
            [[s.value] for s in samples], dtype=torch.float32
        )  # (N, 1)

        device = self.neural_network.device

        # 2. Create DataLoader
        dataset = TensorDataset(board_tensors, target_policies, target_values)
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # 3. Set up optimizer
        optimizer = torch.optim.Adam(
            self.neural_network.model.parameters(), lr=self.config.learning_rate
        )

        # 4. Training loop
        final_loss = 0.0
        self.neural_network.model.train()

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_boards, batch_policies, batch_values in dataloader:
                batch_boards = batch_boards.to(device)
                batch_policies = batch_policies.to(device)
                batch_values = batch_values.to(device)

                # Forward pass
                pred_policies, pred_values = self.neural_network.model(batch_boards)

                # Policy loss: cross-entropy
                policy_loss = -torch.sum(
                    batch_policies * torch.log(pred_policies + 1e-8)
                ) / batch_boards.size(0)

                # Value loss: MSE
                value_loss = torch.mean((pred_values - batch_values) ** 2)

                # Combined loss
                loss = policy_loss + value_loss

                # Backward + optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                final_loss = epoch_loss / batch_count

        # 5. After training
        self.neural_network.model.eval()

        # Save weights
        model_path = DEFAULT_MODEL_PATH
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.neural_network.save_weights(model_path)

        return TrainingSummary(
            games_played=self._games_completed if hasattr(self, "_games_completed") else 0,
            total_samples=len(samples),
            final_loss=final_loss,
        )

    def run(self) -> TrainingSummary:
        """セルフプレイ→学習の一連の流れを実行

        1. config.num_games分のゲームをセルフプレイで生成
        2. 収集したサンプルで学習を実行

        Returns:
            学習結果のTrainingSummary
        """
        samples = self.generate_games(self.config.num_games)
        summary = self.train(samples)
        return summary

    def _play_single_game(self) -> List[TrainingSample]:
        """1ゲームをセルフプレイで実行し、学習サンプルを返す

        Returns:
            このゲームから生成されたTrainingSampleのリスト
        """
        board = Board()
        current_color = BLACK

        # 各手の記録: (board_tensor, policy, color_at_move)
        game_records: List[tuple] = []
        consecutive_passes = 0

        while True:
            legal_moves = board.get_legal_moves(current_color)

            if not legal_moves:
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    # 両者パス → ゲーム終了
                    break
                # パス: 相手の手番へ
                current_color = WHITE if current_color == BLACK else BLACK
                continue

            consecutive_passes = 0

            # MCTSでポリシーを取得
            mcts_policy = self.mcts.get_policy(board, current_color)
            policy_array = np.array(mcts_policy, dtype=np.float32)

            # 盤面をテンソルにエンコード
            board_tensor = self.neural_network.encode_board(board, current_color)

            # 記録を保存（valueは後で確定）
            game_records.append((board_tensor, policy_array, current_color))

            # MCTSポリシーに基づいて手を選択（訪問比率に比例）
            move = self._select_move(policy_array, legal_moves)

            # 手を適用
            board.place_stone(move[0], move[1], current_color)

            # 盤面が埋まったらゲーム終了
            if board.is_full():
                break

            # 手番交代
            current_color = WHITE if current_color == BLACK else BLACK

        # ゲーム結果を判定
        game_result = self._determine_game_result(board)

        # 各サンプルのvalueをゲーム結果から確定
        samples: List[TrainingSample] = []
        for board_tensor, policy, color_at_move in game_records:
            # 勝者の色に基づいて、その手番のプレイヤー視点のvalue
            if game_result is None:
                # 引き分け
                value = 0.0
            elif game_result == color_at_move:
                # この手番のプレイヤーが勝利
                value = 1.0
            else:
                # この手番のプレイヤーが敗北
                value = -1.0

            samples.append(
                TrainingSample(
                    board_tensor=board_tensor,
                    policy=policy,
                    value=value,
                )
            )

        return samples

    def _select_move(
        self, policy: np.ndarray, legal_moves: List[tuple]
    ) -> tuple:
        """MCTSポリシーに基づいて手を選択

        ポリシーの確率分布に比例して合法手から1手を選択する。

        Args:
            policy: 64要素のポリシーベクトル
            legal_moves: 合法手のリスト

        Returns:
            選択された手の座標 (row, col)
        """
        # 合法手のインデックスと確率を抽出
        move_probs = []
        for move in legal_moves:
            index = move[0] * BOARD_SIZE + move[1]
            move_probs.append(policy[index])

        move_probs = np.array(move_probs, dtype=np.float32)

        # 確率の合計が0の場合（エッジケース）、均一分布にフォールバック
        prob_sum = move_probs.sum()
        if prob_sum > 0:
            move_probs = move_probs / prob_sum
        else:
            move_probs = np.ones(len(legal_moves), dtype=np.float32) / len(legal_moves)

        # 確率に比例して手を選択
        chosen_index = np.random.choice(len(legal_moves), p=move_probs)
        return legal_moves[chosen_index]

    def _determine_game_result(self, board: Board):
        """ゲーム結果を判定

        Args:
            board: 終了時の盤面

        Returns:
            BLACK, WHITE, or None (引き分け)
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


