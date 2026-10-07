"""Self-play training module.

Provides the SelfPlay class for generating training data through
self-play games and training the neural network with the collected samples.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

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
    temperature_moves: int = 20  # 0-120
    parallel_games: int = 1  # 1-256, 自己対戦で同時に進めるゲーム数

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
        if not (0 <= self.temperature_moves <= 120):
            raise ValueError(
                f"temperature_moves must be between 0 and 120, got {self.temperature_moves}"
            )
        if not (1 <= self.parallel_games <= 256):
            raise ValueError(
                f"parallel_games must be between 1 and 256, got {self.parallel_games}"
            )


@dataclass
class TrainingSummary:
    """学習結果サマリー"""

    games_played: int
    total_samples: int
    final_loss: float


class _ParallelGameState:
    """_generate_games_parallel が同時に進める1ゲーム分の対局状態"""

    __slots__ = ("board", "color", "move_number", "consecutive_passes", "records")

    def __init__(self) -> None:
        self.board = Board()
        self.color = BLACK
        self.move_number = 0
        self.consecutive_passes = 0
        self.records: List[tuple] = []


class SelfPlay:
    """セルフプレイによる学習データ生成・学習"""

    def __init__(self, neural_network: NeuralNetwork, config: SelfPlayConfig, model_save_path: str = DEFAULT_MODEL_PATH):
        """セルフプレイモジュールを初期化

        Args:
            neural_network: 評価に使用するNeuralNetworkインスタンス
            config: セルフプレイ設定
            model_save_path: 学習後の重みファイル保存先パス

        Raises:
            ValueError: configのパラメータが範囲外の場合
        """
        config.validate()
        self.neural_network = neural_network
        self.config = config
        self.model_save_path = model_save_path
        self.mcts = MCTS(
            neural_network=neural_network,
            simulations=config.simulations,
        )
        # Adamのモーメント推定をイテレーションをまたいで保持するため、
        # optimizerはtrain()の呼び出しごとではなくここで一度だけ生成する。
        self.optimizer = torch.optim.Adam(
            self.neural_network.model.parameters(), lr=config.learning_rate
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
        if self.config.parallel_games > 1:
            return self._generate_games_parallel(num_games)

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

    def _generate_games_parallel(self, num_games: int) -> List[TrainingSample]:
        """複数ゲームを同時に進め、MCTSのリーフ評価をバッチ化して生成する

        対局そのもののロジック（温度スケジュール・パス処理・結果判定）は
        _play_single_game と同一。異なるのは、config.parallel_games局を
        同時に手番へ進め、各局のMCTS探索のNN呼び出しを
        MCTS.run_batched() でまとめて1回のバッチ推論にする点のみ
        （GPU利用効率を上げるための実行順序の変更であり、対局結果は変わらない）。

        空いたスロットには残りのゲームを都度補充する（continuous batching）ため、
        終盤で対局が終了するゲームが出てもバッチサイズが急に小さくならない。

        Args:
            num_games: 生成するゲーム数

        Returns:
            全完了ゲームから収集したTrainingSampleのリスト
        """
        all_samples: List[TrainingSample] = []
        self._games_completed = 0
        parallel = self.config.parallel_games

        active: Dict[int, _ParallelGameState] = {}
        remaining_games = num_games
        next_game_id = 0

        try:
            while True:
                # 空きスロットに新しいゲームを補充
                while len(active) < parallel and remaining_games > 0:
                    active[next_game_id] = _ParallelGameState()
                    next_game_id += 1
                    remaining_games -= 1

                if not active:
                    break

                # 合法手がないゲームはパスを消化する。二連続パスなら終局。
                searching_ids = []
                for game_id, state in list(active.items()):
                    if self._advance_passes(state):
                        searching_ids.append(game_id)
                    else:
                        all_samples.extend(self._finalize_game(state.board, state.records))
                        del active[game_id]
                        self._games_completed += 1

                if not searching_ids:
                    continue

                requests = [(active[gid].board, active[gid].color) for gid in searching_ids]
                roots = self.mcts.run_batched(requests, add_noise=True)

                for game_id, root in zip(searching_ids, roots):
                    state = active[game_id]
                    policy_array = np.array(self.mcts.root_visit_policy(root), dtype=np.float32)

                    board_tensor = self.neural_network.encode_board(state.board, state.color)
                    state.records.append((board_tensor, policy_array, state.color))

                    legal_moves = state.board.get_legal_moves(state.color)
                    greedy = state.move_number >= self.config.temperature_moves
                    move = self._select_move(policy_array, legal_moves, greedy=greedy)
                    state.move_number += 1
                    state.board.place_stone(move[0], move[1], state.color)

                    if state.board.is_full():
                        all_samples.extend(self._finalize_game(state.board, state.records))
                        del active[game_id]
                        self._games_completed += 1
                    else:
                        state.color = WHITE if state.color == BLACK else BLACK
        except KeyboardInterrupt:
            # 中断時: 完了済みゲームのサンプルは保持、進行中のゲームは破棄
            pass

        return all_samples

    def _advance_passes(self, state: "_ParallelGameState") -> bool:
        """合法手がある手番まで進める（_play_single_gameのパス処理と同一）

        Args:
            state: 対局状態

        Returns:
            合法手がある手番に到達できればTrue。二連続パスで終局していればFalse。
        """
        while not state.board.get_legal_moves(state.color):
            state.consecutive_passes += 1
            if state.consecutive_passes >= 2:
                return False
            state.color = WHITE if state.color == BLACK else BLACK
        state.consecutive_passes = 0
        return True

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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            if batch_count > 0:
                final_loss = epoch_loss / batch_count

        # 5. After training
        self.neural_network.model.eval()

        # Save weights
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        self.neural_network.save_weights(self.model_save_path)

        return TrainingSummary(
            games_played=self._games_completed if hasattr(self, "_games_completed") else 0,
            total_samples=len(samples),
            final_loss=final_loss,
        )

    def reset_optimizer(self) -> None:
        """Adamのモーメント推定をリセットする。

        ゲート処理でモデル重みを直前の採用済み状態に巻き戻した後にこれを呼ばないと、
        棄却された候補の勾配軌跡から蓄積したモーメントが残ったまま、
        巻き戻し後の（別の）パラメータに対して次イテレーションの更新がかかってしまう。
        """
        self.optimizer = torch.optim.Adam(
            self.neural_network.model.parameters(), lr=self.config.learning_rate
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
        move_number = 0

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

            # MCTSでポリシーを取得（探索多様性のためルートにディリクレノイズを付加）
            mcts_policy = self.mcts.get_policy(board, current_color, add_noise=True)
            policy_array = np.array(mcts_policy, dtype=np.float32)

            # 盤面をテンソルにエンコード
            board_tensor = self.neural_network.encode_board(board, current_color)

            # 記録を保存（valueは後で確定）
            game_records.append((board_tensor, policy_array, current_color))

            # 序盤(temperature_moves手)は訪問比率に比例して確率的に選択し多様性を確保。
            # それ以降は最も訪問数の多い手を決定的に選び、終盤の価値ラベルへの
            # ノイズ混入を防ぐ（AlphaZero方式の温度スケジュール）。
            greedy = move_number >= self.config.temperature_moves
            move = self._select_move(policy_array, legal_moves, greedy=greedy)
            move_number += 1

            # 手を適用
            board.place_stone(move[0], move[1], current_color)

            # 盤面が埋まったらゲーム終了
            if board.is_full():
                break

            # 手番交代
            current_color = WHITE if current_color == BLACK else BLACK

        return self._finalize_game(board, game_records)

    def _select_move(
        self, policy: np.ndarray, legal_moves: List[tuple], greedy: bool = False
    ) -> tuple:
        """MCTSポリシーに基づいて手を選択

        greedy=Falseの場合、ポリシーの確率分布に比例して合法手から1手を選択する
        （探索多様性の確保）。greedy=Trueの場合、最も確率（訪問比率）の高い手を
        決定的に選択する（終盤の学習ターゲットへのノイズ混入を防ぐ）。

        Args:
            policy: 64要素のポリシーベクトル
            legal_moves: 合法手のリスト
            greedy: Trueの場合、確率的サンプリングではなく決定的に最善手を選ぶ

        Returns:
            選択された手の座標 (row, col)
        """
        # 合法手のインデックスと確率を抽出
        move_probs = []
        for move in legal_moves:
            index = move[0] * BOARD_SIZE + move[1]
            move_probs.append(policy[index])

        move_probs = np.array(move_probs, dtype=np.float32)

        if greedy:
            return legal_moves[int(np.argmax(move_probs))]

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

    def _finalize_game(self, board: Board, records: List[tuple]) -> List[TrainingSample]:
        """終局した対局の記録から、valueを確定したTrainingSampleを作成する

        _play_single_game と _generate_games_parallel の両方から使われる、
        対局終了時の共通処理。

        Args:
            board: 終了時の盤面
            records: (board_tensor, policy, color_at_move) のリスト

        Returns:
            TrainingSampleのリスト
        """
        game_result = self._determine_game_result(board)

        samples: List[TrainingSample] = []
        for board_tensor, policy, color_at_move in records:
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


