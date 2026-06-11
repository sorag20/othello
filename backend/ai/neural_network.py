"""PyTorch neural network model definition.

Provides OthelloNet (ResNet-based nn.Module) with policy and value heads,
and the NeuralNetwork wrapper class for prediction, encoding, and weight management.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game.board import Board
from game.constants import BLACK, BOARD_SIZE, WHITE


class ModelLoadError(Exception):
    """モデルファイルのロード失敗"""
    pass


class ResidualBlock(nn.Module):
    """残差ブロック: conv→BN→ReLU→conv→BN + skip connection→ReLU"""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class OthelloNet(nn.Module):
    """AlphaZeroニューラルネットワーク

    ResNetベースのアーキテクチャで、ポリシーヘッドとバリューヘッドを持つ。
    """

    def __init__(self, num_residual_blocks: int = 10, num_filters: int = 128):
        """ResNetベースのネットワークを構築

        Args:
            num_residual_blocks: 残差ブロック数（デフォルト10）
            num_filters: フィルタ数（デフォルト128）
        """
        super().__init__()

        # Input convolutional layer: 3 channels → num_filters
        self.input_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Policy head: conv 128→2, 1×1, BN, ReLU, flatten, FC→64
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        # Value head: conv 128→1, 1×1, BN, ReLU, flatten, FC→256, ReLU, FC→1, tanh
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """入力テンソル(batch, 3, 8, 8)→(policy[batch, 64], value[batch, 1])

        Args:
            x: 入力テンソル shape (batch, 3, 8, 8)

        Returns:
            (policy, value) タプル
            - policy: shape (batch, 64), softmax正規化済み
            - value: shape (batch, 1), tanh活性化済み [-1, 1]
        """
        # Input block
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)  # Flatten to (batch, 2*8*8)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)  # Flatten to (batch, 1*8*8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class NeuralNetwork:
    """ニューラルネットワークのラッパー

    OthelloNetのインスタンスを管理し、盤面の評価や重みの保存・ロードを行う。
    """

    def __init__(self, model_path: Optional[str] = None):
        """モデルを初期化（オプショナルに重みをロード）

        Args:
            model_path: モデル重みファイルパス。Noneの場合はランダム初期化。

        Raises:
            ModelLoadError: モデルファイルのロードに失敗した場合
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = OthelloNet()
        self.model.to(self.device)
        self.model.eval()

        if model_path is not None:
            self.load_weights(model_path)

    def predict(self, board: Board, color: int) -> Tuple[np.ndarray, float]:
        """盤面を評価し(policy[64], value)を返す

        盤面をエンコードしてニューラルネットワークに入力し、
        非合法手をマスクして再正規化したポリシーとバリューを返す。

        Args:
            board: 現在の盤面
            color: 現在の手番色 (BLACK=1 or WHITE=2)

        Returns:
            (policy, value) タプル
            - policy: numpy array shape (64,), 合法手のみ正規化済み
            - value: float in [-1, 1]
        """
        self.model.eval()
        with torch.no_grad():
            # Encode board to tensor
            input_tensor = self.encode_board(board, color)
            input_batch = input_tensor.unsqueeze(0).to(self.device)  # (1, 3, 8, 8)

            # Forward pass
            policy, value = self.model(input_batch)

            # Convert to numpy
            policy_np = policy.squeeze(0).cpu().numpy()  # (64,)
            value_float = value.squeeze().item()

        # Mask illegal moves
        legal_moves = board.get_legal_moves(color)
        legal_mask = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        for row, col in legal_moves:
            legal_mask[row * BOARD_SIZE + col] = 1.0

        # Apply mask
        policy_np = policy_np * legal_mask

        # Re-normalize
        policy_sum = policy_np.sum()
        if policy_sum > 0:
            policy_np = policy_np / policy_sum
        else:
            # No legal moves - return uniform over all positions (edge case)
            # This shouldn't happen in normal gameplay since predict is called
            # when there are legal moves available
            if len(legal_moves) > 0:
                for row, col in legal_moves:
                    policy_np[row * BOARD_SIZE + col] = 1.0 / len(legal_moves)

        return policy_np, value_float

    def encode_board(self, board: Board, color: int) -> torch.Tensor:
        """盤面を3×8×8テンソルにエンコード

        Channel 0: 現在のプレイヤーの石 (1.0 for occupied, 0.0 otherwise)
        Channel 1: 相手プレイヤーの石 (1.0 for occupied, 0.0 otherwise)
        Channel 2: 手番情報 (黒番=全1.0, 白番=全0.0)

        Args:
            board: 現在の盤面
            color: 現在の手番色 (BLACK=1 or WHITE=2)

        Returns:
            torch.Tensor shape (3, 8, 8), dtype float32
        """
        state = board.get_state()
        opponent = WHITE if color == BLACK else BLACK

        tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if state[row][col] == color:
                    tensor[0][row][col] = 1.0
                elif state[row][col] == opponent:
                    tensor[1][row][col] = 1.0

        # Channel 2: turn info (all 1.0 if black's turn, all 0.0 if white's turn)
        if color == BLACK:
            tensor[2] = 1.0

        return torch.from_numpy(tensor)

    def load_weights(self, path: str) -> None:
        """モデル重みをファイルからロード

        Args:
            path: 重みファイルパス

        Raises:
            ModelLoadError: ファイルが見つからない、または読み込みに失敗した場合
        """
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except FileNotFoundError:
            raise ModelLoadError(f"モデルファイルが見つかりません: {path}")
        except Exception as e:
            raise ModelLoadError(f"モデルのロードに失敗しました: {path}, エラー: {e}")

    def save_weights(self, path: str) -> None:
        """モデル重みをファイルに保存

        Args:
            path: 保存先ファイルパス

        Raises:
            ModelLoadError: 保存に失敗した場合
        """
        try:
            torch.save(self.model.state_dict(), path)
        except Exception as e:
            raise ModelLoadError(f"モデルの保存に失敗しました: {path}, エラー: {e}")
