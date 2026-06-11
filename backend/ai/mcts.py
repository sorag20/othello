"""Monte Carlo Tree Search (MCTS) implementation.

Provides MCTSNode and MCTS classes implementing the PUCT algorithm
for tree search with neural network leaf evaluation and backpropagation.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from game.board import Board
from game.constants import BLACK, BOARD_SIZE, WHITE


class MCTSNode:
    """MCTSの探索木ノード

    各ノードはゲーム状態、訪問回数、累計価値、子ノードを保持する。
    """

    def __init__(
        self,
        board: Board,
        color: int,
        parent: Optional["MCTSNode"] = None,
        move: Optional[Tuple[int, int]] = None,
        prior: float = 0.0,
    ):
        """ノードを初期化

        Args:
            board: この時点での盤面状態
            color: このノードでの手番色
            parent: 親ノード（ルートの場合はNone）
            move: 親からこのノードに至った手
            prior: ニューラルネットワークから得た事前確率
        """
        self.board = board
        self.color = color
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children: Dict[Tuple[int, int], "MCTSNode"] = {}
        self._visit_count: int = 0
        self._total_value: float = 0.0

    @property
    def visit_count(self) -> int:
        """訪問回数を返す"""
        return self._visit_count

    @property
    def value(self) -> float:
        """平均価値 Q = W / N を返す（未訪問は0.0）"""
        if self._visit_count == 0:
            return 0.0
        return self._total_value / self._visit_count

    @property
    def is_expanded(self) -> bool:
        """ノードが展開済みかどうか"""
        return len(self.children) > 0


class MCTS:
    """Monte Carlo Tree Search

    PUCT アルゴリズムを使用し、ニューラルネットワークによるリーフ評価と
    バックプロパゲーションを行う。
    """

    def __init__(self, neural_network, simulations: int = 800, c_puct: float = 1.0):
        """MCTSを初期化

        Args:
            neural_network: NeuralNetworkインスタンス（predict メソッドを持つ）
            simulations: シミュレーション回数（デフォルト800）
            c_puct: 探索定数（デフォルト1.0）
        """
        self.neural_network = neural_network
        self.simulations = simulations
        self.c_puct = c_puct

    def search(self, board: Board, color: int) -> Tuple[int, int]:
        """探索を実行し、最も訪問数の多い手を返す

        Args:
            board: 現在の盤面
            color: 現在の手番色

        Returns:
            最も訪問数の多い手の座標 (row, col)
        """
        root = MCTSNode(board=board.copy(), color=color)

        # ルートを展開
        self._expand(root)

        # ルートに子がない場合（合法手なし）
        if not root.children:
            # パスの場合：相手の合法手を確認
            opponent = WHITE if color == BLACK else BLACK
            opponent_moves = board.get_legal_moves(opponent)
            if opponent_moves:
                # パスせざるを得ないが、相手は打てる - 本来ここには来ないはず
                # 安全のためパスを返す代わりに例外的に(-1,-1)を返す
                pass
            return (-1, -1)

        # シミュレーション実行
        for _ in range(self.simulations):
            node = root
            # SELECT: 未展開ノードに到達するまでPUCTで選択
            node = self._select(node)

            # ゲーム終了ノードか確認
            value = self._get_terminal_value(node)
            if value is not None:
                # ターミナルノード: 実際の結果をバックプロパゲート
                self._backpropagate(node, value)
                continue

            # EXPAND: リーフノードを展開
            if not node.is_expanded:
                value = self._expand(node)
            else:
                # 既に展開済みだが子がない（合法手なし） -> パスノード扱い
                value = self._evaluate_pass_node(node)

            # BACKPROPAGATE
            self._backpropagate(node, value)

        # 最も訪問数の多い手を返す
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        return best_move

    def get_policy(self, board: Board, color: int) -> List[float]:
        """MCTS探索後のポリシー（訪問比率）を返す

        Args:
            board: 現在の盤面
            color: 現在の手番色

        Returns:
            64要素のリスト（各セルの訪問比率、合計1.0）
        """
        root = MCTSNode(board=board.copy(), color=color)

        # ルートを展開
        self._expand(root)

        if not root.children:
            # 合法手がない場合、均一分布を返す
            return [1.0 / 64.0] * 64

        # シミュレーション実行
        for _ in range(self.simulations):
            node = root
            node = self._select(node)

            value = self._get_terminal_value(node)
            if value is not None:
                self._backpropagate(node, value)
                continue

            if not node.is_expanded:
                value = self._expand(node)
            else:
                value = self._evaluate_pass_node(node)

            self._backpropagate(node, value)

        # 訪問比率を64要素のリストとして返す
        policy = [0.0] * (BOARD_SIZE * BOARD_SIZE)
        total_visits = sum(child.visit_count for child in root.children.values())

        if total_visits > 0:
            for move, child in root.children.items():
                index = move[0] * BOARD_SIZE + move[1]
                policy[index] = child.visit_count / total_visits

        return policy

    def _select(self, node: MCTSNode) -> MCTSNode:
        """PUCTアルゴリズムでリーフノードまでトラバース

        Args:
            node: ルートノード

        Returns:
            展開されていないリーフノード
        """
        while node.is_expanded:
            # 子がなければリーフとして扱う
            if not node.children:
                return node
            # PUCTで最良の子を選択
            best_score = -float("inf")
            best_child = None
            for child in node.children.values():
                score = self._puct_score(node, child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child
        return node

    def _puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """PUCTスコアを計算

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(parent)) / (1 + N(child))

        Args:
            parent: 親ノード
            child: 子ノード

        Returns:
            PUCTスコア
        """
        q_value = child.value
        exploration = (
            self.c_puct
            * child.prior
            * math.sqrt(parent.visit_count)
            / (1 + child.visit_count)
        )
        return q_value + exploration

    def _expand(self, node: MCTSNode) -> float:
        """ノードを展開し、ニューラルネットワークで評価

        Args:
            node: 展開するノード

        Returns:
            ニューラルネットワークのバリュー評価値
        """
        legal_moves = node.board.get_legal_moves(node.color)

        if not legal_moves:
            # 合法手がない場合、パスまたはゲーム終了
            # ゲーム終了判定
            terminal_value = self._get_terminal_value(node)
            if terminal_value is not None:
                return terminal_value
            # パスの場合は相手のターンを評価
            return self._evaluate_pass_node(node)

        # ニューラルネットワークで評価
        policy, value = self.neural_network.predict(node.board, node.color)

        # 各合法手に対して子ノードを作成
        opponent = WHITE if node.color == BLACK else BLACK
        for move in legal_moves:
            child_board = node.board.copy()
            child_board.place_stone(move[0], move[1], node.color)
            prior = policy[move[0] * BOARD_SIZE + move[1]]
            child = MCTSNode(
                board=child_board,
                color=opponent,
                parent=node,
                move=move,
                prior=float(prior),
            )
            node.children[move] = child

        return value

    def _evaluate_pass_node(self, node: MCTSNode) -> float:
        """パスノード（合法手なし、ゲーム終了でない）を評価

        Args:
            node: パスノード

        Returns:
            評価値
        """
        # 相手にターンを渡してニューラルネットワークで評価
        opponent = WHITE if node.color == BLACK else BLACK
        _, value = self.neural_network.predict(node.board, opponent)
        # 相手視点の値なので符号を反転
        return -value

    def _get_terminal_value(self, node: MCTSNode) -> Optional[float]:
        """ゲーム終了ノードの値を返す

        Args:
            node: 確認するノード

        Returns:
            終了していれば値（1.0=現在手番の勝ち, -1.0=負け, 0.0=引き分け）
            終了していなければNone
        """
        # 両プレイヤーとも合法手がない、または盤面が埋まっている
        if node.board.is_full():
            return self._compute_game_result(node)

        current_moves = node.board.get_legal_moves(node.color)
        opponent = WHITE if node.color == BLACK else BLACK
        opponent_moves = node.board.get_legal_moves(opponent)

        if not current_moves and not opponent_moves:
            # 両者とも合法手なし -> ゲーム終了
            return self._compute_game_result(node)

        return None

    def _compute_game_result(self, node: MCTSNode) -> float:
        """ゲーム結果を計算

        Args:
            node: ターミナルノード

        Returns:
            現在の手番視点での結果値
        """
        counts = node.board.count_stones()
        black_count = counts["black"]
        white_count = counts["white"]

        if black_count == white_count:
            return 0.0
        elif (node.color == BLACK and black_count > white_count) or (
            node.color == WHITE and white_count > black_count
        ):
            return 1.0
        else:
            return -1.0

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """リーフからルートまで値をバックプロパゲート

        各レベルで値を反転させる（プレイヤーが交互するため）。

        Args:
            node: リーフノード
            value: リーフノードの評価値
        """
        current = node
        current_value = value
        while current is not None:
            current._visit_count += 1
            current._total_value += current_value
            # 親に戻る際に値を反転（交互手番）
            current_value = -current_value
            current = current.parent
