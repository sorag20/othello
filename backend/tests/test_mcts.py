"""Tests for MCTS implementation.

Unit tests for MCTSNode and MCTS classes including tree search,
node selection, PUCT scoring, and move policy generation.
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from ai.mcts import MCTS, MCTSNode
from game.board import Board
from game.constants import BLACK, BOARD_SIZE, WHITE


class TestMCTSNode:
    """MCTSNodeのユニットテスト"""

    def test_initialization(self):
        """ノードの初期状態が正しいことを確認"""
        board = Board()
        node = MCTSNode(board=board, color=BLACK)

        assert node.board is board
        assert node.color == BLACK
        assert node.parent is None
        assert node.move is None
        assert node.prior == 0.0
        assert node.visit_count == 0
        assert node.value == 0.0
        assert node.is_expanded is False
        assert node.children == {}

    def test_initialization_with_parent(self):
        """親ノード付きの初期化が正しいことを確認"""
        board = Board()
        parent = MCTSNode(board=board, color=BLACK)
        child = MCTSNode(
            board=board, color=WHITE, parent=parent, move=(2, 3), prior=0.25
        )

        assert child.parent is parent
        assert child.move == (2, 3)
        assert child.prior == 0.25
        assert child.color == WHITE

    def test_value_after_visits(self):
        """訪問後の平均価値が正しく計算されることを確認"""
        board = Board()
        node = MCTSNode(board=board, color=BLACK)

        # 手動で訪問カウントと合計値を設定
        node._visit_count = 4
        node._total_value = 2.0

        assert node.visit_count == 4
        assert node.value == 0.5

    def test_is_expanded_with_children(self):
        """子ノードがある場合にis_expandedがTrueを返すことを確認"""
        board = Board()
        parent = MCTSNode(board=board, color=BLACK)
        child = MCTSNode(board=board, color=WHITE, parent=parent, move=(2, 3))
        parent.children[(2, 3)] = child

        assert parent.is_expanded is True


class TestMCTSPUCT:
    """PUCTスコア計算のテスト"""

    def test_puct_score_unvisited_child(self):
        """未訪問の子ノードのPUCTスコアが事前確率に依存することを確認"""
        nn = MagicMock()
        mcts = MCTS(neural_network=nn, simulations=10, c_puct=1.0)

        board = Board()
        parent = MCTSNode(board=board, color=BLACK)
        parent._visit_count = 10

        child = MCTSNode(board=board, color=WHITE, parent=parent, move=(2, 3), prior=0.5)
        # child: visit_count=0, value=0.0

        score = mcts._puct_score(parent, child)
        # Q = 0.0, exploration = 1.0 * 0.5 * sqrt(10) / (1+0) = 0.5 * sqrt(10)
        expected = 0.0 + 1.0 * 0.5 * math.sqrt(10) / 1.0
        assert abs(score - expected) < 1e-6

    def test_puct_score_visited_child(self):
        """訪問済みの子ノードのPUCTスコアが正しく計算されることを確認"""
        nn = MagicMock()
        mcts = MCTS(neural_network=nn, simulations=10, c_puct=2.0)

        board = Board()
        parent = MCTSNode(board=board, color=BLACK)
        parent._visit_count = 100

        child = MCTSNode(board=board, color=WHITE, parent=parent, move=(2, 3), prior=0.3)
        child._visit_count = 20
        child._total_value = 10.0  # Q = 0.5

        score = mcts._puct_score(parent, child)
        # Q = 10/20 = 0.5, exploration = 2.0 * 0.3 * sqrt(100) / (1+20) = 0.6 * 10 / 21
        expected_q = 0.5
        expected_exploration = 2.0 * 0.3 * math.sqrt(100) / 21.0
        expected = expected_q + expected_exploration
        assert abs(score - expected) < 1e-6

    def test_puct_prefers_high_prior_unvisited(self):
        """PUCTが高事前確率の未訪問ノードを優先することを確認"""
        nn = MagicMock()
        mcts = MCTS(neural_network=nn, simulations=10, c_puct=1.0)

        board = Board()
        parent = MCTSNode(board=board, color=BLACK)
        parent._visit_count = 10

        child_high = MCTSNode(
            board=board, color=WHITE, parent=parent, move=(2, 3), prior=0.8
        )
        child_low = MCTSNode(
            board=board, color=WHITE, parent=parent, move=(3, 2), prior=0.2
        )

        score_high = mcts._puct_score(parent, child_high)
        score_low = mcts._puct_score(parent, child_low)

        assert score_high > score_low


class TestMCTSSearch:
    """MCTS探索のテスト"""

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
            value = 0.0
            return policy, value

        nn.predict.side_effect = mock_predict
        return nn

    def test_search_returns_legal_move(self):
        """探索が合法手を返すことを確認"""
        nn = self._create_mock_nn()
        mcts = MCTS(neural_network=nn, simulations=50, c_puct=1.0)

        board = Board()
        legal_moves = board.get_legal_moves(BLACK)

        result = mcts.search(board, BLACK)

        assert result in legal_moves

    def test_search_with_few_simulations(self):
        """少ないシミュレーションでも合法手を返すことを確認"""
        nn = self._create_mock_nn()
        mcts = MCTS(neural_network=nn, simulations=5, c_puct=1.0)

        board = Board()
        legal_moves = board.get_legal_moves(BLACK)

        result = mcts.search(board, BLACK)
        assert result in legal_moves

    def test_visit_count_proportional_to_confidence(self):
        """高い事前確率のノードがより多く訪問されることを確認"""
        nn = MagicMock()

        def biased_predict(board, color):
            legal_moves = board.get_legal_moves(color)
            policy = np.zeros(64, dtype=np.float32)
            if legal_moves:
                # 最初の合法手に高い確率を与える
                first_move = legal_moves[0]
                policy[first_move[0] * BOARD_SIZE + first_move[1]] = 0.9
                # 残りに低い確率を分配
                remaining = len(legal_moves) - 1
                if remaining > 0:
                    for r, c in legal_moves[1:]:
                        policy[r * BOARD_SIZE + c] = 0.1 / remaining
            return policy, 0.0

        nn.predict.side_effect = biased_predict
        mcts = MCTS(neural_network=nn, simulations=100, c_puct=1.0)

        board = Board()
        legal_moves = board.get_legal_moves(BLACK)

        # 探索を実行
        root = MCTSNode(board=board.copy(), color=BLACK)
        mcts._expand(root)
        for _ in range(100):
            node = mcts._select(root)
            terminal_val = mcts._get_terminal_value(node)
            if terminal_val is not None:
                mcts._backpropagate(node, terminal_val)
                continue
            if not node.is_expanded:
                value = mcts._expand(node)
            else:
                value = mcts._evaluate_pass_node(node)
            mcts._backpropagate(node, value)

        # 最初の合法手が最も多く訪問されているか確認
        first_move = legal_moves[0]
        first_visits = root.children[first_move].visit_count
        total_visits = sum(c.visit_count for c in root.children.values())

        # 高い事前確率のノードが全体の少なくとも30%以上訪問されているはず
        assert first_visits / total_visits > 0.3

    def test_policy_output_sums_to_one(self):
        """ポリシー出力の合計が1.0であることを確認"""
        nn = self._create_mock_nn()
        mcts = MCTS(neural_network=nn, simulations=50, c_puct=1.0)

        board = Board()
        policy = mcts.get_policy(board, BLACK)

        assert len(policy) == 64
        assert abs(sum(policy) - 1.0) < 1e-6

    def test_policy_only_has_legal_moves(self):
        """ポリシーが合法手の位置にのみ正の値を持つことを確認"""
        nn = self._create_mock_nn()
        mcts = MCTS(neural_network=nn, simulations=50, c_puct=1.0)

        board = Board()
        legal_moves = board.get_legal_moves(BLACK)
        legal_indices = {r * BOARD_SIZE + c for r, c in legal_moves}

        policy = mcts.get_policy(board, BLACK)

        for i in range(64):
            if i in legal_indices:
                # 合法手は正の値を持つはず
                assert policy[i] >= 0.0
            else:
                # 非合法手は0のはず
                assert policy[i] == 0.0

    def test_backpropagation_alternates_value(self):
        """バックプロパゲーションで値が交互に反転されることを確認"""
        nn = self._create_mock_nn()
        mcts = MCTS(neural_network=nn, simulations=10, c_puct=1.0)

        board = Board()
        root = MCTSNode(board=board, color=BLACK)
        child = MCTSNode(board=board, color=WHITE, parent=root, move=(2, 3))
        grandchild = MCTSNode(board=board, color=BLACK, parent=child, move=(3, 2))

        # grandchildから値1.0をバックプロパゲート
        mcts._backpropagate(grandchild, 1.0)

        # grandchild: +1.0
        assert grandchild._total_value == 1.0
        assert grandchild._visit_count == 1
        # child: -1.0 (反転)
        assert child._total_value == -1.0
        assert child._visit_count == 1
        # root: +1.0 (再度反転)
        assert root._total_value == 1.0
        assert root._visit_count == 1

    def test_configurable_simulations(self):
        """シミュレーション回数が設定可能であることを確認"""
        nn = MagicMock()
        mcts_default = MCTS(neural_network=nn)
        mcts_custom = MCTS(neural_network=nn, simulations=100)

        assert mcts_default.simulations == 800
        assert mcts_custom.simulations == 100

    def test_configurable_c_puct(self):
        """探索定数c_puctが設定可能であることを確認"""
        nn = MagicMock()
        mcts_default = MCTS(neural_network=nn)
        mcts_custom = MCTS(neural_network=nn, c_puct=2.5)

        assert mcts_default.c_puct == 1.0
        assert mcts_custom.c_puct == 2.5
