"""AI強さ評価スクリプト。

固定戦略（ランダム・貪欲法）との対戦勝率で AI の強さを定量評価する。

使用例:
    # backend/ ディレクトリから実行
    python train/evaluate.py
    python train/evaluate.py --games 200
    python train/evaluate.py --model-path models/weights/model.pt --simulations 100
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mcts import MCTS
from ai.neural_network import NeuralNetwork
from game.board import Board
from game.constants import BLACK, WHITE

DEFAULTS = {
    "model_path": "models/weights/model.pt",
    "games": 100,
    "simulations": 100,
}


# ---------------------------------------------------------------------------
# プレイヤー戦略
# ---------------------------------------------------------------------------

class RandomPlayer:
    """合法手からランダムに選ぶ"""
    def choose_move(self, board: Board, color: int):
        moves = board.get_legal_moves(color)
        return random.choice(moves) if moves else None


class GreedyPlayer:
    """最も多く裏返せる手を選ぶ"""
    def choose_move(self, board: Board, color: int):
        moves = board.get_legal_moves(color)
        if not moves:
            return None
        return max(moves, key=lambda m: len(board.copy().place_stone(m[0], m[1], color)))


class AIPlayer:
    """MCTS + ニューラルネットワーク"""
    def __init__(self, model_path: str, simulations: int):
        self.nn = NeuralNetwork(model_path=model_path)
        self.mcts = MCTS(neural_network=self.nn, simulations=simulations)

    def choose_move(self, board: Board, color: int):
        moves = board.get_legal_moves(color)
        if not moves:
            return None
        return self.mcts.search(board, color)


# ---------------------------------------------------------------------------
# 対局
# ---------------------------------------------------------------------------

def play_game(black_player, white_player) -> str:
    """1局を実行して結果を返す ('black' / 'white' / 'draw')"""
    board = Board()
    players = {BLACK: black_player, WHITE: white_player}
    current = BLACK
    consecutive_passes = 0

    while True:
        moves = board.get_legal_moves(current)
        if not moves:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            current = WHITE if current == BLACK else BLACK
            continue

        consecutive_passes = 0
        move = players[current].choose_move(board, current)
        if move is None:
            break
        board.place_stone(move[0], move[1], current)

        if board.is_full():
            break
        current = WHITE if current == BLACK else BLACK

    counts = board.count_stones()
    if counts["black"] > counts["white"]:
        return "black"
    elif counts["white"] > counts["black"]:
        return "white"
    return "draw"


def run_match(ai_player, opponent, num_games: int, opponent_name: str) -> dict:
    """AI を黒・白それぞれ num_games/2 局ずつ対戦させて集計する"""
    half = num_games // 2
    wins = draws = losses = 0

    # AI が黒番
    for i in range(half):
        result = play_game(ai_player, opponent)
        if result == "black":
            wins += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1
        print(f"\r  {opponent_name}: {i + 1}/{num_games} 局完了...", end="", flush=True)

    # AI が白番
    for i in range(half):
        result = play_game(opponent, ai_player)
        if result == "white":
            wins += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1
        print(f"\r  {opponent_name}: {half + i + 1}/{num_games} 局完了...", end="", flush=True)

    print()
    total = wins + draws + losses
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "total": total,
        "win_rate": wins / total if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Othello AI 強さ評価")
    parser.add_argument("--model-path", type=str, default=DEFAULTS["model_path"],
                        help=f"モデル重みファイルパス (default: {DEFAULTS['model_path']})")
    parser.add_argument("--games", type=int, default=DEFAULTS["games"],
                        help=f"対戦局数（偶数推奨） (default: {DEFAULTS['games']})")
    parser.add_argument("--simulations", type=int, default=DEFAULTS["simulations"],
                        help=f"MCTS シミュレーション数/手 (default: {DEFAULTS['simulations']})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"エラー: モデルファイルが見つかりません: {args.model_path}")
        print("先に python train/train.py で学習してください。")
        sys.exit(1)

    print("=== Othello AI 評価 ===")
    print(f"  モデル      : {args.model_path}")
    print(f"  対戦局数    : {args.games} 局/相手")
    print(f"  シミュレーション: {args.simulations}")
    print()

    ai = AIPlayer(model_path=args.model_path, simulations=args.simulations)
    opponents = [
        ("ランダム",   RandomPlayer()),
        ("貪欲法",     GreedyPlayer()),
    ]

    results = {}
    for name, opponent in opponents:
        print(f"vs {name}:")
        results[name] = run_match(ai, opponent, args.games, name)

    print()
    print("=== 評価結果 ===")
    print(f"  {'相手':<10} {'勝率':>6}  {'勝':>4} {'分':>4} {'負':>4}")
    print(f"  {'-'*34}")
    for name, r in results.items():
        print(f"  {name:<10} {r['win_rate']:>6.1%}  {r['wins']:>4} {r['draws']:>4} {r['losses']:>4}")


if __name__ == "__main__":
    main()
