"""AI強さ評価スクリプト。

固定戦略（ランダム・貪欲法）および Edax との対戦勝率で AI の強さを定量評価する。
--engine で評価対象を切り替えられる: 'internal'（MCTS + ニューラルネット、デフォルト）
または 'minimax'（bitboard alpha-beta エンジン）。

使用例:
    # backend/ ディレクトリから実行
    python train/evaluate.py
    python train/evaluate.py --games 200
    python train/evaluate.py --model-path models/weights/model.pt --simulations 100
    python train/evaluate.py --edax-levels 1 5 10 --edax-games 20
    python train/evaluate.py --no-edax  # Edax を除外して簡易評価
    python train/evaluate.py --engine minimax  # minimax エンジンを評価
    python train/evaluate.py --engine minimax --minimax-time-limit 1.0 --edax-levels 1 5
"""

import argparse
import csv
import os
import queue
import random
import subprocess
import sys
import threading
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mcts import MCTS
from ai.minimax_engine import MinimaxEngine
from ai.neural_network import NeuralNetwork
from game.board import Board
from game.constants import BLACK, WHITE

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EDAX_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../edax-reversi"))
_EDAX_BINARY = os.path.join(_EDAX_DIR, "bin", "mEdax-armv8-a")

DEFAULTS = {
    "model_path": "models/weights/model.pt",
    "games": 100,
    "simulations": 100,
    "edax_levels": [1, 5, 10],
    "edax_games": 100,
    "engine": "internal",
    "minimax_time_limit": 3.0,
    "minimax_endgame_threshold": 10,
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


class MinimaxPlayer:
    """bitboard alpha-beta minimax エンジン"""
    def __init__(self, time_limit: float, endgame_empty_threshold: int):
        self.engine = MinimaxEngine(
            time_limit=time_limit,
            endgame_empty_threshold=endgame_empty_threshold,
        )

    def choose_move(self, board: Board, color: int):
        moves = board.get_legal_moves(color)
        if not moves:
            return None
        return self.engine.compute_move(board, color)


class EdaxPlayer:
    """Edax エンジン（サブプロセス）との通信プレイヤー。

    Edax の標準テキストプロトコルを使用:
      - go   : Edax が現在のプレイヤーとして着手 → "Edax plays X\n"
      - <移動> : ユーザー着手（例: "f5"）     → "You play X\n"
      - pa   : パス着手                        → "You play PA\n"
      - init : 盤面リセット
    """

    def __init__(
        self,
        level: int,
        binary: str = _EDAX_BINARY,
        edax_dir: str = _EDAX_DIR,
    ):
        self.level = level
        self.binary = binary
        self.edax_dir = edax_dir
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None

    # ------------------------------------------------------------------
    # プロセス管理
    # ------------------------------------------------------------------

    def _launch(self) -> None:
        """Edax プロセスを起動して初期設定を送る。"""
        self._proc = subprocess.Popen(
            ["stdbuf", "-oL", self.binary, "-q"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=self.edax_dir,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # 起動・ブック読み込み完了まで待機
        self._stall(0.5)
        self._cmd(f"level {self.level}")
        self._stall(0.2)
        self._cmd("mode 3")   # 手動モード（auto-play 無効）
        self._stall(0.1)

    def _read_loop(self) -> None:
        for line in self._proc.stdout:
            self._queue.put(line)
        self._queue.put(None)   # プロセス終了センチネル

    def _cmd(self, text: str) -> None:
        self._proc.stdin.write(text + "\n")
        self._proc.stdin.flush()

    def _stall(self, timeout: float) -> None:
        """キューに `timeout` 秒間新データが来なくなるまでドレインする。"""
        while True:
            try:
                self._queue.get(timeout=timeout)
            except queue.Empty:
                break

    def _read_until(self, marker: str, search_timeout: float = 120.0) -> str | None:
        """指定文字列を含む行が来るまで読み続ける。タイムアウト時は None。"""
        deadline = time.monotonic() + search_timeout
        while time.monotonic() < deadline:
            try:
                line = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if line is None:
                return None
            if marker in line:
                return line
        return None

    # ------------------------------------------------------------------
    # 対局インタフェース
    # ------------------------------------------------------------------

    def new_game(self) -> None:
        """新しい対局を開始する（プロセスを再利用）。"""
        if self._proc is None or self._proc.poll() is not None:
            self._launch()
        else:
            self._cmd("init")
            self._stall(0.2)

    def choose_move(self, _board: Board, _color: int):
        """Edax に着手させ (row, col) を返す。パス・エラー時は None。"""
        self._cmd("go")
        line = self._read_until("Edax plays")
        if line is None:
            return None
        # "Edax plays F5" → move_str = "f5"
        move_str = line.strip().split()[-1].lower()
        if move_str == "pa":
            return None   # Edax がパス
        col = ord(move_str[0]) - ord("a")
        row = int(move_str[1]) - 1
        return (row, col)

    def notify_move(self, row: int, col: int) -> None:
        """AI（相手）の着手を Edax に通知する。"""
        move_str = chr(ord("a") + col) + str(row + 1)
        self._cmd(move_str)
        self._read_until("You play", search_timeout=5.0)

    def notify_pass(self) -> None:
        """現在のプレイヤーのパスを Edax に通知する。"""
        self._cmd("pa")
        self._read_until("You play", search_timeout=5.0)

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._cmd("quit")
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.terminate()
        self._proc = None


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


def play_game_vs_edax(ai_player, edax: EdaxPlayer, ai_color: int) -> str:
    """AI と Edax の 1 局。ai_color が AI の石色。"""
    board = Board()
    edax.new_game()

    current = BLACK
    consecutive_passes = 0

    while True:
        moves = board.get_legal_moves(current)

        if not moves:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            if current == ai_color:
                # AI 側のパス: Edax に「相手がパスした」ことを通知する
                edax.notify_pass()
            else:
                # Edax 側のパス: go で Edax 自身にパスを認識させる（Edax plays PA）
                edax.choose_move(board, current)
            current = WHITE if current == BLACK else BLACK
            continue

        consecutive_passes = 0

        if current == ai_color:
            move = ai_player.choose_move(board, current)
            if move is None:
                break
            board.place_stone(move[0], move[1], current)
            edax.notify_move(move[0], move[1])
        else:
            move = edax.choose_move(board, current)
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

    for i in range(half):
        result = play_game(ai_player, opponent)
        if result == "black":
            wins += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1
        print(f"\r  {opponent_name}: {i + 1}/{num_games} 局完了...", end="", flush=True)

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


def run_match_vs_edax(ai_player, edax: EdaxPlayer, num_games: int, opponent_name: str) -> dict:
    """AI を黒・白それぞれ num_games/2 局ずつ Edax と対戦させて集計する"""
    half = num_games // 2
    wins = draws = losses = 0

    for i in range(half):
        result = play_game_vs_edax(ai_player, edax, BLACK)
        if result == "black":
            wins += 1
        elif result == "draw":
            draws += 1
        else:
            losses += 1
        print(f"\r  {opponent_name}: {i + 1}/{num_games} 局完了...", end="", flush=True)

    for i in range(half):
        result = play_game_vs_edax(ai_player, edax, WHITE)
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
# CSV ログ
# ---------------------------------------------------------------------------

def _append_csv_row(csv_path: str, row: dict) -> None:
    """CSV にデータ行を追記する。カラムが増えた場合はヘッダーごとファイルを書き直す。"""
    fieldnames = list(row.keys())

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerow(row)
        return

    # 既存ヘッダーを取得
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        existing_header = next(csv.reader(f), [])

    new_fields = [f for f in fieldnames if f not in existing_header]
    if not new_fields:
        # 同じカラム構成: 追記のみ
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_header, restval="", extrasaction="ignore")
            writer.writerow(row)
    else:
        # 新カラムが増加: ファイル全体をヘッダー更新して書き直す
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        all_fieldnames = existing_header + new_fields
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames, restval="", extrasaction="ignore")
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Othello AI 強さ評価")
    parser.add_argument("--engine", type=str, choices=["internal", "minimax"],
                        default=DEFAULTS["engine"],
                        help="評価するエンジン: 'internal'（MCTS+NN）または 'minimax'"
                             f"（bitboard alpha-beta） (default: {DEFAULTS['engine']})")
    parser.add_argument("--model-path", type=str, default=DEFAULTS["model_path"],
                        help=f"モデル重みファイルパス。--engine internal のみ使用 (default: {DEFAULTS['model_path']})")
    parser.add_argument("--games", type=int, default=DEFAULTS["games"],
                        help=f"ランダム・貪欲法との対戦局数（偶数推奨） (default: {DEFAULTS['games']})")
    parser.add_argument("--simulations", type=int, default=DEFAULTS["simulations"],
                        help=f"MCTS シミュレーション数/手。--engine internal のみ使用 (default: {DEFAULTS['simulations']})")
    parser.add_argument("--minimax-time-limit", type=float, default=DEFAULTS["minimax_time_limit"],
                        help="minimax の1手あたり探索時間予算（秒）。--engine minimax のみ使用 "
                             f"(default: {DEFAULTS['minimax_time_limit']})")
    parser.add_argument("--minimax-endgame-threshold", type=int,
                        default=DEFAULTS["minimax_endgame_threshold"],
                        help="minimax が完全読みに切り替える空きマス数。--engine minimax のみ使用 "
                             f"(default: {DEFAULTS['minimax_endgame_threshold']})")
    parser.add_argument("--edax-levels", type=int, nargs="+",
                        default=DEFAULTS["edax_levels"],
                        help=f"Edax の探索レベルリスト (default: {DEFAULTS['edax_levels']})")
    parser.add_argument("--edax-games", type=int, default=DEFAULTS["edax_games"],
                        help=f"Edax との対戦局数（偶数推奨） (default: {DEFAULTS['edax_games']})")
    parser.add_argument("--no-edax", action="store_true",
                        help="Edax との対戦をスキップする")
    parser.add_argument("--edax-binary", type=str, default=_EDAX_BINARY,
                        help=f"Edax バイナリパス (default: {_EDAX_BINARY})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.engine == "internal" and not os.path.exists(args.model_path):
        print(f"エラー: モデルファイルが見つかりません: {args.model_path}")
        print("先に python train/train.py で学習してください。")
        sys.exit(1)

    use_edax = not args.no_edax
    if use_edax and not os.path.exists(args.edax_binary):
        print(f"警告: Edax バイナリが見つかりません: {args.edax_binary}")
        print("  Edax との対戦をスキップします。--no-edax で警告を抑制できます。")
        use_edax = False

    print("=== Othello AI 評価 ===")
    print(f"  エンジン        : {args.engine}")
    if args.engine == "internal":
        print(f"  モデル          : {args.model_path}")
        print(f"  シミュレーション: {args.simulations}")
    else:
        print(f"  探索時間予算    : {args.minimax_time_limit}秒/手")
        print(f"  終盤完全読み閾値: 空きマス{args.minimax_endgame_threshold}以下")
    print(f"  対戦局数        : {args.games} 局/相手（ランダム・貪欲法）")
    if use_edax:
        print(f"  Edax レベル     : {args.edax_levels}")
        print(f"  Edax 対戦局数   : {args.edax_games} 局/レベル")
    print()

    run_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.engine == "internal":
        ai = AIPlayer(model_path=args.model_path, simulations=args.simulations)
    else:
        ai = MinimaxPlayer(
            time_limit=args.minimax_time_limit,
            endgame_empty_threshold=args.minimax_endgame_threshold,
        )

    results = {}

    # ランダム・貪欲法との対戦
    simple_opponents = [
        ("ランダム", RandomPlayer()),
        ("貪欲法",   GreedyPlayer()),
    ]
    for name, opponent in simple_opponents:
        print(f"vs {name}:")
        results[name] = run_match(ai, opponent, args.games, name)

    # Edax との対戦
    if use_edax:
        for level in args.edax_levels:
            name = f"Edax L{level}"
            edax = EdaxPlayer(level=level, binary=args.edax_binary)
            try:
                print(f"vs {name}:")
                results[name] = run_match_vs_edax(ai, edax, args.edax_games, name)
            finally:
                edax.close()

    print()
    print("=== 評価結果 ===")
    print(f"  {'相手':<12} {'勝率':>6}  {'勝':>4} {'分':>4} {'負':>4}")
    print(f"  {'-'*36}")
    for name, r in results.items():
        print(f"  {name:<12} {r['win_rate']:>6.1%}  {r['wins']:>4} {r['draws']:>4} {r['losses']:>4}")

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "othell_evaluete_log.csv")

    csv_row = {
        "run_datetime": run_dt,
        "engine": args.engine,
        "games_per_opponent": args.games,
    }
    if args.engine == "internal":
        csv_row["model_path"] = args.model_path
        csv_row["simulations"] = args.simulations
    else:
        csv_row["minimax_time_limit"] = args.minimax_time_limit
        csv_row["minimax_endgame_threshold"] = args.minimax_endgame_threshold
    if use_edax:
        csv_row["edax_games_per_level"] = args.edax_games
        csv_row["edax_levels"] = ",".join(map(str, args.edax_levels))

    for name, r in results.items():
        prefix = name
        csv_row[f"{prefix}_win_rate"] = round(r["win_rate"], 6)
        csv_row[f"{prefix}_wins"] = r["wins"]
        csv_row[f"{prefix}_draws"] = r["draws"]
        csv_row[f"{prefix}_losses"] = r["losses"]

    _append_csv_row(csv_path, csv_row)
    print(f"\n  ログ: {csv_path}")


if __name__ == "__main__":
    main()
