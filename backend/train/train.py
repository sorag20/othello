"""セルフプレイによる AI 学習スクリプト。

使用例:
    # backend/ ディレクトリから実行
    python train/train.py
    python train/train.py --iterations 20 --games-per-iter 20
    python train/train.py --model-path models/weights/model.pt
"""

import argparse
import copy
import csv
import os
import sys
import time
from datetime import datetime
from typing import List

# backend/ をモジュール検索パスに追加（直接実行時も import が通るよう）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mcts import MCTS
from ai.neural_network import NeuralNetwork
from ai.self_play import SelfPlay, SelfPlayConfig, TrainingSample
from game.board import Board
from game.constants import BLACK, WHITE

DEFAULTS = {
    "model_path": "models/weights/model.pt",
    "iterations": 50,
    "games_per_iter": 200,
    "simulations": 100,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "buffer_size": 100000,
    "arena_games": 20,
    "promotion_threshold": 0.5,
    "temperature_moves": 20,
    "parallel_games": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Othello AI セルフプレイ学習")
    parser.add_argument("--iterations", type=int, default=DEFAULTS["iterations"], help=f"イテレーション数 (default: {DEFAULTS['iterations']})")
    parser.add_argument("--games-per-iter", type=int, default=DEFAULTS["games_per_iter"], help=f"1イテレーションあたりの対局数 (default: {DEFAULTS['games_per_iter']})")
    parser.add_argument("--simulations", type=int, default=DEFAULTS["simulations"], help=f"MCTS シミュレーション数/手 (1–10000, default: {DEFAULTS['simulations']})")
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"], help=f"学習率 (0.0001–0.1, default: {DEFAULTS['learning_rate']})")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"], help=f"バッチサイズ (1–1024, default: {DEFAULTS['batch_size']})")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"], help=f"エポック数 (1–1000, default: {DEFAULTS['epochs']})")
    parser.add_argument("--buffer-size", type=int, default=DEFAULTS["buffer_size"], help=f"リプレイバッファの最大サンプル数。直近N件を保持して学習に使う (default: {DEFAULTS['buffer_size']})")
    parser.add_argument(
        "--temperature-moves",
        type=int,
        default=DEFAULTS["temperature_moves"],
        help=(
            "自己対戦で確率的に手を選ぶ手数。これを超えた手数以降は最も訪問数の多い手を"
            f"決定的に選び、終盤の学習ターゲットのノイズを減らす (0-120, default: {DEFAULTS['temperature_moves']})"
        ),
    )
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=DEFAULTS["parallel_games"],
        help=(
            "自己対戦で同時に進めるゲーム数。1より大きい値にすると、"
            "MCTSのリーフ評価を複数ゲーム分まとめてバッチ推論するため"
            f"GPU利用効率が上がる (1-256, default: {DEFAULTS['parallel_games']})"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULTS["model_path"],
        help=f"モデル重みファイルパス (default: {DEFAULTS['model_path']})",
    )
    parser.add_argument(
        "--arena-games",
        type=int,
        default=DEFAULTS["arena_games"],
        help=f"昇格判定用のアリーナ対戦局数（偶数推奨） (default: {DEFAULTS['arena_games']})",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=DEFAULTS["promotion_threshold"],
        help=f"model_path を上書きするために必要なアリーナ勝率 (default: {DEFAULTS['promotion_threshold']})",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="ゲート処理を無効化し、毎イテレーション無条件で model_path を上書きする（旧動作）",
    )
    return parser.parse_args()


def _append_csv_row(csv_path: str, row: dict) -> None:
    """CSV にデータ行を追記する。カラムが増えた場合はヘッダーごとファイルを書き直す。

    単純に既存ファイルの有無だけでヘッダー要否を判定すると、後からフィールドを
    追加した際に新しい列数のデータ行が古いヘッダーの下に無言で追記され、
    全カラムが1つずつずれてしまう（実際に発生していた）。
    """
    fieldnames = list(row.keys())

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        existing_header = next(csv.reader(f), [])

    new_fields = [f for f in fieldnames if f not in existing_header]
    if not new_fields:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_header, restval="", extrasaction="ignore")
            writer.writerow(row)
    else:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
        all_fieldnames = existing_header + new_fields
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames, restval="", extrasaction="ignore")
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow(row)


def _arena_battle(candidate: NeuralNetwork, incumbent: NeuralNetwork, num_games: int, simulations: int) -> float:
    """candidate と incumbent を対戦させ、candidate の得点率を返す（勝=1.0, 引=0.5, 負=0.0）。

    先手/後手を半分ずつ交代することで、手番による有利さの偏りを避ける。
    """
    candidate_mcts = MCTS(neural_network=candidate, simulations=simulations)
    incumbent_mcts = MCTS(neural_network=incumbent, simulations=simulations)
    half = num_games // 2

    score = 0.0
    for i in range(num_games):
        candidate_color = BLACK if i < half else WHITE
        board = Board()
        current = BLACK
        consecutive_passes = 0

        while True:
            legal_moves = board.get_legal_moves(current)
            if not legal_moves:
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
                current = WHITE if current == BLACK else BLACK
                continue

            consecutive_passes = 0
            mcts = candidate_mcts if current == candidate_color else incumbent_mcts
            move = mcts.search(board, current)
            board.place_stone(move[0], move[1], current)

            if board.is_full():
                break
            current = WHITE if current == BLACK else BLACK

        counts = board.count_stones()
        if counts["black"] == counts["white"]:
            score += 0.5
        else:
            winner = BLACK if counts["black"] > counts["white"] else WHITE
            if winner == candidate_color:
                score += 1.0

    return score / num_games


def main() -> None:
    args = parse_args()

    config = SelfPlayConfig(
        num_games=args.games_per_iter,
        simulations=args.simulations,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        temperature_moves=args.temperature_moves,
        parallel_games=args.parallel_games,
    )

    print("=== Othello AI 学習設定 ===")
    print(f"  イテレーション数     : {args.iterations}")
    print(f"  1イテレーションの対局: {args.games_per_iter}")
    print(f"  総対局数             : {args.iterations * args.games_per_iter}")
    print(f"  シミュレーション数   : {config.simulations}")
    print(f"  学習率               : {config.learning_rate}")
    print(f"  バッチサイズ         : {config.batch_size}")
    print(f"  エポック数           : {config.epochs}")
    print(f"  リプレイバッファ     : 最大 {args.buffer_size} サンプル")
    print(f"  温度スケジュール     : 最初の{config.temperature_moves}手は確率的選択、以降は決定的選択")
    print(f"  並列ゲーム数         : {config.parallel_games}" + ("（GPUバッチ推論有効）" if config.parallel_games > 1 else ""))
    print(f"  モデルパス           : {args.model_path}")
    print()

    # モデル初期化（model_path に既存モデルがあればそこから再開、なければランダム初期化）
    if os.path.exists(args.model_path):
        nn = NeuralNetwork(args.model_path)
        print(f"既存モデルから学習を再開します: {args.model_path}")
    else:
        nn = NeuralNetwork()
        print("ランダム初期化から学習を開始します。")

    print(f"デバイス: {nn.device}")
    if args.no_gate:
        print("ゲート処理: 無効（毎イテレーション無条件で model_path を上書き）")
    else:
        print(f"ゲート処理: 有効（アリーナ {args.arena_games}局, 昇格閾値 {args.promotion_threshold:.1%}）")
    print()

    # ゲート処理用: 直前に採用（昇格）済みの重みを保持する。
    # 新しいイテレーションの学習結果（candidate）がこれに対してアリーナ対戦で
    # 閾値以上の得点を取れない場合は model_path を上書きせず、
    # 次イテレーションの自己対戦もこの重みからやり直す。
    best_state_dict = copy.deepcopy(nn.model.state_dict())
    incumbent_nn = NeuralNetwork()
    incumbent_nn.model.load_state_dict(best_state_dict)
    incumbent_nn.model.eval()
    promotions = 0
    rejections = 0

    run_dt = datetime.now()
    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    model_filename = (
        f"model"
        f"_iter{args.iterations}"
        f"_games{args.games_per_iter}"
        f"_sim{args.simulations}"
        f"_lr{args.learning_rate}"
        f"_bs{args.batch_size}"
        f"_ep{args.epochs}"
        f"_{run_dt.strftime('%Y%m%d_%H%M%S')}.pt"
    )
    model_save_path = os.path.join(model_dir, model_filename)

    sp = SelfPlay(nn, config, model_save_path=model_save_path)
    total_start = time.time()

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "othell_training_log.csv")

    losses: list[float] = []

    # リプレイバッファ: 過去イテレーション分のサンプルも保持し、
    # 直近1イテレーションだけへの過学習・忘却を防ぐ
    replay_buffer: List[TrainingSample] = []

    for i in range(1, args.iterations + 1):
        print(f"=== イテレーション {i}/{args.iterations} ===")
        iter_start = time.time()

        # セルフプレイ
        sp_start = time.time()
        samples = sp.generate_games(args.games_per_iter)
        games_done = sp._games_completed if hasattr(sp, "_games_completed") else args.games_per_iter
        sp_elapsed = time.time() - sp_start
        print(f"  対局: {games_done}, サンプル: {len(samples)} ({sp_elapsed:.1f}s)")

        if not samples:
            print("  サンプルが収集できませんでした。スキップします。")
            continue

        # バッファに追加し、上限を超えた古いサンプルは破棄
        replay_buffer.extend(samples)
        if len(replay_buffer) > args.buffer_size:
            del replay_buffer[: len(replay_buffer) - args.buffer_size]
        print(f"  バッファ: {len(replay_buffer)}/{args.buffer_size} サンプル")

        # 学習（バッファ全体を使用）
        train_start = time.time()
        summary = sp.train(replay_buffer)
        train_elapsed = time.time() - train_start
        iter_elapsed = time.time() - iter_start
        print(f"  損失: {summary.final_loss:.4f} (学習: {train_elapsed:.1f}s, 合計: {iter_elapsed:.1f}s)")
        print()

        losses.append(summary.final_loss)

        if args.no_gate:
            best_state_dict = copy.deepcopy(nn.model.state_dict())
            nn.save_weights(args.model_path)
            print(f"  保存先（model_path）: {args.model_path}")
        else:
            # ゲート処理: 直前の採用済みモデル(incumbent)とアリーナ対戦し、
            # 閾値以上の得点を取れた場合のみ model_path を更新する（劣化防止）
            arena_start = time.time()
            win_rate = _arena_battle(nn, incumbent_nn, args.arena_games, config.simulations)
            arena_elapsed = time.time() - arena_start
            print(f"  アリーナ: candidate得点率 {win_rate:.1%} ({args.arena_games}局, {arena_elapsed:.1f}s)")

            if win_rate >= args.promotion_threshold:
                promotions += 1
                best_state_dict = copy.deepcopy(nn.model.state_dict())
                incumbent_nn.model.load_state_dict(best_state_dict)
                incumbent_nn.model.eval()
                nn.save_weights(args.model_path)
                print(f"  → 昇格: model_path を更新しました ({args.model_path})")
            else:
                rejections += 1
                nn.model.load_state_dict(best_state_dict)
                nn.model.eval()
                # 巻き戻し後のパラメータと矛盾するモーメントを引きずらないようoptimizerもリセット
                sp.reset_optimizer()
                print("  → 見送り: 閾値未満のため model_path は更新せず、直前の採用済み重みに復帰しました")
        print()

    # 最終的に採用済み(best)の重みを、今回の学習実行のスナップショットとしても保存する
    nn.model.load_state_dict(best_state_dict)
    nn.model.eval()
    nn.save_weights(model_save_path)

    total_elapsed = time.time() - total_start
    print(f"=== 完了 ({total_elapsed:.1f}s) ===")
    if not args.no_gate:
        print(f"  昇格: {promotions}回 / 見送り: {rejections}回")
    print(f"  保存先: {model_save_path}")

    if losses:
        csv_row = {
            "run_datetime": run_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "iterations": args.iterations,
            "games_per_iter": args.games_per_iter,
            "simulations": config.simulations,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "buffer_size": args.buffer_size,
            "total_sec": round(total_elapsed, 2),
            "first_loss": round(losses[0], 6),
            "final_loss": round(losses[-1], 6),
            "avg_loss": round(sum(losses) / len(losses), 6),
            "gate_enabled": not args.no_gate,
            "promotions": promotions if not args.no_gate else "",
            "rejections": rejections if not args.no_gate else "",
        }
        _append_csv_row(csv_path, csv_row)
        print(f"  ログ  : {csv_path}")

    print()
    print("バックエンドを再起動すると学習済みモデルが有効になります。")


if __name__ == "__main__":
    main()
