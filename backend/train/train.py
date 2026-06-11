"""セルフプレイによる AI 学習スクリプト。

使用例:
    # backend/ ディレクトリから実行
    python train/train.py
    python train/train.py --iterations 20 --games-per-iter 20
    python train/train.py --model-path models/weights/model.pt
"""

import argparse
import os
import sys
import time

# backend/ をモジュール検索パスに追加（直接実行時も import が通るよう）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.neural_network import NeuralNetwork, ModelLoadError
from ai.self_play import SelfPlay, SelfPlayConfig

DEFAULTS = {
    "model_path": "models/weights/model.pt",
    "iterations": 10,
    "games_per_iter": 10,
    "simulations": 100,
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Othello AI セルフプレイ学習")
    parser.add_argument("--iterations", type=int, default=DEFAULTS["iterations"], help=f"イテレーション数 (default: {DEFAULTS['iterations']})")
    parser.add_argument("--games-per-iter", type=int, default=DEFAULTS["games_per_iter"], help=f"1イテレーションあたりの対局数 (default: {DEFAULTS['games_per_iter']})")
    parser.add_argument("--simulations", type=int, default=DEFAULTS["simulations"], help=f"MCTS シミュレーション数/手 (1–10000, default: {DEFAULTS['simulations']})")
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"], help=f"学習率 (0.0001–0.1, default: {DEFAULTS['learning_rate']})")
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"], help=f"バッチサイズ (1–1024, default: {DEFAULTS['batch_size']})")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"], help=f"エポック数 (1–1000, default: {DEFAULTS['epochs']})")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULTS["model_path"],
        help=f"モデル重みファイルパス (default: {DEFAULTS['model_path']})",
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="既存の重みを読み込まずランダム初期化から学習する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SelfPlayConfig(
        num_games=args.games_per_iter,
        simulations=args.simulations,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    print("=== Othello AI 学習設定 ===")
    print(f"  イテレーション数     : {args.iterations}")
    print(f"  1イテレーションの対局: {args.games_per_iter}")
    print(f"  総対局数             : {args.iterations * args.games_per_iter}")
    print(f"  シミュレーション数   : {config.simulations}")
    print(f"  学習率               : {config.learning_rate}")
    print(f"  バッチサイズ         : {config.batch_size}")
    print(f"  エポック数           : {config.epochs}")
    print(f"  モデルパス           : {args.model_path}")
    print()

    # モデル初期化
    if not args.no_load and os.path.exists(args.model_path):
        print(f"既存の重みをロード中: {args.model_path}")
        try:
            nn = NeuralNetwork(model_path=args.model_path)
            print("ロード完了。継続学習を開始します。")
        except ModelLoadError as e:
            print(f"警告: ロードに失敗しました ({e})。ランダム初期化で開始します。")
            nn = NeuralNetwork()
    else:
        if args.no_load:
            print("--no-load 指定: ランダム初期化から学習を開始します。")
        else:
            print("既存の重みが見つかりません。ランダム初期化から学習を開始します。")
        nn = NeuralNetwork()

    print(f"デバイス: {nn.device}")
    print()

    sp = SelfPlay(nn, config)
    total_start = time.time()

    for i in range(1, args.iterations + 1):
        print(f"=== イテレーション {i}/{args.iterations} ===")

        # セルフプレイ
        start = time.time()
        samples = sp.generate_games(args.games_per_iter)
        games_done = sp._games_completed if hasattr(sp, "_games_completed") else args.games_per_iter
        print(f"  対局: {games_done}, サンプル: {len(samples)} ({time.time() - start:.1f}s)")

        if not samples:
            print("  サンプルが収集できませんでした。スキップします。")
            continue

        # 学習
        start = time.time()
        summary = sp.train(samples)
        print(f"  損失: {summary.final_loss:.4f} ({time.time() - start:.1f}s)")
        print()

    total_elapsed = time.time() - total_start
    print(f"=== 完了 ({total_elapsed:.1f}s) ===")
    print(f"  保存先: {DEFAULTS['model_path']}")
    print()
    print("バックエンドを再起動すると学習済みモデルが有効になります。")


if __name__ == "__main__":
    main()
