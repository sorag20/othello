# オセロ AI アプリケーション

複数種類のAIと対戦できるオセロWebアプリケーション。AlphaZero型AI（MCTS + ニューラルネット）、Edax（外部エンジン）、自作のbitboard αβ探索AI（Minimax）から対戦相手を選べる。

## アーキテクチャ

```
┌─────────────────┐     ┌──────────────────────────────┐     ┌───────────┐
│  Frontend       │     │  Backend                     │     │  Database │
│  (Vue.js)       │────▶│  (Flask + Python AI)         │────▶│  (MySQL)  │
│  Port 8080      │     │  Port 5000                   │     │  Port 3306│
│  Nginx proxy    │     │  MCTS+NN / Minimax(αβ) / Edax │     │           │
└─────────────────┘     └──────────────────────────────┘     └───────────┘
```

- **Frontend**: Vue.js 3 + Vite, Nginx でサーブ、`/api/*` をバックエンドにプロキシ
- **Backend**: Python Flask, SQLAlchemy ORM。AIは3種類切り替え可能:
  - `internal`: AlphaZero型AI（MCTS + PyTorchニューラルネット）
  - `edax`: [Edax](https://github.com/abulmo/edax-reversi) 外部エンジン
  - `minimax`: 自作のbitboard αβ探索AI（`backend/ai/minimax_engine.py`、外部依存なし）
- **Database**: MySQL 8.0, 対局履歴の永続化

## クイックスタート

```bash
# 全サービスを起動（初回はビルドに数分かかります）
docker-compose up --build

# バックグラウンドで起動
docker-compose up --build -d
```

起動後、ブラウザで http://localhost:8080 にアクセスしてゲームを開始できます。

### 初回起動時の注意

初回起動時にはAIモデルの重みファイルが存在しないため、AIは「最初の合法手を選ぶ」フォールバックモードで動作します。より強いAIと対戦するにはセルフプレイ学習を実行してください。

## AIの学習（セルフプレイ）

バックエンドコンテナ内でセルフプレイ学習を実行できます：

```bash
# バックエンドコンテナに入る
docker-compose exec backend bash

# Pythonでセルフプレイを実行
python -c "
from ai.neural_network import NeuralNetwork
from ai.self_play import SelfPlay, SelfPlayConfig

nn = NeuralNetwork()
config = SelfPlayConfig(
    num_games=100,       # 対局数
    simulations=800,     # MCTS シミュレーション数/手
    learning_rate=0.001,
    batch_size=64,
    epochs=10,
)
sp = SelfPlay(nn, config)
summary = sp.run()
print(f'Games: {summary.games_played}, Samples: {summary.total_samples}, Loss: {summary.final_loss:.4f}')
nn.save_weights('models/weights/model.pt')
print('Model saved. Restart the backend to use the trained AI.')
"
```

学習完了後、バックエンドを再起動するとAIが学習済みモデルを使用します：

```bash
docker-compose restart backend
```

## AI の評価

`backend/train/evaluate.py` でランダム・貪欲法・Edax との対戦勝率を計測し、AI の強さを定量評価できます。

### 基本的な使い方

```bash
# backend/ ディレクトリから実行
cd backend

# デフォルト設定で評価（ランダム・貪欲法・Edax L1/5/10 各 20 局）
python train/evaluate.py

# モデル・局数・シミュレーション数を指定
python train/evaluate.py \
  --model-path models/weights/model.pt \
  --games 200 \
  --simulations 200

# Edax レベルと局数を変更
python train/evaluate.py --edax-levels 1 5 10 15 20 --edax-games 40

# Edax なしで素早く評価（ランダム・貪欲法のみ）
python train/evaluate.py --no-edax

# minimax エンジンを評価（モデルファイル不要）
python train/evaluate.py --engine minimax --edax-levels 1 5 10
python train/evaluate.py --engine minimax --minimax-time-limit 1.0 --games 200
```

### オプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--engine` | `internal` | 評価するエンジン: `internal`（MCTS+NN）または `minimax`（bitboard αβ） |
| `--model-path` | 最新の `models/weights/*.pt` | 評価するモデルの重みファイル（`--engine internal` のみ） |
| `--games` | `100` | ランダム・貪欲法との対戦局数（偶数推奨） |
| `--simulations` | `100` | MCTS シミュレーション数/手（`--engine internal` のみ） |
| `--minimax-time-limit` | `3.0` | minimaxの1手あたり探索時間予算（秒）（`--engine minimax` のみ） |
| `--minimax-endgame-threshold` | `10` | minimaxが完全読みに切り替える空きマス数（`--engine minimax` のみ） |
| `--edax-levels` | `1 5 10` | Edax の探索レベル（複数指定可） |
| `--edax-games` | `20` | Edax との対戦局数/レベル（偶数推奨） |
| `--no-edax` | — | Edax との対戦をスキップ |
| `--edax-binary` | `../edax-reversi/bin/mEdax-*` | Edax バイナリのパス |

### Edax について

[Edax](https://github.com/abulmo/edax-reversi) はオセロの強力なリファレンスエンジンで、レベル（探索深度）を指定して強さを調整できます。

| レベル | 目安 | 1 局の時間 |
|---|---|---|
| 1 | 弱（練習用） | < 0.1 秒/局 |
| 5 | 中程度 | 数秒/局 |
| 10 | 強 | 数十秒/局 |
| 15–20 | 非常に強 | 数分/局 |

Edax を使用するには事前のビルドと `data/eval.dat` の配置が必要です。詳細は `../edax-reversi/README.md` を参照してください。

### 評価ログ

実行結果は `models/logs/othell_evaluete_log.csv` に追記されます。

```
=== 評価結果 ===
  相手           勝率    勝    分    負
  ----------------------------------
  ランダム       87.0%   87    0    13
  貪欲法         62.0%   62    0    38
  Edax L1        15.0%   15    0     5
  Edax L5         5.0%    5    0    15
  Edax L10        0.0%    0    0    20
```

## Minimax AI（αβ探索エンジン）

`backend/ai/minimax_engine.py` は、AlphaZero型AI（MCTS+NN）とは別系統の、外部バイナリやモデルファイルに依存しない自作AIエンジン。盤面を2つの64bit整数（自石/相手石のbitboard）で表現し、反復深化negamax + αβ枝刈り + 置換表で中盤を探索、空きマスが少なくなった終盤は完全読み（石差の厳密解）に切り替える。

- 環境変数 `MINIMAX_TIME_LIMIT`（デフォルト3.0秒）: 1手あたりの探索時間予算
- 環境変数 `MINIMAX_ENDGAME_EMPTY_THRESHOLD`（デフォルト10）: この空きマス数以下で完全読みに切り替え
- ゲーム作成時に `opponent_type: "minimax"` を指定すると対戦できる（外部エンジンと違い常に利用可能）

### CodinGame提出用ファイル

[`codingame/`](codingame/) ディレクトリに、CodinGameの「Othello」bot-programming用の単一C++ファイルを別途用意している。CodinGameは2ターン目以降150ms/手という厳しい時間制約があり、PyTorchによるニューラルネット推論よりもbitboard + αβ探索の方が有利という判断から、Web版のPython実装とは別にC++で移植したもの。詳細は [`codingame/README.md`](codingame/README.md) を参照（CodinGameの方針に従い、提出用のソースコードは公開リポジトリに含めず解説のみ掲載している）。

## クラウドでの学習（RunPod）

ローカルPCを長時間占有せずに学習したい場合、[RunPod](https://www.runpod.io/) のGPU Podでセルフプレイ学習を実行できる。手順・スクリプトは [`runpod/README.md`](runpod/README.md) を参照。

> 注: `train.py --parallel-games N` で複数局を同時に進めると、各局のMCTSリーフ評価をまとめてバッチ推論するためGPUを活用できる（詳細は同READMEを参照）。

## API エンドポイント

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/games` | 新規ゲーム作成 |
| GET | `/api/games/{id}` | ゲーム状態取得 |
| POST | `/api/games/{id}/moves` | 手を打つ (`{"row": 0-7, "col": 0-7}`) |
| GET | `/api/games` | 対局履歴一覧 (`?page=1&per_page=20`) |
| GET | `/api/games/{id}/replay` | リプレイデータ取得 |
| GET | `/api/health` | ヘルスチェック |

## 開発

### ローカル開発（Docker外）

```bash
# バックエンド
cd backend
pip install -r requirements.txt
python app.py

# フロントエンド
cd frontend
npm install
npm run dev
```

### テスト実行

```bash
# バックエンドテスト
cd backend
pytest tests/ -v

# フロントエンドテスト
cd frontend
npm run test
```

## 停止・クリーンアップ

```bash
# 停止
docker-compose down

# データベースボリュームも削除して完全クリーンアップ
docker-compose down -v
```

## 技術スタック

- **Frontend**: Vue.js 3, Pinia, Vue Router, Vite, Nginx
- **Backend**: Python 3.11, Flask, SQLAlchemy, PyTorch, gunicorn
- **AI**: AlphaZero (MCTS + ResNet, 800 simulations/move) / Minimax (bitboard + αβ探索 + 完全読み) / Edax
- **Database**: MySQL 8.0
- **Infrastructure**: Docker, docker-compose
