# オセロ AI アプリケーション

AlphaZeroアーキテクチャベースのAIと対戦できるオセロWebアプリケーション。

## アーキテクチャ

```
┌─────────────────┐     ┌──────────────────────────────┐     ┌───────────┐
│  Frontend       │     │  Backend                     │     │  Database │
│  (Vue.js)       │────▶│  (Flask + PyTorch AI)        │────▶│  (MySQL)  │
│  Port 8080      │     │  Port 5000                   │     │  Port 3306│
│  Nginx proxy    │     │  MCTS + Neural Network       │     │           │
└─────────────────┘     └──────────────────────────────┘     └───────────┘
```

- **Frontend**: Vue.js 3 + Vite, Nginx でサーブ、`/api/*` をバックエンドにプロキシ
- **Backend**: Python Flask, AlphaZero AI (PyTorch), SQLAlchemy ORM
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
- **AI**: AlphaZero (MCTS + ResNet), 800 simulations/move
- **Database**: MySQL 8.0
- **Infrastructure**: Docker, docker-compose
