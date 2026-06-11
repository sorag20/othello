# Implementation Plan: オセロAIアプリケーション

## Overview

AlphaZeroアーキテクチャベースのAIと対戦可能なオセロWebアプリケーションの実装計画。バックエンド（Python Flask）、AIエンジン（PyTorch）、フロントエンド（Vue.js）、データベース（MySQL）、Docker環境の順に段階的に構築する。

## Tasks

- [x] 1. プロジェクト構造とDocker環境のセットアップ
  - [x] 1.1 バックエンドのプロジェクト構造を作成
    - `backend/` ディレクトリ配下にFlaskアプリケーションの基本構造を作成
    - `app.py`, `config.py`, `requirements.txt` を作成
    - `routes/`, `services/`, `models/`, `game/`, `ai/`, `tests/` ディレクトリを作成
    - Flask, SQLAlchemy, PyTorch, Hypothesis 等の依存パッケージを `requirements.txt` に定義
    - _Requirements: 7.1, 7.4_

  - [x] 1.2 フロントエンドのプロジェクト構造を作成
    - Vue.js + Vite でフロントエンドプロジェクトを初期化
    - `src/components/`, `src/views/`, `src/stores/`, `src/api/`, `src/utils/`, `src/router/` を作成
    - Pinia, Vue Router, Vitest 等の依存パッケージを設定
    - _Requirements: 7.3_

  - [x] 1.3 Docker環境の構成ファイルを作成
    - `docker-compose.yml` でフロントエンド、バックエンド、MySQL の3サービスを定義
    - 各サービスの `Dockerfile` を作成
    - ヘルスチェック、再起動ポリシー、依存関係の起動順序を設定
    - 環境変数（DB接続情報、APIベースURL）を定義
    - MySQLの名前付きボリュームを設定してデータ永続化
    - Nginx設定でAPIプロキシを構成
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10_

  - [x] 1.4 データベーススキーマとSQLAlchemyモデルを作成
    - `models/database.py` でDB初期化処理を実装
    - `models/game_session.py` で `GameSession` SQLAlchemyモデルを実装
    - MySQL初期化スクリプト（`init.sql`）を作成
    - _Requirements: 5.3, 7.5_

- [x] 2. ゲームロジックの実装
  - [x] 2.1 ボード状態管理クラスを実装
    - `game/constants.py` に定数（EMPTY=0, BLACK=1, WHITE=2）を定義
    - `game/board.py` に `Board` クラスを実装
    - 初期盤面生成、状態取得、石配置、合法手計算、石数カウント、コピーメソッドを実装
    - _Requirements: 2.1, 2.3, 2.6_

  - [ ]* 2.2 Property 1: 合法手の健全性と完全性テスト
    - **Property 1: 合法手の健全性と完全性**
    - Hypothesisで有効なオセロ盤面を生成し、`get_legal_moves`の返す手の集合が正しいことを検証
    - **Validates: Requirements 2.1, 2.6**

  - [x] 2.3 オセロルールエンジンを実装
    - `game/rules.py` に `OthelloRules` クラスを実装
    - `is_valid_move`, `get_flipped_stones`, `is_game_over`, `determine_winner` を実装
    - 8方向（水平、垂直、斜め）の石反転ロジックを実装
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 2.4 Property 2: 不正な手の拒否と状態保持テスト
    - **Property 2: 不正な手の拒否と状態保持**
    - 不正な位置に対して手が拒否され、盤面が変更されないことを検証
    - **Validates: Requirements 2.2**

  - [ ]* 2.5 Property 3: 石反転の正確性テスト
    - **Property 3: 石反転の正確性**
    - 合法手を打った後に反転される石の集合が正しく、それ以外の石が変更されないことを検証
    - **Validates: Requirements 2.3**

  - [ ]* 2.6 Property 4: パス処理の正確性テスト
    - **Property 4: パス処理の正確性**
    - 合法手がないプレイヤーのターンが正しく相手に渡されることを検証
    - **Validates: Requirements 2.4**

  - [ ]* 2.7 Property 5: ゲーム終了と勝者判定テスト
    - **Property 5: ゲーム終了と勝者判定**
    - 終了条件を満たす盤面で正しく勝者が判定されることを検証
    - **Validates: Requirements 2.5**

- [x] 3. チェックポイント - ゲームロジック検証
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. AIエンジンの実装
  - [x] 4.1 ニューラルネットワークモデルを実装
    - `ai/neural_network.py` に `OthelloNet` (PyTorch nn.Module) を実装
    - ResNet構造（残差ブロック10層、フィルター数128）を構築
    - Policy head（64出力、softmax）と Value head（スカラー出力、tanh）を実装
    - `NeuralNetwork` ラッパークラスで `predict`, `encode_board`, `load_weights`, `save_weights` を実装
    - _Requirements: 4.1, 4.2, 8.1, 8.2, 8.3_

  - [x] 4.2 盤面エンコーダを実装
    - `ai/board_encoder.py` で盤面を3×8×8テンソルにエンコードする処理を実装
    - Channel 0: 現在のプレイヤーの石、Channel 1: 相手の石、Channel 2: 手番情報
    - デコード処理（テンソル→盤面）も実装
    - _Requirements: 8.1, 8.4_

  - [ ]* 4.3 Property 6: 盤面エンコーディングのラウンドトリップテスト
    - **Property 6: 盤面エンコーディングのラウンドトリップ**
    - エンコード→デコードで元の盤面が復元されることを検証
    - **Validates: Requirements 8.1, 8.4**

  - [ ]* 4.4 Property 7: ニューラルネットワーク出力の不変条件テスト
    - **Property 7: ニューラルネットワーク出力の不変条件**
    - ポリシーベクトルのサイズ64、非合法手確率0、合法手確率合計1.0、バリュー[-1,1]を検証
    - **Validates: Requirements 4.2, 4.8, 8.2, 8.3, 8.5**

  - [x] 4.5 MCTSアルゴリズムを実装
    - `ai/mcts.py` に `MCTSNode` クラスと `MCTS` クラスを実装
    - PUCTアルゴリズムによるノード選択を実装
    - ニューラルネットワークによるリーフ評価を実装
    - バックプロパゲーション処理を実装
    - 訪問回数最多の手を選択するロジックを実装
    - シミュレーション回数の設定対応（デフォルト800）
    - _Requirements: 4.3, 4.4, 4.5_

  - [ ]* 4.6 Property 8: MCTS手選択の正確性テスト
    - **Property 8: MCTS手選択の正確性**
    - MCTS探索後に訪問回数最多のノードの手が選択されることを検証
    - **Validates: Requirements 4.5**

  - [x] 4.7 AIエンジンオーケストレータを実装
    - `ai/engine.py` に `AIEngine` クラスを実装
    - モデルロード、タイムアウト付き手計算（5秒制限）を実装
    - 非合法手マスキングと再正規化処理を実装
    - エラーハンドリング（ModelLoadError, AITimeoutError）を実装
    - _Requirements: 3.2, 3.3, 4.5, 4.6, 4.7, 4.8_

- [x] 5. チェックポイント - AIエンジン検証
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. バックエンドAPIの実装
  - [x] 6.1 ゲームサービス層を実装
    - `services/game_service.py` に `GameService` クラスを実装
    - 新規ゲーム作成、手の適用、AI手番処理、ゲーム状態取得のロジックを実装
    - パス判定と自動パス処理を実装
    - ゲーム終了判定と結果保存処理を実装
    - DB保存リトライ処理（最大3回）を実装
    - _Requirements: 2.4, 3.1, 3.2, 3.3, 3.4, 5.1, 5.5_

  - [x] 6.2 履歴サービス層を実装
    - `services/history_service.py` に `HistoryService` クラスを実装
    - 対局履歴の取得（新しい順、ページング）を実装
    - リプレイデータの取得と盤面再構築を実装
    - _Requirements: 5.2, 5.4, 5.6_

  - [ ]* 6.3 Property 9: リプレイ状態再構築の正確性テスト
    - **Property 9: リプレイ状態再構築の正確性**
    - 有効な手順シーケンスから各中間盤面が正しく再構築されることを検証
    - **Validates: Requirements 5.4**

  - [x] 6.4 APIルートを実装
    - `routes/game_routes.py` に全APIエンドポイントを実装
    - POST /api/games（新規ゲーム作成、201応答）
    - GET /api/games/{id}（ゲーム状態取得、200応答）
    - POST /api/games/{id}/moves（手を打つ、200応答）
    - GET /api/games（対局履歴一覧、200応答）
    - GET /api/games/{id}/replay（リプレイデータ取得、200応答）
    - リクエストバリデーションとエラーハンドリングを実装
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x] 6.5 Flask アプリケーションファクトリを完成
    - `app.py` にアプリケーションファクトリパターンを実装
    - CORS設定、DB初期化、ルート登録、エラーハンドラを設定
    - グローバル例外ハンドラで内部詳細を非公開にする処理を実装
    - _Requirements: 6.5, 6.8_

  - [ ]* 6.6 バックエンドAPIのユニットテスト
    - pytest で全APIエンドポイントの正常系・異常系テストを作成
    - 不正な手、存在しないゲームID、パラメータ不正のエラー処理テスト
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

- [x] 7. チェックポイント - バックエンドAPI検証
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. フロントエンドの実装
  - [x] 8.1 ゲームストアとAPIクライアントを実装
    - `stores/gameStore.js` にPiniaストアを実装（ゲーム状態管理）
    - `api/gameApi.js` にAPIクライアントを実装（全エンドポイント対応）
    - エラーハンドリングとローディング状態管理を実装
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 8.2 ゲームボードコンポーネントを実装
    - `components/GameBoard.vue` に8×8ボードの描画を実装
    - `components/BoardCell.vue` に個別セル（石、合法手表示）を実装
    - 初期盤面表示（中央に黒2白2）を実装
    - 合法手の視覚的表示（配置インジケータ）を実装
    - 石配置クリックイベントの処理を実装
    - 不正な手クリック時の視覚的フィードバック（1秒以上表示）を実装
    - AI手番時の入力無効化を実装
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.7_

  - [x] 8.3 ゲームステータスとUI補助コンポーネントを実装
    - `components/GameStatus.vue` にターン表示、石カウント表示を実装
    - `components/ThinkingIndicator.vue` にAI思考中インジケータを実装
    - `components/GameResult.vue` に対局結果表示と新規ゲーム開始ボタンを実装
    - パス通知表示を実装
    - _Requirements: 1.5, 1.6, 1.8, 3.5, 3.6_

  - [x] 8.4 ゲーム画面のビューと統合を実装
    - `views/GameView.vue` にゲーム画面全体のレイアウトと状態管理を実装
    - プレイヤーの手番処理（クリック→API→ボード更新、500ms以内）を実装
    - AIの手番処理（自動トリガー→思考中表示→結果反映）を実装
    - ゲーム終了処理と結果表示を実装
    - _Requirements: 1.3, 3.1, 3.4, 3.5, 3.6_

  - [x] 8.5 対局履歴とリプレイ機能を実装
    - `components/GameHistory.vue` に対局履歴一覧を実装
    - `views/HistoryView.vue` に履歴画面を実装
    - リプレイ機能（1手ずつ前進・後退）を実装
    - `router/index.js` にルーティング設定を追加
    - _Requirements: 5.2, 5.4_

  - [ ]* 8.6 フロントエンドのユニットテスト
    - Vitest + Vue Test Utils でコンポーネントテストを作成
    - ゲームボードの表示、石配置、合法手表示のテスト
    - ゲームフロー（新規ゲーム→手の応酬→終了）のテスト
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 9. チェックポイント - フロントエンド検証
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. セルフプレイ学習機能の実装
  - [x] 10.1 セルフプレイモジュールを実装
    - `ai/self_play.py` に `SelfPlay` クラスを実装
    - 同一モデル重みでの自己対戦によるデータ生成を実装
    - 各手番で学習サンプル（盤面テンソル、MCTSポリシー、ゲーム結果）を記録
    - 中断時の部分データ保持（完了ゲームのサンプルのみ保持）を実装
    - _Requirements: 9.1, 9.2, 9.6_

  - [x] 10.2 学習処理を実装
    - ポリシークロスエントロピー + バリューMSEの結合損失関数を実装
    - パラメータ設定可能な学習処理（学習率、バッチサイズ、エポック数）を実装
    - 学習完了時のモデル重み保存を実装
    - 学習サマリー（ゲーム数、サンプル数、最終損失値）の出力を実装
    - パラメータの有効範囲バリデーションを実装
    - _Requirements: 9.3, 9.4, 9.5, 9.7_

  - [ ]* 10.3 Property 10: 学習サンプルの妥当性テスト
    - **Property 10: 学習サンプルの妥当性**
    - 盤面テンソル形状(3,8,8)、ポリシー合計1.0、結果値{-1,0,1}を検証
    - **Validates: Requirements 9.2**

  - [ ]* 10.4 Property 11: セルフプレイパラメータの範囲検証テスト
    - **Property 11: セルフプレイパラメータの範囲検証**
    - 有効範囲内のパラメータは受理、範囲外は拒否されることを検証
    - **Validates: Requirements 9.5**

- [x] 11. 全体統合とDocker環境テスト
  - [x] 11.1 全コンポーネントの結合とDocker起動確認
    - docker-compose.yml の全サービスの起動確認
    - フロントエンド→バックエンド→DB の接続確認
    - AIモデルのロードと対戦動作の確認
    - セッション管理（切断後300秒保持）の動作確認
    - _Requirements: 3.7, 7.2, 7.7, 7.9_

  - [ ]* 11.2 統合テストの作成
    - API全エンドポイントのE2Eテストを作成
    - ゲーム作成→手の応酬→ゲーム終了→履歴確認のフルフローテスト
    - AIタイムアウト時の処理テスト
    - DB保存失敗時のリトライテスト
    - _Requirements: 3.2, 3.3, 5.1, 5.5_

- [x] 12. 最終チェックポイント - 全テスト合格確認
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- タスクに `*` マークが付いたものはオプション（スキップ可能）で、主にテスト関連のサブタスク
- 各タスクは具体的な要件に紐づき、トレーサビリティを確保
- チェックポイントで段階的に品質を検証
- Property テストは Hypothesis ライブラリを使用し、各プロパティは設計ドキュメントに対応
- ユニットテストは pytest（バックエンド）と Vitest（フロントエンド）を使用
- Docker環境は全サービスを統合して動作確認するために最後に全体テストを実施

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "1.4"] },
    { "id": 2, "tasks": ["2.1"] },
    { "id": 3, "tasks": ["2.2", "2.3"] },
    { "id": 4, "tasks": ["2.4", "2.5", "2.6", "2.7"] },
    { "id": 5, "tasks": ["4.1", "4.2"] },
    { "id": 6, "tasks": ["4.3", "4.4", "4.5"] },
    { "id": 7, "tasks": ["4.6", "4.7"] },
    { "id": 8, "tasks": ["6.1", "6.2"] },
    { "id": 9, "tasks": ["6.3", "6.4", "6.5"] },
    { "id": 10, "tasks": ["6.6"] },
    { "id": 11, "tasks": ["8.1"] },
    { "id": 12, "tasks": ["8.2", "8.3"] },
    { "id": 13, "tasks": ["8.4", "8.5"] },
    { "id": 14, "tasks": ["8.6"] },
    { "id": 15, "tasks": ["10.1"] },
    { "id": 16, "tasks": ["10.2"] },
    { "id": 17, "tasks": ["10.3", "10.4"] },
    { "id": 18, "tasks": ["11.1"] },
    { "id": 19, "tasks": ["11.2"] }
  ]
}
```
