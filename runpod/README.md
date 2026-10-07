# RunPod での学習手順

セルフプレイ学習（`backend/train/train.py`）を [RunPod](https://www.runpod.io/) の GPU Pod 上で実行するための手順。

## 先に知っておくこと: GPU はどこで効くか

MCTS (`backend/ai/mcts.py`) はもともと1シミュレーションにつき盤面1枚ずつニューラルネットに推論をかける実装（バッチサイズ1の逐次呼び出し）だったため、セルフプレイ生成フェーズはGPUを強力にしても速くならない問題があった。これは解消済みで、`--parallel-games N` を指定すると複数局を同時に進め、各局のMCTSリーフ評価をラウンドごとにまとめて1回のバッチ推論にする（`NeuralNetwork.predict_batch` / `MCTS.run_batched`）。探索アルゴリズム自体は変えていないため対局の質は変わらない。

実用上の結論:

- RunPodで学習する際は `--parallel-games 32`~`128` 程度を指定するとGPUのバッチ推論を活用できる（VRAMに余裕があればもっと大きくてもよい。1局あたりの盤面は8×8と小さいのでVRAM消費は小さい）。
- ローカルCPUで少数対局を試す分には `--parallel-games` を指定しない（デフォルト1 = 従来の逐次実行）ままで問題ない。
- 最上位GPU（A100等）に大きな投資をする効果は薄い。バッチ推論を使っても1バッチの計算量自体は小さいネットワークなので、**RTX A4000 / RTX 3090 / RTX 4090 あたりの安価なGPUで十分**なことが多い。
- それでもRunPodを使う価値はある: (1) ローカルPCを長時間占有せず放置できる、(2) `--parallel-games` でのバッチ推論も学習ステップもGPUで素直に速くなる、(3) 別マシンで並行して対局・評価ができる。

## 事前準備

1. RunPod アカウントを作成し、支払い方法を登録する。
2. RunPod ダッシュボード → **Settings → SSH Public Keys** に、ローカルマシンの SSH 公開鍵（`~/.ssh/id_ed25519.pub` など）を登録する。
3. ローカルマシンに `rsync` があることを確認する（macOS には標準搭載）。

## 手順1: Network Volume を作成する（学習結果を永続化）

Pod を止めたり削除しても学習結果（モデル重み・ログ）を残すため、Network Volume を使う。

1. ダッシュボード → **Storage → Network Volumes → New Network Volume**
2. サイズは **10GB** で十分（モデル1つ約11MB、既存の全チェックポイントを合計しても200MB弱、ログもテキストで軽量。依存関係はPodのコンテナディスク側に入るためVolumeを圧迫しない）
3. GPU Pod を作る際に選ぶデータセンターと同じリージョンを選ぶ必要がある（先にGPUの空きがあるリージョンを確認してから作成するとよい）

## 手順2: GPU Pod を作成する

1. ダッシュボード → **Pods → Deploy**
2. GPUを選択（前述の理由により、**RTX A4000 / RTX 3090 / RTX 4090** クラスの安価なもので開始を推奨。足りなければ後からより大きいGPUに乗り換えればよい）
3. Pod Template は **RunPod PyTorch**（CUDA + PyTorch が最初から入っている公式テンプレート）を選択
4. **Network Volume**: 手順1で作成したボリュームを選び、マウント先を `/workspace` にする
5. **On-Demand** で開始する（Spotは安いが中断されうる。学習は途中の重みが `model.pt` に保存されるため中断自体は致命的ではないが、まずはOn-Demandで慣れるのがおすすめ）
6. SSH アクセスが有効になっていることを確認してデプロイ

## 手順3: 接続情報を確認する

Pod が起動したら、ダッシュボードの Pod詳細 → **Connect → SSH** に表示される接続コマンドから、ホスト・ポートを控える。

```bash
ssh root@<POD_HOST> -p <POD_PORT> -i ~/.ssh/id_ed25519
```

一度手動で接続できることを確認しておく（`known_hosts` の初回確認もここで済ませる）。

## 手順4: ローカルからコードを転送する

このリポジトリはGitHubに push していないため、`rsync` で直接転送する。プロジェクトルートで:

```bash
export POD_HOST=<POD_HOST>
export POD_PORT=<POD_PORT>
export SSH_KEY=~/.ssh/id_ed25519   # 使う鍵が違う場合のみ指定

# コードを送る（models/weights の既存 .pt ファイルは送らない）
./runpod/sync_to_pod.sh

# ローカルに既存の model.pt があり、そこから続きを学習したい場合
./runpod/sync_to_pod.sh --with-model
```

`backend/ai`, `backend/game`, `backend/train`, `requirements*.txt` などが `/workspace/othello/backend/` に配置される。

## 手順5: Pod上で依存関係をインストールする

Pod に SSH 接続し、以下を実行する。

```bash
cd /workspace/othello/backend

# ローカルと完全に同じバージョンのPyTorch(CUDA)を入れる
# （テンプレート付属のバージョンのままでも動くことが多いが、再現性のため推奨）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements-train.txt

# GPUが認識されているか確認
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`False` が返る場合は、ドライバが新しいCUDAバージョンに対応していない可能性がある。`nvidia-smi` で表示されるCUDA Versionを確認し、`cu121` を `cu118` に変えて入れ直す。

## 手順6: 学習を実行する（tmuxでSSH切断に耐える）

```bash
apt-get update && apt-get install -y tmux   # 入っていない場合のみ

tmux new -s train
cd /workspace/othello/backend

bash scripts/train_on_pod.sh \
  --iterations 50 \
  --games-per-iter 200 \
  --simulations 100 \
  --parallel-games 64 \
  --model-path /workspace/othello/backend/models/weights/model.pt
```

`Ctrl+b` → `d` でtmuxセッションから抜けてもSSH接続を切断してよい。学習は継続する。

再接続:

```bash
ssh root@<POD_HOST> -p <POD_PORT> -i ~/.ssh/id_ed25519
tmux attach -t train
```

## 手順7: 進捗を確認する

tmuxにアタッチせず別セッションからも確認できる（`scripts/train_on_pod.sh` が標準出力をログファイルにも複製しているため）。

```bash
ssh root@<POD_HOST> -p <POD_PORT> -i ~/.ssh/id_ed25519 \
  "tail -f /workspace/othello/backend/models/logs/train_*.log"
```

## 手順8: 学習結果をローカルに取得する

学習が完了したら（または途中経過を確認したくなったら）、プロジェクトルートで:

```bash
export POD_HOST=<POD_HOST>
export POD_PORT=<POD_PORT>
export SSH_KEY=~/.ssh/id_ed25519

./runpod/sync_from_pod.sh
```

`backend/models/weights/` と `backend/models/logs/` に反映される。ローカルの `docker-compose restart backend` で新しいモデルが有効になる。

## 手順9: 後片付け（課金停止）

学習が終わったら **Pod を Terminate**（停止ではなく削除）してGPU課金を止める。Network Volume はGPUと独立して保持され、次回はそこに新しいPodを繋げば `model.pt` から再開できる。

再開する場合:

1. 同じ Network Volume を使って新しい Pod を作る（手順2〜3）
2. 前回終了時に Volume に残っている `model.pt` がそのまま使えるので、コード転送は `sync_to_pod.sh`（`--with-model` は不要、Volume側に既にある）
3. `scripts/train_on_pod.sh` を再実行（`--model-path` が既存ファイルを検出すると自動的に続きから学習する。`backend/train/train.py` 内 `NeuralNetwork(args.model_path)` の分岐を参照）

## コストの目安

- RTX A4000: 概ね $0.2〜0.3/時間帯
- RTX 3090 / 4090: 概ね $0.3〜0.5/時間帯
- Network Volume: 数GB〜数十GBなら月額で数十円〜数百円程度

実際の料金はRunPodダッシュボードの最新表示を確認すること（変動する）。
