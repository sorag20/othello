#!/usr/bin/env bash
# RunPod Pod 上でセルフプレイ学習を実行するラッパースクリプト。
#
# tmux 等で SSH セッションから切り離しても継続できるようにし、
# 標準出力をログファイルにも保存する（tail -f で別セッションから進捗確認できる）。
#
# 使用例（Pod 上、backend/ ディレクトリの外からでも実行可能）:
#   bash scripts/train_on_pod.sh --iterations 50 --games-per-iter 200 --simulations 100
#   bash scripts/train_on_pod.sh --model-path models/weights/model.pt --no-gate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BACKEND_DIR"

mkdir -p models/weights models/logs

echo "=== デバイス確認 ==="
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
echo

LOG_FILE="models/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "ログファイル: $LOG_FILE"
echo "(tmux から抜けても学習は継続します。別セッションから 'tail -f ${BACKEND_DIR}/${LOG_FILE}' で進捗確認できます)"
echo

python train/train.py "$@" 2>&1 | tee "$LOG_FILE"
