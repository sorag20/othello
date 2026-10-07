#!/usr/bin/env bash
# ローカルの backend/ ディレクトリを RunPod Pod に同期する（学習コードの転送用）。
#
# 使用例:
#   POD_HOST=123.45.67.89 POD_PORT=40022 ./runpod/sync_to_pod.sh
#   POD_HOST=123.45.67.89 POD_PORT=40022 SSH_KEY=~/.ssh/id_ed25519 ./runpod/sync_to_pod.sh
#   POD_HOST=... POD_PORT=... ./runpod/sync_to_pod.sh --with-model   # ローカルの model.pt も送って続きから学習
#
# 必要な環境変数:
#   POD_HOST  RunPod ダッシュボードの Connect から取得した接続先ホスト（必須）
#   POD_PORT  同じく接続用ポート（省略時 22）
#   POD_USER  接続ユーザー（省略時 root。RunPod PyTorch テンプレートは通常 root）
#   SSH_KEY   使用する SSH 秘密鍵のパス（省略時デフォルトの鍵を使用）
#   REMOTE_DIR Pod 側の配置先（省略時 /workspace/othello/backend）
#   SSH_EXTRA_OPTS 追加のsshオプション（スペース区切り）。
#     `ssh.runpod.io` 経由のプロキシ接続で "doesn't support PTY" エラーが出る場合、
#     SSH_EXTRA_OPTS="-tt -q" を指定する（PTY強制割り当て + ログイン時のバナー抑制）。
set -euo pipefail

POD_HOST="${POD_HOST:?POD_HOST を指定してください（例: POD_HOST=1.2.3.4）}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/othello/backend}"
SSH_KEY="${SSH_KEY:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/../backend" && pwd)"

SSH_OPTS=(-p "$POD_PORT" -o StrictHostKeyChecking=accept-new)
if [ -n "$SSH_KEY" ]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi
if [ -n "${SSH_EXTRA_OPTS:-}" ]; then
  SSH_OPTS+=($SSH_EXTRA_OPTS)
fi
SSH_CMD="ssh ${SSH_OPTS[*]}"

echo "==> ${POD_USER}@${POD_HOST}:${POD_PORT} の ${REMOTE_DIR} にコードを同期します"
ssh "${SSH_OPTS[@]}" "${POD_USER}@${POD_HOST}" "mkdir -p '${REMOTE_DIR}/models/weights' '${REMOTE_DIR}/models/logs'"

rsync -avz --progress \
  -e "$SSH_CMD" \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.env' \
  --exclude '.DS_Store' \
  --exclude 'models/weights/*.pt' \
  --exclude 'models/logs/*' \
  "$BACKEND_DIR/" "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/"

if [[ "${1:-}" == "--with-model" ]]; then
  if [ -f "$BACKEND_DIR/models/weights/model.pt" ]; then
    echo "==> 既存の model.pt を送信します（続きから学習）"
    rsync -avz --progress -e "$SSH_CMD" \
      "$BACKEND_DIR/models/weights/model.pt" "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/models/weights/model.pt"
  else
    echo "警告: ${BACKEND_DIR}/models/weights/model.pt が見つかりません（Pod側はランダム初期化で開始されます）"
  fi
fi

chmod +x "$BACKEND_DIR/scripts/train_on_pod.sh" 2>/dev/null || true
echo "==> 同期完了"
