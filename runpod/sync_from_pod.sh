#!/usr/bin/env bash
# RunPod Pod で学習した重みとログをローカルに取得する。
#
# 使用例:
#   POD_HOST=123.45.67.89 POD_PORT=40022 ./runpod/sync_from_pod.sh
#
# 環境変数は sync_to_pod.sh と同じ
# （POD_HOST, POD_PORT, POD_USER, SSH_KEY, REMOTE_DIR, SSH_EXTRA_OPTS）。
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

mkdir -p "$BACKEND_DIR/models/weights" "$BACKEND_DIR/models/logs"

echo "==> 学習済みモデルを取得します"
rsync -avz --progress -e "$SSH_CMD" \
  "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/models/weights/" "$BACKEND_DIR/models/weights/"

echo "==> ログを取得します"
rsync -avz --progress -e "$SSH_CMD" \
  "${POD_USER}@${POD_HOST}:${REMOTE_DIR}/models/logs/" "$BACKEND_DIR/models/logs/"

echo "==> 取得完了: $BACKEND_DIR/models/weights/ , $BACKEND_DIR/models/logs/"
echo "    backend を再起動すると新しいモデルが有効になります (docker-compose restart backend)"
