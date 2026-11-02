#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_BIN="$(cat "$PROJECT_DIR/runtime/server_path.txt")"
export LD_LIBRARY_PATH="$(dirname "$SERVER_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$SERVER_BIN" \
  --model "$PROJECT_DIR/models/Qwen3-4B-Q4_K_M.gguf" \
  --alias Qwen3-4B --host 127.0.0.1 --port "${QWEN_PORT:-8080}" \
  --ctx-size 8192 --threads "${QWEN_THREADS:-4}" --threads-batch "${QWEN_THREADS:-4}" \
  --n-gpu-layers 0 --parallel 1 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' "$@"
