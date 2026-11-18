#!/usr/bin/env bash
# Serve one pinned student model on loopback for the rule-distillation pilot.
# Usage: scripts/rule_pilot_serve.sh <model-key> [port]   (keys: config/rule_pilot.lock.json)
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="${1:?model key required}"
PORT="${2:-8081}"
FILE="$(python3 -c "import json,sys; print(json.load(open('$PROJECT_DIR/config/rule_pilot.lock.json'))['models'][sys.argv[1]]['filename'])" "$KEY")"
SERVER_BIN="$(cat "$PROJECT_DIR/runtime/server_path.txt")"
export LD_LIBRARY_PATH="$(dirname "$SERVER_BIN")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$SERVER_BIN" \
  --model "$PROJECT_DIR/models/$FILE" \
  --alias "$KEY" --host 127.0.0.1 --port "$PORT" \
  --ctx-size "${STUDENT_CTX:-8192}" --threads "${STUDENT_THREADS:-4}" --threads-batch "${STUDENT_THREADS:-4}" \
  --n-gpu-layers 0 --parallel "${STUDENT_PARALLEL:-1}" --jinja --no-webui
