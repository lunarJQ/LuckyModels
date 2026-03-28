#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="$ROOT_DIR/.venv/bin"
RUNTIME_HOST="${LUCKYMODELS_RUNTIME_HOST:-127.0.0.1}"
REQUESTED_RUNTIME_PORT="${LUCKYMODELS_RUNTIME_PORT:-38081}"

if [[ ! -x "$VENV_BIN/python" ]]; then
  echo "Virtual environment not found: $VENV_BIN/python"
  echo "Please create it first."
  exit 1
fi

RUNTIME_PORT="$("$VENV_BIN/python" -c "import socket, sys; host=sys.argv[1]; port=int(sys.argv[2]); sock=socket.socket(); 
try:
    sock.bind((host, port))
    print(port)
except OSError:
    fallback=socket.socket()
    fallback.bind((host, 0))
    print(fallback.getsockname()[1])
    fallback.close()
finally:
    sock.close()" "$RUNTIME_HOST" "$REQUESTED_RUNTIME_PORT")"

if [[ "$RUNTIME_PORT" != "$REQUESTED_RUNTIME_PORT" ]]; then
  echo "Requested runtime port ${REQUESTED_RUNTIME_PORT} is busy, switched to ${RUNTIME_PORT}"
fi

export LUCKYMODELS_MANIFEST_PATH="$ROOT_DIR/artifacts/models/casrel/en/v1/manifest.yaml"
export LUCKYMODELS_PROFILE_PATH="$ROOT_DIR/artifacts/models/casrel/en/v1/profiles/default.yaml"

cd "$ROOT_DIR"
echo "Starting LuckyModels runtime on ${RUNTIME_HOST}:${RUNTIME_PORT}"
"$VENV_BIN/uvicorn" runtime.server:app --reload --host "$RUNTIME_HOST" --port "$RUNTIME_PORT"
