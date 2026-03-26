#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
API_HOST="${LUCKYMODELS_API_HOST:-127.0.0.1}"
REQUESTED_API_PORT="${LUCKYMODELS_API_PORT:-38080}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Virtual environment not found: $VENV_PY"
  echo "Please create it first."
  exit 1
fi

API_PORT="$("$VENV_PY" -c "import socket, sys; host=sys.argv[1]; port=int(sys.argv[2]); sock=socket.socket();
try:
    sock.bind((host, port))
    print(port)
except OSError:
    fallback=socket.socket()
    fallback.bind((host, 0))
    print(fallback.getsockname()[1])
    fallback.close()
finally:
    sock.close()" "$API_HOST" "$REQUESTED_API_PORT")"

if [[ "$API_PORT" != "$REQUESTED_API_PORT" ]]; then
  echo "Requested API port ${REQUESTED_API_PORT} is busy, switched to ${API_PORT}"
fi

cd "$ROOT_DIR/apps/api"
echo "Starting LuckyModels API dashboard on ${API_HOST}:${API_PORT}"
exec "$ROOT_DIR/.venv/bin/uvicorn" main:app --host "$API_HOST" --port "$API_PORT"
