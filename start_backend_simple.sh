#!/bin/bash
# Start backend only. Stops any process on 8001 first so restart works.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -ti :8001 2>/dev/null) || true
  if [ -n "$PIDS" ]; then
    echo "Stopping existing backend on port 8001 (PIDs: $PIDS)..."
    echo "$PIDS" | xargs kill 2>/dev/null || true
    sleep 2
  fi
fi

"$SCRIPT_DIR/venv/bin/python3" "$SCRIPT_DIR/api_server.py"
