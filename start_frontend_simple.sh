#!/bin/bash
# Start frontend only. Stops any process on 5006 first so restart works.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PORT=${PORT:-5006}
if command -v lsof >/dev/null 2>&1; then
  PIDS=$(lsof -ti :$PORT 2>/dev/null) || true
  if [ -n "$PIDS" ]; then
    echo "Stopping existing frontend on port $PORT (PIDs: $PIDS)..."
    echo "$PIDS" | xargs kill 2>/dev/null || true
    sleep 2
  fi
fi

cd "$SCRIPT_DIR/RandDKnowledgeGraph" || exit 1
export PORT
npm run dev
