#!/bin/bash
# Launch NesyX: backend (8001) + frontend (5006), then open browser.
# Run from project root: ./launch_app.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Do not use set -e — we handle errors explicitly so one failure doesn't exit the script

echo "🚀 Launching NesyX (backend + frontend)..."
echo ""

# Free ports 8001 and 5006 so we can bind (ignore errors)
for port in 8001 5006; do
  if command -v lsof >/dev/null 2>&1; then
    PIDS=$(lsof -ti :$port 2>/dev/null) || true
    if [ -n "$PIDS" ]; then
      echo "Stopping process on port $port (PIDs: $PIDS)..."
      echo "$PIDS" | xargs kill 2>/dev/null || true
      sleep 2
    fi
  fi
done

# Start backend (use venv python directly so we don't rely on source in background)
echo "Starting backend on http://127.0.0.1:8001 ..."
"$SCRIPT_DIR/venv/bin/python3" "$SCRIPT_DIR/api_server.py" &
BACKEND_PID=$!
sleep 2

# Wait for backend (up to 60s — LLM load can be slow)
BACKEND_READY=
for i in $(seq 1 60); do
  if curl -s -o /dev/null --connect-timeout 2 http://127.0.0.1:8001/api/documents 2>/dev/null; then
    BACKEND_READY=1
    echo "✅ Backend ready"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "❌ Backend process died. Check logs above."
    exit 1
  fi
  sleep 1
done

if [ -z "$BACKEND_READY" ]; then
  echo "❌ Backend did not become ready in time."
  kill "$BACKEND_PID" 2>/dev/null || true
  exit 1
fi

# Start frontend
echo "Starting frontend on http://127.0.0.1:5006 ..."
export PORT=5006
(cd "$SCRIPT_DIR/RandDKnowledgeGraph" && npm run dev) &
FRONTEND_PID=$!
sleep 5

# Wait for frontend (up to 60s — Vite first start can be slow)
FRONTEND_READY=
for i in $(seq 1 60); do
  if curl -s -o /dev/null --connect-timeout 2 http://127.0.0.1:5006 2>/dev/null; then
    FRONTEND_READY=1
    echo "✅ Frontend ready"
    break
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "⚠️  Frontend process exited. Check RandDKnowledgeGraph (npm install? node version?)."
    break
  fi
  sleep 1
done

echo ""
if [ -n "$FRONTEND_READY" ]; then
  echo "🌐 Opening app in browser..."
  open "http://127.0.0.1:5006" 2>/dev/null || true
  echo "✅ App running: http://127.0.0.1:5006  (backend: http://127.0.0.1:8001)"
else
  echo "⚠️  Frontend not ready. When it is, open: http://127.0.0.1:5006"
fi
echo ""
echo "Press Ctrl+C to stop both servers."

cleanup() {
  echo ""
  echo "Stopping backend and frontend..."
  kill "$BACKEND_PID" 2>/dev/null || true
  kill "$FRONTEND_PID" 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

wait
