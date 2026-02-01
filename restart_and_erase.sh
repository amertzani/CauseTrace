#!/bin/bash
# Stop app, erase all persistent data, then restart for a clean test.
# Run from project root: ./restart_and_erase.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "🛑 Stopping backend and frontend..."
for port in 8001 5006; do
  if command -v lsof >/dev/null 2>&1; then
    PIDS=$(lsof -ti :$port 2>/dev/null) || true
    if [ -n "$PIDS" ]; then
      echo "  Stopping port $port (PIDs: $PIDS)"
      echo "$PIDS" | xargs kill 2>/dev/null || true
      sleep 2
    fi
  fi
done
echo ""

echo "🧹 Erasing all persistent data..."
for name in knowledge_graph.pkl knowledge_backup.json documents_store.json causal_graphs_store.json; do
  path="$SCRIPT_DIR/$name"
  if [ -f "$path" ]; then
    rm -f "$path"
    echo "  Removed $name"
  fi
done
echo "✅ All data erased."
echo ""

echo "🚀 Starting app (clean state)..."
exec "$SCRIPT_DIR/launch_app.sh"
