#!/bin/bash
# Open app in browser. Start servers first with: ./launch_app.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if command -v lsof >/dev/null 2>&1; then
  if ! lsof -ti :5006 >/dev/null 2>&1; then
    echo "❌ Nothing is running on port 5006. Start the app first: ./launch_app.sh"
    exit 1
  fi
fi

open "http://127.0.0.1:5006" 2>/dev/null || open "http://localhost:5006" 2>/dev/null || true
echo "✅ Opened http://127.0.0.1:5006"
