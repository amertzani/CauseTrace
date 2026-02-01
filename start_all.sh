#!/bin/bash

# Start Both Backend and Frontend Servers
# This script starts both servers in separate terminal windows/tabs

set -e  # Exit on error

echo "🚀 Starting NesyX Application..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is optimized for macOS"
    echo "   On other systems, you may need to start servers manually"
fi

# Ports (use env vars if set)
BACKEND_PORT=${API_PORT:-8001}
FRONTEND_PORT=${PORT:-5006}

# Clean ports so nothing is left from previous runs
clean_port() {
    local port=$1
    local pids
    pids=$(lsof -ti :$port -sTCP:LISTEN 2>/dev/null) || true
    if [ -n "$pids" ]; then
        echo "🧹 Killing process(es) on port $port: $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

echo "🧹 Cleaning ports ${BACKEND_PORT} and ${FRONTEND_PORT}..."
clean_port $BACKEND_PORT
clean_port $FRONTEND_PORT
echo ""

# Start backend in a new terminal window (macOS)
echo "🔧 Starting backend server..."
osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && ./start_backend.sh\""

# Wait a moment for backend to start
sleep 3

# Start frontend in a new terminal window (macOS)
echo "🔧 Starting frontend server..."
osascript -e "tell application \"Terminal\" to do script \"cd '$SCRIPT_DIR' && ./start_frontend.sh\""

echo ""
echo "✅ Both servers are starting in separate terminal windows"
echo ""
echo "📋 Server URLs:"
echo "   Backend API:  http://localhost:${BACKEND_PORT}"
echo "   Backend Docs: http://localhost:${BACKEND_PORT}/docs"
echo "   Frontend:     http://localhost:${FRONTEND_PORT}"
echo ""
echo "💡 To stop the servers, close the terminal windows or press Ctrl+C in each"
echo ""

