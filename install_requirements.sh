#!/usr/bin/env bash
# Two-stage install to avoid pip ResolutionTooDeep on macOS.
# Use a dedicated venv to avoid conflicts with conda/base (e.g. botocore/urllib3).
# Usage: ./install_requirements.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi
echo "Activating venv..."
source venv/bin/activate

echo "Stage 1/2: Installing core dependencies..."
pip install --upgrade pip -q
pip install -r requirements-core.txt

echo ""
echo "Stage 2/2: Installing ML/NLP/causal stack (BLIS_ARCH=generic for macOS)..."
BLIS_ARCH=generic pip install -r requirements-ml.txt

echo ""
echo "Done. Activate with: source venv/bin/activate"
echo "Optional: python -m spacy download en_core_web_sm"
