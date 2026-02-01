#!/bin/bash

# macOS Setup Script for NesyX Knowledge Graph Application
# This script sets up the Python environment and installs dependencies

# Don't exit on error for optional packages - we'll handle errors individually
set +e

echo "🚀 Setting up NesyX for macOS..."
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    echo "   You can install it using Homebrew: brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Found Python $PYTHON_VERSION"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Installing pip..."
    python3 -m ensurepip --upgrade
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📥 Installing Python dependencies..."
echo "   This may take several minutes, especially for ML libraries..."

# Install basic dependencies first (critical - fail if these don't install)
set -e
echo "   Installing core dependencies..."
pip install gradio>=4.0.0
pip install rdflib>=6.0.0
pip install huggingface_hub
pip install PyPDF2
pip install python-docx
pip install pandas
pip install networkx
pip install matplotlib
pip install plotly
export BLIS_ARCH=generic
pip install numpy
pip install fastapi
pip install uvicorn[standard]
pip install pydantic
pip install python-multipart
set +e  # Allow failures for optional packages

# Install ML dependencies (these may take longer and may fail on some systems)
echo "   Installing ML dependencies (this may take a while)..."
echo "   Installing transformers..."
pip install transformers>=4.30.0 || echo "   ⚠️  transformers installation failed (optional)"
echo "   Installing torch..."
pip install torch>=2.0.0 || echo "   ⚠️  torch installation failed (optional)"
echo "   Installing accelerate..."
pip install accelerate>=0.20.0 || echo "   ⚠️  accelerate installation failed (optional)"

# Note: bitsandbytes may not work on macOS (it's primarily for CUDA)
# Skip it if installation fails
echo "   Installing bitsandbytes (may fail on macOS - that's okay)..."
pip install bitsandbytes>=0.41.0 || echo "   ⚠️  bitsandbytes installation skipped (not critical for macOS)"

# Install NLP libraries
echo "   Installing NLP libraries..."
echo "   Installing clean-text..."
pip install clean-text>=0.6.0 || echo "   ⚠️  clean-text installation skipped"

# Check for Xcode Command Line Tools (needed for spaCy compilation on macOS)
echo "   Checking for build tools..."
if ! xcode-select -p &> /dev/null; then
    echo "   ⚠️  Xcode Command Line Tools not found"
    echo "   💡 Installing spaCy may fail. You can install build tools with:"
    echo "      xcode-select --install"
    echo "   Continuing anyway..."
fi

# Try to install spaCy - this often fails on macOS, especially Apple Silicon
echo "   Installing spaCy (this may fail on macOS - that's okay, it's optional)..."
SPACY_INSTALLED=false

# Try installing spaCy with pip
if export BLIS_ARCH=generic && pip install spacy>=3.7.0; then
    SPACY_INSTALLED=true
    echo "   ✅ spaCy installed successfully"
    
    # Try to download the language model
    echo "   Downloading spaCy English language model..."
    if python3 -m spacy download en_core_web_sm; then
        echo "   ✅ spaCy model downloaded successfully"
    else
        echo "   ⚠️  spaCy model download failed (can be done later with: python3 -m spacy download en_core_web_sm)"
    fi
else
    echo "   ⚠️  spaCy installation failed (this is common on macOS)"
    echo "   💡 spaCy is optional - the app will work without it"
    echo "   💡 To install spaCy later, try:"
    echo "      1. Install Xcode Command Line Tools: xcode-select --install"
    echo "      2. Then: pip install spacy && python3 -m spacy download en_core_web_sm"
    echo "   Or use conda: conda install -c conda-forge spacy"
fi

# Note: neuralcoref, OpenNRE, REL, and blink-lite may have compatibility issues
# Install them but don't fail if they don't work
echo "   Installing optional NLP libraries..."
pip install neuralcoref>=4.0.0 || echo "   ⚠️  neuralcoref installation skipped"
pip install OpenNRE>=2.0.0 || echo "   ⚠️  OpenNRE installation skipped"
pip install REL>=2.0.0 || echo "   ⚠️  REL installation skipped"
pip install blink-lite>=0.1.0 || echo "   ⚠️  blink-lite installation skipped"

echo ""
echo "✅ Python dependencies installed!"
echo ""
echo "📋 Next steps:"
echo "   1. Install Node.js dependencies: cd RandDKnowledgeGraph && npm install"
echo "   2. Run the app: ./launch_app.sh"
echo ""

