#!/bin/bash
# Setup script for SLM Pipeline

echo "🚀 Setting up SLM Pipeline..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install it first:"
    echo "   brew install ollama"
    exit 1
fi

echo "✅ Ollama found"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama is not running. Starting it..."
    echo "   Please run: ollama serve"
    exit 1
fi

echo "✅ Ollama is running"

# Pull Mistral model if not already available
echo "📥 Checking for Mistral model..."
if ! ollama list | grep -q "mistral"; then
    echo "   Downloading Mistral model (this may take a while)..."
    ollama pull mistral
fi

echo "✅ Mistral model available"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install --quiet -r requirements.txt

echo "✅ Dependencies installed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p slm_pipeline/memory/vector_db
mkdir -p slm_pipeline/memory/structured

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Place video files in phone_videos/raw/"
echo "   2. Run preprocessing (if not done): python Phase_b_preprocess.py && python phase_c_transcribe.py"
echo "   3. Run the pipeline: python slm_pipeline/orchestrator.py"
echo "   4. Query your memories: python slm_pipeline/cli.py query \"your question\""
echo ""
