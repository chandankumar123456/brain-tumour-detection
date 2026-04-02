#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Brain Tumor Segmentation Dashboard – Startup Script
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
UVICORN="$VENV_DIR/bin/uvicorn"

echo "═══════════════════════════════════════════════════════"
echo "  🧠 Brain Tumor Segmentation Dashboard"
echo "  Multi-Path Fusion Network + Global Attention"
echo "  Anurag University · Wu et al. 2023"
echo "═══════════════════════════════════════════════════════"

# ── Virtual environment ──────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

echo "📦 Installing Python dependencies..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r "$BACKEND_DIR/requirements.txt"

echo "✅ Dependencies installed."

# ── Model Sanity Check ───────────────────────────────────────
echo ""
echo "🔎 Running model sanity check..."
cd "$BACKEND_DIR"
"$PYTHON" -c "
from model import MultiPathFusionNet
import torch
m = MultiPathFusionNet(in_channels=4, num_classes=4)
x = torch.randn(1, 4, 256, 256)
y = m(x)
params = sum(p.numel() for p in m.parameters())
print(f'  ✓ Model OK — Input: {tuple(x.shape)} → Output: {tuple(y.shape)}')
print(f'  ✓ Parameters: {params:,}')
"

# ── Check for trained weights ────────────────────────────────
echo ""
if [ -f "$BACKEND_DIR/models/best_model.pth" ]; then
  echo "✅ Trained weights found at backend/models/best_model.pth"
else
  echo "⚠️  No trained weights found. Run 'python train.py' first to train the model."
  echo "   The API will return an error until weights are available."
fi

echo ""
echo "🚀 Starting FastAPI server..."
echo "   Frontend: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
cd "$BACKEND_DIR"
"$UVICORN" main:app --reload --host 0.0.0.0 --port 8000
