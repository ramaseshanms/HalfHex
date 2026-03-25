#!/bin/bash
# Environment setup script for QNN HTP Runtime project
# Source this file: source setup_env.sh

set -e

echo "═══════════════════════════════════════"
echo "QNN HTP Runtime — Environment Setup"
echo "═══════════════════════════════════════"

# ── Step 1: Verify paths (set these to your actual paths) ─────────────────
export ANDROID_NDK=${ANDROID_NDK:-/path/to/ndk/r26b}
export QNN_SDK=${QNN_SDK:-/path/to/qnn-sdk}
export PATH=$QNN_SDK/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

echo "ANDROID_NDK: $ANDROID_NDK"
echo "QNN_SDK:     $QNN_SDK"

# ── Step 2: Python dependencies ───────────────────────────────────────────
echo ""
echo "Installing Python dependencies..."
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.51.0 optimum onnx onnxruntime onnxsim
pip install numpy pandas matplotlib seaborn huggingface_hub

# ── Step 3: Download model weights ────────────────────────────────────────
echo ""
echo "Downloading Qwen3-1.7B weights..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-1.7B',
    local_dir='./models/qwen3-1.7b-hf',
    ignore_patterns=['*.gguf', '*.bin']
)
"

echo ""
echo "═══════════════════════════════════════"
echo "Environment setup complete!"
echo "═══════════════════════════════════════"
