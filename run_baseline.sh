#!/bin/bash
# Day 1: Build and run llama.cpp baseline on device
set -e

echo "═══════════════════════════════════════"
echo "llama.cpp Baseline Profiling"
echo "═══════════════════════════════════════"

# ── Step 1: Build llama.cpp for Android ──────────────────────────────────
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DGGML_NEON=ON \
  -DGGML_OPENMP=ON

cmake --build build-android --config Release -j$(nproc)
cd ..

# ── Step 2: Push to device ───────────────────────────────────────────────
adb push llama.cpp/build-android/bin/llama-cli /data/local/tmp/
adb shell "chmod +x /data/local/tmp/llama-cli"

echo "NOTE: You must push the Qwen3-1.7B-Q4_0.gguf model to /data/local/tmp/"
echo "  adb push /path/to/Qwen3-1.7B-Q4_0.gguf /data/local/tmp/"

# ── Step 3: CPU-only baseline ────────────────────────────────────────────
echo ""
echo "--- CPU Baseline (NEON only) ---"
adb shell "
  cd /data/local/tmp &&
  taskset 0xF0 ./llama-cli \
    -m Qwen3-1.7B-Q4_0.gguf \
    -p 'Explain the theory of relativity in detail.' \
    -n 200 \
    --threads 4 \
    --ctx-size 512 \
    -ngl 0 \
    --no-mmap \
    2>&1 | grep -E '(eval time|load time|sample time|prompt eval)'
"

# ── Step 4: Vulkan baseline ─────────────────────────────────────────────
echo ""
echo "--- Vulkan Baseline ---"
adb shell "
  cd /data/local/tmp &&
  taskset 0xF0 ./llama-cli \
    -m Qwen3-1.7B-Q4_0.gguf \
    -p 'Explain the theory of relativity in detail.' \
    -n 200 \
    --threads 4 \
    --ctx-size 512 \
    -ngl 99 \
    --no-mmap \
    2>&1 | grep -E '(eval time|load time|sample time|prompt eval)'
"

echo ""
echo "═══════════════════════════════════════"
echo "Record these numbers in baselines/llamacpp_baseline.txt"
echo "═══════════════════════════════════════"
