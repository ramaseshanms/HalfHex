#!/bin/bash
# ============================================================================
# HalfHex Baseline Profiling — llama.cpp on Nothing Phone 3a Pro
# ============================================================================
#
# PURPOSE:
#   Cross-compile llama.cpp with OpenMP, push to device sandbox, and run
#   structured benchmarks using llama-bench. Records tokens/sec for prefill
#   and decode at multiple thread counts.
#
# PREREQUISITES:
#   - Android NDK r26b (set ANDROID_NDK env var)
#   - ADB connected to Nothing Phone 3a Pro
#   - Device sandbox initialized (run device/sandbox_init.sh first)
#   - Qwen3-1.7B-Q4_K_M.gguf at /data/local/tmp/halfhex/models/
#
# RESULTS:
#   Written to baselines/llamacpp_baseline.txt
#
# WHAT THIS SCRIPT DOES NOT DO:
#   - Does NOT modify CPU governor (benchmarks at stock settings)
#   - Does NOT touch files outside /data/local/tmp/halfhex/
#   - Does NOT require root access
# ============================================================================

set -e

SANDBOX="/data/local/tmp/halfhex"
MODEL_NAME="Qwen3-1.7B-Q4_K_M.gguf"

echo "═══════════════════════════════════════════════════════"
echo "  HalfHex Baseline Profiling (llama.cpp + OpenMP)"
echo "═══════════════════════════════════════════════════════"

# ── Verify environment ─────────────────────────────────────────────────────
if [ -z "$ANDROID_NDK" ]; then
    echo "[ERROR] ANDROID_NDK not set. Export it first:"
    echo "  export ANDROID_NDK=/path/to/android-ndk-r26b"
    exit 1
fi

if ! adb devices 2>/dev/null | grep -q "device$"; then
    echo "[ERROR] No Android device connected via ADB."
    exit 1
fi

# ── Step 1: Build llama.cpp for Android with OpenMP ────────────────────────
echo ""
echo "--- Step 1: Building llama.cpp (OpenMP enabled) ---"

if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
fi

cd llama.cpp

# Configure: OpenMP ON for real multi-threading, NEON auto-detected
cmake -B build-android \
  -G "Unix Makefiles" \
  -DCMAKE_MAKE_PROGRAM="$ANDROID_NDK/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/make" \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=ON \
  -DGGML_VULKAN=OFF \
  -DBUILD_SHARED_LIBS=OFF

cmake --build build-android --config Release --target llama-bench llama-cli -j$(nproc)
cd ..

# ── Step 2: Push binaries + OpenMP library to device sandbox ───────────────
echo ""
echo "--- Step 2: Pushing to device ---"

# Push OpenMP shared library (required by our build)
OMP_LIB="$ANDROID_NDK/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/lib/clang/17/lib/linux/aarch64/libomp.so"
MSYS_NO_PATHCONV=1 adb push "$OMP_LIB" "$SANDBOX/lib/libomp.so"

# Push binaries
MSYS_NO_PATHCONV=1 adb push llama.cpp/build-android/bin/llama-bench "$SANDBOX/bin/llama-bench"
MSYS_NO_PATHCONV=1 adb push llama.cpp/build-android/bin/llama-cli "$SANDBOX/bin/llama-cli"
adb shell "chmod +x $SANDBOX/bin/llama-bench $SANDBOX/bin/llama-cli"

# Verify model exists
if ! adb shell "test -f $SANDBOX/models/$MODEL_NAME" 2>/dev/null; then
    echo "[ERROR] Model not found on device. Push it first:"
    echo "  adb push models/$MODEL_NAME $SANDBOX/models/"
    exit 1
fi

# ── Step 3: Run structured benchmarks ──────────────────────────────────────
echo ""
echo "--- Step 3: Running benchmarks ---"

# Function to run llama-bench with given thread count
run_bench() {
    local threads=$1
    local label=$2
    echo ""
    echo "--- $label ---"
    adb shell "
        cd $SANDBOX
        echo '[PRE] Temp:' \$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf \"%.1f C\", \$1/1000}')
        LD_LIBRARY_PATH=./lib ./bin/llama-bench \
            -m models/$MODEL_NAME \
            -t $threads \
            -ngl 0 \
            -p 512 \
            -n 200 \
            -r 1 \
            2>&1
        echo '[POST] Temp:' \$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf \"%.1f C\", \$1/1000}')
    "
}

# Run at multiple thread counts to find optimal
run_bench 2 "2 threads"
run_bench 4 "4 threads"
run_bench 6 "6 threads"
run_bench 8 "8 threads (all cores)"

# Sustained decode test (512 output tokens)
echo ""
echo "--- Sustained decode: tg512 @ 8 threads ---"
adb shell "
    cd $SANDBOX
    echo '[PRE] Temp:' \$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf \"%.1f C\", \$1/1000}')
    LD_LIBRARY_PATH=./lib ./bin/llama-bench \
        -m models/$MODEL_NAME \
        -t 8 \
        -ngl 0 \
        -p 0 \
        -n 512 \
        -r 1 \
        2>&1
    echo '[POST] Temp:' \$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null | awk '{printf \"%.1f C\", \$1/1000}')
    echo '[POST] Mem:' \$(cat /proc/meminfo | grep MemAvailable | awk '{print int(\$2/1024)}') 'MB'
    echo '[POST] Battery:' \$(dumpsys battery 2>/dev/null | grep level | awk '{print \$2}') '%'
"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Benchmarks complete. Update baselines/llamacpp_baseline.txt"
echo "═══════════════════════════════════════════════════════"
