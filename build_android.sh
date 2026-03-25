#!/bin/bash
# Build the QNN runtime for Android (aarch64)
set -e

echo "═══════════════════════════════════════"
echo "Building QNN LLM Runtime for Android"
echo "═══════════════════════════════════════"

if [ -z "$ANDROID_NDK" ]; then
    echo "ERROR: ANDROID_NDK not set. Export it first."
    exit 1
fi

if [ -z "$QNN_SDK" ]; then
    echo "ERROR: QNN_SDK not set. Export it first."
    exit 1
fi

BUILD_DIR=build-android

cmake -B $BUILD_DIR \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DCMAKE_BUILD_TYPE=Release \
  -DQNN_SDK_ROOT=$QNN_SDK \
  -DPROFILING_ENABLED=ON \
  -S qnn_llm_runtime

cmake --build $BUILD_DIR --config Release -j$(nproc)

echo ""
echo "Build complete: $BUILD_DIR/qnn_benchmark"
echo "═══════════════════════════════════════"
