# HalfHex Project Instructions

## Project Overview
HalfHex is a QNN-native LLM inference runtime targeting Qwen3-1.7B on
Snapdragon 7s Gen 3 (Nothing Phone 3a Pro) via Hexagon HTP.

## Architecture
- C++ runtime lives in `qnn_llm_runtime/` under the `halfhex::` namespace
- Python pipeline scripts are at project root
- Device scripts are in `device/`
- Documentation is in `docs/`

## Device Safety Rules
- Everything on the phone goes under `/data/local/tmp/halfhex/`
- NEVER touch files outside this sandbox
- Always use oom_guard.sh wrapper when running inference
- Thermal limit: 55°C (abort above this)
- Memory reserve: keep 1GB system RAM available at all times
- OOM score: 800 (kill our process before user apps)

## Key Model Constants (Qwen3-1.7B)
- Layers: 28
- KV heads: 4 (GQA)
- Q heads: 16
- Head dim: 128
- Hidden size: 2048
- Vocab size: 151,936
- Max context: 512 (our deployment target)

## Build
```bash
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 \
  -DQNN_SDK_ROOT=$QNN_SDK -DPROFILING_ENABLED=ON \
  -S qnn_llm_runtime
cmake --build build-android -j$(nproc)
```

## Commit Guidelines
- Extremely detailed commit messages
- No AI attribution in commits (configured in .claude/settings.json)
- Commit regularly for backup
