# Build Guide — HalfHex QNN LLM Runtime

This document provides step-by-step instructions for building every component
of the HalfHex project, from host-side Python tools to the cross-compiled
Android binary. Follow each section in order.

---

## Table of Contents

1. [Host Machine Requirements](#1-host-machine-requirements)
2. [SDK Installation](#2-sdk-installation)
3. [Python Environment Setup](#3-python-environment-setup)
4. [Model Download](#4-model-download)
5. [ONNX Export](#5-onnx-export)
6. [ONNX Simplification and Validation](#6-onnx-simplification-and-validation)
7. [Calibration Data Generation](#7-calibration-data-generation)
8. [QNN INT8 Conversion](#8-qnn-int8-conversion)
9. [QNN INT4 Conversion](#9-qnn-int4-conversion)
10. [Cross-Compiling the Runtime](#10-cross-compiling-the-runtime)
11. [Building SentencePiece for Android](#11-building-sentencepiece-for-android)
12. [Building llama.cpp Baseline](#12-building-llamacpp-baseline)
13. [Deploying to Device](#13-deploying-to-device)
14. [Verification Checklist](#14-verification-checklist)

---

## 1. Host Machine Requirements

| Component      | Required Version  | Notes                                      |
|----------------|-------------------|--------------------------------------------|
| OS             | Ubuntu 22.04 LTS  | Native preferred. WSL2 acceptable.         |
| Python         | 3.10 (exact)      | QNN SDK has strict version pins.           |
| Android NDK    | r26b (exact)      | Later versions break QNN integration.      |
| CMake          | >= 3.18           | Ubuntu 22.04 ships 3.22.                   |
| ADB            | Latest            | From Android SDK platform-tools.           |
| Git            | >= 2.30           | For cloning dependencies.                  |
| Disk space     | >= 20 GB free     | Model weights + ONNX + QNN artifacts.      |

**Do NOT use newer NDK versions.** The QNN SDK's model compilation tools
depend on specific libc++ symbols from NDK r26b. NDK r27+ changes the
ABI in ways that cause silent runtime failures.

---

## 2. SDK Installation

### 2.1 Qualcomm AI Engine Direct SDK (QNN SDK)

1. Create an account at https://qpm.qualcomm.com
2. Download **QNN SDK >= v2.20**
3. Extract to a permanent location:

```bash
export QNN_SDK=/opt/qnn-sdk-v2.20
tar -xzf qnn-sdk-v2.20.tar.gz -C /opt/
```

4. Verify the SDK contents:

```bash
ls $QNN_SDK/bin/x86_64-linux-clang/qnn-onnx-converter   # Must exist
ls $QNN_SDK/lib/aarch64-android/libQnnHtp.so             # Must exist
ls $QNN_SDK/lib/aarch64-android/libQnnHtpV73Skel.so      # Must exist
```

### 2.2 Android NDK r26b

```bash
export ANDROID_NDK=/opt/android-ndk-r26b
# Download from: https://developer.android.com/ndk/downloads
# Verify:
$ANDROID_NDK/ndk-build --version
```

### 2.3 Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export ANDROID_NDK=/opt/android-ndk-r26b
export QNN_SDK=/opt/qnn-sdk-v2.20
export PATH=$QNN_SDK/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
```

---

## 3. Python Environment Setup

```bash
# Create isolated environment
python3.10 -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.51.0 optimum onnx onnxruntime onnxsim
pip install numpy pandas matplotlib seaborn huggingface_hub
```

**Why exact versions?**
- `torch==2.2.0`: ONNX export stability. 2.3+ changes the export graph structure.
- `transformers==4.51.0`: Qwen3 model support. Earlier versions lack Qwen3 architecture.
- CPU-only PyTorch: We only need it for weight export, not training.

---

## 4. Model Download

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-1.7B',
    local_dir='./models/qwen3-1.7b-hf',
    ignore_patterns=['*.gguf', '*.bin']
)
"
```

This downloads ~3.4 GB of SafeTensors weights.

---

## 5. ONNX Export

```bash
python export_to_onnx.py
```

Expected output:
```
[PROFILE] Model load: ~15s
[PROFILE] Model params: 1700.0M
[PROFILE] Model size fp16: 3.40GB
[PROFILE] ONNX export: ~120s
[PROFILE] ONNX graph nodes: ~5000
```

Output: `./models/qwen3-1.7b-onnx/model.onnx`

---

## 6. ONNX Simplification and Validation

```bash
# Simplify (removes redundant ops before QNN conversion)
python -m onnxsim \
  ./models/qwen3-1.7b-onnx/model.onnx \
  ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --overwrite-input-shape "input_ids:1,1" "attention_mask:1,512" \
  --check-n 3

# Validate inference on host
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('./models/qwen3-1.7b-onnx/model_simplified.onnx')
inputs = {
    'input_ids': np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
    'attention_mask': np.ones((1, 5), dtype=np.int64),
}
outputs = sess.run(None, inputs)
print(f'Output shape: {outputs[0].shape}')  # Should be [1, 5, 151936]
print('Validation PASSED')
"
```

---

## 7. Calibration Data Generation

```bash
python generate_calibration_data.py
```

This creates 50 calibration samples in `./calibration_data/`.
These represent diverse prompts for INT8/INT4 quantization.

---

## 8. QNN INT8 Conversion

```bash
mkdir -p ./models/qwen3-1.7b-qnn ./logs

# Convert ONNX to QNN
qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --output_path ./models/qwen3-1.7b-qnn/model.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --quantization_overrides ./quantization_config.json \
  --act_bw 8 --weight_bw 8 --bias_bw 32 \
  --float_fallback --keep_int64_inputs --verbose \
  2>&1 | tee ./logs/qnn_conversion.log

# Compile to HTP binary
qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn/model.cpp \
  -b ./models/qwen3-1.7b-qnn/model.bin \
  -o ./models/qwen3-1.7b-qnn/ \
  -t android_aarch64 \
  2>&1 | tee ./logs/qnn_compilation.log
```

**Check the conversion log for `[FALLBACK]` warnings.** Any op that falls
back to CPU will destroy your throughput.

---

## 9. QNN INT4 Conversion

Only attempt this after INT8 is validated and producing correct output.

```bash
mkdir -p ./models/qwen3-1.7b-qnn-int4

qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --output_path ./models/qwen3-1.7b-qnn-int4/model.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --act_bw 8 --weight_bw 4 --bias_bw 32 \
  --float_fallback \
  --quantization_overrides ./quantization_config_int4.json \
  --verbose \
  2>&1 | tee ./logs/qnn_int4_conversion.log

qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn-int4/model.cpp \
  -b ./models/qwen3-1.7b-qnn-int4/model.bin \
  -o ./models/qwen3-1.7b-qnn-int4/ \
  -t android_aarch64 \
  2>&1 | tee -a ./logs/qnn_int4_conversion.log
```

INT4 weights with INT8 activations (W4A8) uses the HTP V73's dedicated
4-bit dot-product engines. This is your biggest performance gain.

---

## 10. Cross-Compiling the Runtime

```bash
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DCMAKE_BUILD_TYPE=Release \
  -DQNN_SDK_ROOT=$QNN_SDK \
  -DPROFILING_ENABLED=ON \
  -S qnn_llm_runtime

cmake --build build-android --config Release -j$(nproc)
```

Output: `build-android/qnn_benchmark`

To disable profiling for production (saves ~48MB):
```bash
cmake -DPROFILING_ENABLED=OFF ...
```

---

## 11. Building SentencePiece for Android

```bash
git clone https://github.com/google/sentencepiece
cd sentencepiece
mkdir build-android && cd build-android

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DSPM_ENABLE_SHARED=OFF \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

Output: `build-android/src/libsentencepiece.a`

Link this in CMakeLists.txt:
```cmake
target_link_libraries(qnn_benchmark PRIVATE
    ${SENTENCEPIECE_LIB}/libsentencepiece.a
    dl log
)
```

---

## 12. Building llama.cpp Baseline

This is your Day 1 measurement. Do this first.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DGGML_NEON=ON \
  -DGGML_OPENMP=ON

cmake --build build-android --config Release -j$(nproc)
```

---

## 13. Deploying to Device

```bash
# Initialize sandbox
cat device/sandbox_init.sh | adb shell "cat > /data/local/tmp/sandbox_init.sh"
adb shell "chmod +x /data/local/tmp/sandbox_init.sh"
adb shell "sh /data/local/tmp/sandbox_init.sh"

# Push runtime binary
cat build-android/qnn_benchmark | \
  adb shell "cat > /data/local/tmp/halfhex/bin/qnn_benchmark"
adb shell "chmod +x /data/local/tmp/halfhex/bin/qnn_benchmark"

# Push QNN libraries
cat $QNN_SDK/lib/aarch64-android/libQnnHtp.so | \
  adb shell "cat > /data/local/tmp/halfhex/lib/libQnnHtp.so"
cat $QNN_SDK/lib/aarch64-android/libQnnHtpV73Skel.so | \
  adb shell "cat > /data/local/tmp/halfhex/lib/libQnnHtpV73Skel.so"

# Push model
cat models/qwen3-1.7b-qnn-int4/libqwen3_model.so | \
  adb shell "cat > /data/local/tmp/halfhex/models/libqwen3_model.so"

# Run benchmark
adb shell "cd /data/local/tmp/halfhex && \
  LD_LIBRARY_PATH=./lib \
  taskset 0xF0 \
  ./bin/oom_guard.sh 3072 55 \
  ./bin/qnn_benchmark \
    --model ./models/libqwen3_model.so \
    --tokenizer ./tokenizer/tokenizer.model \
    --output-tokens 100"
```

---

## 14. Verification Checklist

Before reporting any performance numbers, verify:

- [ ] libQnnHtpV73Skel.so is present on device (without it, QNN silently uses CPU)
- [ ] Benchmark ran AFTER thermal soak (check "post_soak" in logcat)
- [ ] No throttle events during the timed benchmark
- [ ] oom_score_adj is set (check logcat for "[MEMGUARD] oom_score_adj set to 800")
- [ ] KV cache is mlocked (check logcat for "mlock succeeded")
- [ ] Running on big cores (taskset 0xF0)
- [ ] Device temperature below 40C at benchmark start
- [ ] No `[FALLBACK]` warnings in QNN conversion log
- [ ] System MemAvailable > 4GB during benchmark
