# VAJRA — Phase 1: QNN INT8 Baseline on Hexagon HTP
## Target: Working INT8 inference of Qwen3-1.7B on HTP V73. Beat llama.cpp.

> Exit criteria for Phase 1: A single number.
> `QNN INT8 decode rate > llama.cpp Q4_0 decode rate` on the same device.
> Do not proceed to Phase 2 until this is true and reproducible across 3 runs.

---

## 0. Phase 1 Mindset

This phase has one job: **prove your pipeline works end-to-end**.
Not fast. Not optimised. Working.

Every decision in Phase 1 is conservative by design:
- INT8 not INT4 (safer, more documented, fewer fallbacks)
- No mixed precision (uniform quantization across all layers)
- No async pipelining (sequential execution only)
- No speculative decoding (single model only)

You are building the floor. Phase 2 builds on it.
A shaky floor collapses everything above it.

---

## 1. Toolchain Installation & Verification

### 1.1 Host Machine Requirements
```
OS:           Ubuntu 22.04 LTS (native bare metal preferred)
              WSL2 acceptable but expect 10-15% slower build times
Python:       3.10.x exactly — QNN SDK has hard version checks
              Use pyenv if your system Python differs
RAM:          16GB minimum for model export (Qwen3-1.7B fp16 = ~3.5GB in memory)
Disk:         50GB free (model weights + ONNX + QNN artifacts)
Android NDK:  r26b exactly — document the exact build hash
ADB:          Latest stable platform-tools (≥34.0.4)
```

### 1.2 SDK Installation Order
```bash
# ── Step 1: QNN SDK ──────────────────────────────────────────────────────────
# Source: https://qpm.qualcomm.com → AI Stack → QNN SDK
# Target version: ≥ 2.20.0 (verify HTP V73 support in release notes)
# After download:
unzip qnn-sdk-*.zip -d $HOME/qnn-sdk
export QNN_SDK=$HOME/qnn-sdk
export PATH=$QNN_SDK/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

# Verify key binaries exist — if any missing, SDK install is broken
ls -la $QNN_SDK/bin/x86_64-linux-clang/qnn-onnx-converter
ls -la $QNN_SDK/bin/x86_64-linux-clang/qnn-net-run
ls -la $QNN_SDK/lib/aarch64-android/libQnnHtp.so
ls -la $QNN_SDK/lib/aarch64-android/libQnnHtpV73Skel.so  # V73-specific — must exist

# ── Step 2: Android NDK r26b ─────────────────────────────────────────────────
# Source: https://developer.android.com/ndk/downloads/older_releases
export ANDROID_NDK=$HOME/android-ndk-r26b

# ── Step 3: Python environment ────────────────────────────────────────────────
python3.10 -m venv $HOME/vajra-env
source $HOME/vajra-env/bin/activate

pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu
pip install transformers==4.51.0
pip install optimum[exporters]==1.18.0
pip install onnx==1.16.0
pip install onnxruntime==1.17.0
pip install onnxsim==0.4.35
pip install numpy pandas matplotlib seaborn
pip install huggingface_hub sentencepiece

# Freeze requirements immediately
pip freeze > requirements_phase1.txt
# Commit this file — reproducibility depends on it
```

### 1.3 Device Preparation
```bash
# Physical setup
# - USB-C cable (data-capable, not charge-only)
# - Enable Developer Options: Settings → About → tap Build Number 7x
# - Enable: USB Debugging, Stay Awake, Disable Absolute Volume

# Verify ADB sees the device
adb devices
# Expected output: <serial>    device
# If "unauthorized": accept RSA fingerprint on phone screen

# Root access — required for performance counters
adb root
# Expected: "restarting adbd as root"
# If "adbd cannot run as root": device needs unlocked bootloader or user debug build

# Push QNN HTP libraries to device
adb shell mkdir -p /data/local/tmp/vajra
adb push $QNN_SDK/lib/aarch64-android/libQnnHtp.so        /data/local/tmp/vajra/
adb push $QNN_SDK/lib/aarch64-android/libQnnHtpV73Skel.so /data/local/tmp/vajra/
adb push $QNN_SDK/lib/aarch64-android/libQnnSystem.so     /data/local/tmp/vajra/
adb push $QNN_SDK/bin/aarch64-android/qnn-net-run         /data/local/tmp/vajra/
adb shell chmod +x /data/local/tmp/vajra/qnn-net-run

# Verify HTP is accessible — this is your first go/no-go gate
adb shell "cd /data/local/tmp/vajra && \
  LD_LIBRARY_PATH=/data/local/tmp/vajra \
  ./qnn-net-run --help" 2>&1 | head -5
# Must print usage, not error

# Set CPU governor to performance mode for all cores
adb shell "for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance > \$cpu
done"

# Verify frequency is locked
adb shell cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq
adb shell cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq
# These should match (or be within 5%)
```

---

## 2. llama.cpp Baseline (Day 1 — Do This Before Any QNN Work)

### 2.1 Build llama.cpp for Android
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git log --oneline -1  # record exact commit hash in your baseline file

# Build 1: CPU only (pure NEON baseline)
cmake -B build-cpu \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DGGML_NEON=ON \
  -DGGML_OPENMP=ON \
  -DGGML_VULKAN=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j$(nproc)

# Build 2: Vulkan enabled (Adreno 710 baseline)
cmake -B build-vulkan \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DGGML_NEON=ON \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j$(nproc)

# Push both binaries
adb push build-cpu/bin/llama-cli    /data/local/tmp/vajra/llama-cli-cpu
adb push build-vulkan/bin/llama-cli /data/local/tmp/vajra/llama-cli-vulkan

# Push model (Qwen3-1.7B Q4_0 GGUF)
adb push /path/to/Qwen3-1.7B-Q4_0.gguf /data/local/tmp/vajra/
```

### 2.2 Baseline Measurement Protocol
```bash
# CRITICAL: Run in this exact order every time you take a measurement
# Deviation in protocol = incomparable numbers

# Step 1: Record device temperature before starting
adb shell cat /sys/class/thermal/thermal_zone0/temp

# Step 2: Thermal soak — run for 60 seconds to reach steady state
# (skip this only if device was idle for >30 minutes at room temp)
adb shell "taskset 0xF0 /data/local/tmp/vajra/llama-cli-cpu \
  -m /data/local/tmp/vajra/Qwen3-1.7B-Q4_0.gguf \
  -p 'Tell me about quantum physics.' -n 50 --threads 4 2>/dev/null"

# Step 3: Actual measurement run
adb shell "taskset 0xF0 /data/local/tmp/vajra/llama-cli-cpu \
  -m /data/local/tmp/vajra/Qwen3-1.7B-Q4_0.gguf \
  -p 'Explain the theory of general relativity including its mathematical foundations, experimental evidence, and modern applications in GPS systems and gravitational wave detection.' \
  -n 200 \
  --threads 4 \
  --ctx-size 512 \
  -ngl 0 \
  --no-mmap \
  2>&1" | grep -E '(eval time|load time|prompt eval|tokens per second)'

# Step 4: Record device temperature after
adb shell cat /sys/class/thermal/thermal_zone0/temp

# Step 5: Repeat Steps 2-4 three times. Record all three.
# Use the MEDIAN value, not the best value.
```

### 2.3 Baseline Record File
```
# baselines/phase1_baseline.txt — commit this to git

timestamp:          YYYY-MM-DD HH:MM
device:             Nothing Phone 3a Pro
soc:                Snapdragon 7s Gen 3
llama_cpp_commit:   <git hash>
model:              Qwen3-1.7B-Q4_0.gguf
gguf_sha256:        <sha256sum of model file>

# CPU baseline (llama.cpp NEON, 4 threads, -ngl 0)
run_1_eval_rate:    XX.XX tokens/sec
run_2_eval_rate:    XX.XX tokens/sec
run_3_eval_rate:    XX.XX tokens/sec
cpu_baseline_median: XX.XX tokens/sec    ← YOUR TARGET TO BEAT

run_1_prompt_rate:  XX.XX tokens/sec
prompt_baseline_median: XX.XX tokens/sec

temp_start:         XX°C
temp_end:           XX°C
throttle_observed:  YES/NO

# Vulkan baseline (llama.cpp, -ngl 99)
vulkan_eval_median: XX.XX tokens/sec     ← record but this isn't your primary target
```

---

## 3. Model Export: HuggingFace → ONNX

### 3.1 Download Weights
```bash
python3 << 'EOF'
from huggingface_hub import snapshot_download
import os, hashlib, time

print(f"[{time.strftime('%H:%M:%S')}] Starting download...")
path = snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir="./models/qwen3-1.7b-hf",
    ignore_patterns=["*.gguf", "*.bin", "flax_model*", "tf_model*"],
)
print(f"[{time.strftime('%H:%M:%S')}] Download complete: {path}")

# Verify files exist
required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
for f in required:
    fpath = os.path.join(path, f)
    exists = os.path.exists(fpath)
    print(f"  {'✓' if exists else '✗'} {f}")
EOF
```

### 3.2 Inspect Model Architecture Before Export
```python
# inspect_model.py — run this before export, read every output line
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import json, time

config = AutoConfig.from_pretrained("./models/qwen3-1.7b-hf")
print("=" * 60)
print("QWEN3-1.7B ARCHITECTURE PROFILE")
print("=" * 60)
print(f"Layers:           {config.num_hidden_layers}")
print(f"Hidden dim:       {config.hidden_size}")
print(f"Attention heads:  {config.num_attention_heads}")
print(f"KV heads (GQA):   {config.num_key_value_heads}")  # ← note this for KV cache calc
print(f"FFN intermediate: {config.intermediate_size}")
print(f"Vocab size:       {config.vocab_size}")
print(f"Max position:     {config.max_position_embeddings}")
print(f"Head dim:         {config.hidden_size // config.num_attention_heads}")

# Calculate memory footprints
params_total = sum([
    config.vocab_size * config.hidden_size,  # embeddings
    config.num_hidden_layers * (
        4 * config.hidden_size * config.hidden_size +   # Q,K,V,O projections (approx)
        3 * config.hidden_size * config.intermediate_size  # gate, up, down
    ),
])
print(f"\nEstimated param count: ~{params_total/1e9:.2f}B")
print(f"fp16 weight size:    ~{params_total*2/1e9:.2f} GB")
print(f"INT8 weight size:    ~{params_total*1/1e9:.2f} GB")
print(f"INT4 weight size:    ~{params_total*0.5/1e9:.2f} GB")

# KV cache size at different context lengths
kv_heads = config.num_key_value_heads
head_dim = config.hidden_size // config.num_attention_heads
layers = config.num_hidden_layers
for ctx_len in [128, 256, 512, 1024]:
    kv_bytes = layers * 2 * kv_heads * head_dim * ctx_len * 2  # fp16
    print(f"KV cache @{ctx_len:4d} tokens (fp16): {kv_bytes/1e6:.1f} MB")
```

### 3.3 ONNX Export
```python
# export_onnx.py
import time, os, json
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer
import onnx
from collections import Counter

MODEL_PATH = "./models/qwen3-1.7b-hf"
ONNX_PATH  = "./models/qwen3-1.7b-onnx"
os.makedirs(ONNX_PATH, exist_ok=True)
os.makedirs("./logs", exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] Starting ONNX export...")
t0 = time.perf_counter()

main_export(
    model_name_or_path=MODEL_PATH,
    output=ONNX_PATH,
    task="text-generation-with-past",   # KV cache support — do NOT use text-generation
    opset=17,
    device="cpu",
    fp16=False,                          # keep fp32 in ONNX — QNN quantizes separately
    optimize="O2",
    no_post_process=False,
)

elapsed = time.perf_counter() - t0
print(f"[{time.strftime('%H:%M:%S')}] Export complete in {elapsed:.1f}s")

# Profile exported graph
model_files = [f for f in os.listdir(ONNX_PATH) if f.endswith('.onnx')]
print(f"\nExported ONNX files: {model_files}")

for fname in model_files:
    fpath = os.path.join(ONNX_PATH, fname)
    fsize = os.path.getsize(fpath) / 1e9
    model = onnx.load(fpath)
    op_counts = Counter(n.op_type for n in model.graph.node)
    print(f"\n{'='*50}")
    print(f"File: {fname} ({fsize:.2f} GB)")
    print(f"Total nodes: {len(model.graph.node)}")
    print(f"Inputs:  {[i.name for i in model.graph.input]}")
    print(f"Outputs: {[o.name for o in model.graph.output]}")
    print(f"Op distribution (top 10):")
    for op, count in op_counts.most_common(10):
        print(f"  {op:35s}: {count}")

    # Save op profile to log
    with open(f"./logs/onnx_op_profile_{fname}.json", "w") as f:
        json.dump(dict(op_counts), f, indent=2)
```

### 3.4 ONNX Simplification & Host Validation
```bash
# Simplify: removes redundant Identity nodes, const-folds, cleans graph
python -m onnxsim \
  ./models/qwen3-1.7b-onnx/model.onnx \
  ./models/qwen3-1.7b-onnx/model_sim.onnx \
  --check-n 3 \
  2>&1 | tee ./logs/onnxsim.log

# If model_with_past exists (optimum generates two files), simplify both:
python -m onnxsim \
  ./models/qwen3-1.7b-onnx/model_with_past.onnx \
  ./models/qwen3-1.7b-onnx/model_with_past_sim.onnx \
  --check-n 3 \
  2>&1 | tee ./logs/onnxsim_with_past.log
```

```python
# validate_onnx_host.py — must pass before touching QNN
import onnxruntime as ort
import numpy as np
import time, sys

ONNX_MODEL = "./models/qwen3-1.7b-onnx/model_sim.onnx"
print(f"Loading: {ONNX_MODEL}")

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX_MODEL, sess_options)

input_names = [i.name for i in sess.get_inputs()]
print(f"Model inputs: {input_names}")

# Build minimal inputs
inputs = {
    "input_ids":      np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
    "attention_mask": np.ones((1, 5), dtype=np.int64),
}
# Add position_ids if model requires them
if "position_ids" in input_names:
    inputs["position_ids"] = np.arange(5, dtype=np.int64).reshape(1, 5)

print("Warmup runs...")
for _ in range(3):
    out = sess.run(None, inputs)

print("Timing runs (20 iterations)...")
times = []
for _ in range(20):
    t0 = time.perf_counter()
    out = sess.run(None, inputs)
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
print(f"\nONNX host inference (5-token input):")
print(f"  mean:  {np.mean(times):.1f} ms")
print(f"  p50:   {np.percentile(times, 50):.1f} ms")
print(f"  p95:   {np.percentile(times, 95):.1f} ms")
print(f"  min:   {min(times):.1f} ms")
print(f"  max:   {max(times):.1f} ms")
print(f"  output shape: {out[0].shape}")
print(f"\n✓ ONNX host validation passed")
```

---

## 4. QNN INT8 Conversion

### 4.1 Generate Calibration Dataset
```python
# generate_calibration.py
# Quality of calibration data directly determines INT8 accuracy
# 50 diverse prompts minimum — do not skimp here

import numpy as np
from transformers import AutoTokenizer
import os

TOKENIZER_PATH = "./models/qwen3-1.7b-hf"
OUTPUT_DIR     = "./calibration/int8"
SEQ_LEN        = 128   # calibration sequence length
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

CALIBRATION_PROMPTS = [
    # General knowledge
    "Explain quantum entanglement in simple terms.",
    "What is the capital of France and why is Paris historically significant?",
    "Describe the process of photosynthesis at the molecular level.",
    "How do vaccines train the human immune system?",
    "What caused the French Revolution?",
    # Reasoning
    "If all cats are animals and some animals are pets, what can we conclude about cats?",
    "A train travels 300km in 3 hours. What is its average speed?",
    "Explain the difference between correlation and causation with examples.",
    # Code
    "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes.",
    "Explain what a binary search tree is and when to use one.",
    "What is the difference between TCP and UDP protocols?",
    "How does garbage collection work in modern programming languages?",
    # Language
    "Translate 'The weather is beautiful today' into French, Spanish, and German.",
    "What is the difference between 'effect' and 'affect'?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    # Math
    "Explain the Pythagorean theorem with a practical example.",
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "How do you calculate compound interest?",
    # Science
    "What is the speed of light and why is it considered a universal constant?",
    "Explain how black holes form.",
    "What is CRISPR and how does it work?",
    "Describe the water cycle.",
    # Practical
    "What are the main ingredients in bread and what role does each play?",
    "How does a car engine work?",
    "Explain how GPS determines your location.",
    "What is machine learning and how does it differ from traditional programming?",
    # Indian context (for domain relevance)
    "Explain the significance of the Indian monsoon to agriculture.",
    "What is UPI and how does it work?",
    "Describe the architecture of a typical South Indian temple.",
    "What are the main programming languages used in Indian IT companies?",
    # Longer prompts (stress test tokenizer)
    "I am building an Android application that uses a local language model for offline inference. The model needs to run efficiently on a Snapdragon processor without an internet connection. Can you explain the key technical considerations I should keep in mind?",
    "Explain in detail the differences between various quantization methods for neural networks, including INT8, INT4, GPTQ, AWQ, and how each affects model accuracy and inference speed.",
    "Describe step by step how to set up a complete CI/CD pipeline for an Android application, including automated testing, code signing, and deployment to the Play Store.",
    # Edge cases
    "1 + 1 =",
    "What?",
    "Hello.",
    "Write code.",
    "Why is the sky blue? Please explain using physics.",
    "List the planets in our solar system in order from the sun.",
    "What is recursion? Explain with a simple Python example.",
    "Compare and contrast supervised and unsupervised learning.",
    "What is the time complexity of quicksort?",
    "How does HTTPS work?",
    "Explain neural networks to a 10-year-old.",
    "What is Docker and why do developers use it?",
    "Describe the steps to make a cup of tea.",
    "What is Ohm's Law?",
    "How do you reverse a linked list?",
    "What are design patterns in software engineering?",
    "Explain the CAP theorem in distributed systems.",
    "What is the difference between SQL and NoSQL databases?",
    "How does a compiler work?",
    "What is Fourier transform used for?",
]

print(f"Generating {len(CALIBRATION_PROMPTS)} calibration samples...")
for i, prompt in enumerate(CALIBRATION_PROMPTS):
    enc = tokenizer(
        prompt,
        return_tensors="np",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )
    np.save(f"{OUTPUT_DIR}/input_ids_{i:03d}.npy",      enc["input_ids"])
    np.save(f"{OUTPUT_DIR}/attention_mask_{i:03d}.npy", enc["attention_mask"])

print(f"✓ Saved {len(CALIBRATION_PROMPTS)} samples to {OUTPUT_DIR}/")
print(f"  Shape: input_ids={enc['input_ids'].shape}, dtype={enc['input_ids'].dtype}")
```

### 4.2 Quantization Override Config
```json
// quantization_config_int8.json
// Conservative INT8 config — keep sensitive layers in fp16
{
  "activation_encodings": {},
  "param_encodings": {
    "model.embed_tokens.weight": [
      {"bitwidth": 16, "dtype": "float"}
    ],
    "lm_head.weight": [
      {"bitwidth": 16, "dtype": "float"}
    ],
    "model.norm.weight": [
      {"bitwidth": 16, "dtype": "float"}
    ]
  }
}
// Note: all other weights default to INT8
// Embedding and LM head in fp16 — these directly affect token quality
// Layer norms in fp16 — tiny params but critical for numerical stability
```

### 4.3 QNN Conversion Command
```bash
# This is the core conversion — log everything, read every warning

mkdir -p ./models/qwen3-1.7b-qnn-int8 ./logs

qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_sim.onnx \
  --output_path   ./models/qwen3-1.7b-qnn-int8/qwen3_int8.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --act_bw    8 \
  --weight_bw 8 \
  --bias_bw   32 \
  --float_fallback \
  --keep_int64_inputs \
  --quantization_overrides ./quantization_config_int8.json \
  --input_list ./calibration/int8/input_list.txt \
  --verbose \
  2>&1 | tee ./logs/qnn_int8_conversion.log

# CRITICAL: After conversion, grep for these warning patterns
echo "=== FALLBACK WARNINGS ==="
grep -c "FALLBACK\|fallback" ./logs/qnn_int8_conversion.log
grep "FALLBACK\|fallback" ./logs/qnn_int8_conversion.log | head -20

echo "=== PRECISION CHANGES ==="
grep "precision\|PRECISION" ./logs/qnn_int8_conversion.log | head -20

echo "=== UNSUPPORTED OPS ==="
grep "unsupported\|UNSUPPORTED\|not supported" ./logs/qnn_int8_conversion.log | head -20

# Any op with FALLBACK to CPU is a performance regression
# Investigate each one — some are acceptable (layer norm), most are not
```

### 4.4 Compile to HTP Binary
```bash
# Generate Android shared library from QNN model code
qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn-int8/qwen3_int8.cpp \
  -b ./models/qwen3-1.7b-qnn-int8/qwen3_int8.bin \
  -o ./models/qwen3-1.7b-qnn-int8/ \
  -t aarch64-android \
  2>&1 | tee ./logs/qnn_int8_compile.log

# Verify output
ls -lh ./models/qwen3-1.7b-qnn-int8/
# Must see: libqwen3_int8.so (the compiled model binary)
# Typical size: 800MB–1.2GB for INT8 Qwen3-1.7B

echo "=== Compilation result ==="
if [ -f "./models/qwen3-1.7b-qnn-int8/libqwen3_int8.so" ]; then
  echo "✓ Compilation succeeded"
  ls -lh ./models/qwen3-1.7b-qnn-int8/libqwen3_int8.so
else
  echo "✗ COMPILATION FAILED — check ./logs/qnn_int8_compile.log"
fi
```

---

## 5. Android Runtime: Phase 1 Implementation

### 5.1 CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.22)
project(vajra_runtime VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable aggressive profiling in debug builds
option(PROFILING_ENABLED "Enable microsecond profiling" ON)
if(PROFILING_ENABLED)
  add_definitions(-DPROFILING_ENABLED)
endif()

# Compiler optimisations for Cortex-A715
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv9-a+sve2 -ffast-math -DNDEBUG")

find_library(ANDROID_LOG_LIB log)

# QNN SDK paths (set via cmake -DQNN_SDK_ROOT=...)
if(NOT DEFINED QNN_SDK_ROOT)
  message(FATAL_ERROR "Set -DQNN_SDK_ROOT=/path/to/qnn-sdk")
endif()

add_library(vajra_runtime SHARED
  src/QnnRuntime.cpp
  src/Profiler.cpp
  src/KVCacheManager.cpp
  src/ThermalMonitor.cpp
  src/TokenizerWrapper.cpp
  jni/InferenceJNI.cpp
)

target_include_directories(vajra_runtime PRIVATE
  include/
  ${QNN_SDK_ROOT}/include/QNN/
)

target_link_libraries(vajra_runtime
  ${ANDROID_LOG_LIB}
  android
  dl
)
```

### 5.2 Profiler.h — Microsecond Resolution, Always On
```cpp
#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <android/log.h>

#define LOG_TAG "VAJRA"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#ifdef PROFILING_ENABLED
  #define PROFILE_SCOPE(name) ScopedTimer _pt_##__LINE__(name)
  #define PROFILE_LOG(fmt, ...) LOGI("[PROFILE] " fmt, ##__VA_ARGS__)
#else
  #define PROFILE_SCOPE(name)
  #define PROFILE_LOG(fmt, ...)
#endif

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Micros    = std::chrono::microseconds;

inline int64_t now_us() {
    return std::chrono::duration_cast<Micros>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

class Profiler {
public:
    static Profiler& get() {
        static Profiler instance;
        return instance;
    }

    void record(const std::string& name, double duration_us) {
        timings_[name].push_back(duration_us);
    }

    void print_summary(const std::string& label = "") {
        LOGI("═══════════════════════════════════════════════════");
        LOGI("PROFILE SUMMARY %s", label.c_str());
        LOGI("%-35s %8s %8s %8s %8s", "Region", "mean_us", "p50_us", "p95_us", "count");
        LOGI("───────────────────────────────────────────────────");

        for (auto& [name, vals] : timings_) {
            if (vals.empty()) continue;
            auto sorted = vals;
            std::sort(sorted.begin(), sorted.end());
            double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
            double p50  = sorted[sorted.size() * 50 / 100];
            double p95  = sorted[sorted.size() * 95 / 100];
            LOGI("  %-33s %8.0f %8.0f %8.0f %8zu",
                 name.c_str(), mean, p50, p95, vals.size());
        }
        LOGI("═══════════════════════════════════════════════════");
    }

    void write_csv(const std::string& path) {
        std::ofstream f(path);
        f << "name,duration_us,index\n";
        for (auto& [name, vals] : timings_) {
            for (size_t i = 0; i < vals.size(); ++i) {
                f << name << "," << vals[i] << "," << i << "\n";
            }
        }
        LOGI("[PROFILE] CSV written to: %s", path.c_str());
    }

    void reset() { timings_.clear(); }

    // Print tokens/sec derived from decode_step_total timings
    void print_throughput() {
        auto it = timings_.find("decode_step_total");
        if (it == timings_.end() || it->second.empty()) {
            LOGW("[PROFILE] No decode_step_total timings recorded");
            return;
        }
        auto& vals = it->second;
        double mean_us  = std::accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        auto sorted = vals;
        std::sort(sorted.begin(), sorted.end());

        LOGI("═══════════════════════════════════════════════════");
        LOGI("THROUGHPUT SUMMARY (%zu tokens)", vals.size());
        LOGI("  Mean:      %.2f tok/s  (%.0f us/tok)", 1e6/mean_us, mean_us);
        LOGI("  P50:       %.2f tok/s", 1e6/sorted[sorted.size()*50/100]);
        LOGI("  P95 (best):%.2f tok/s", 1e6/sorted[sorted.size()*5/100]);
        LOGI("  P5 (worst):%.2f tok/s", 1e6/sorted[sorted.size()*95/100]);
        LOGI("  Min tok/s: %.2f  (thermal floor)", 1e6/sorted.back());
        LOGI("═══════════════════════════════════════════════════");
    }

private:
    std::unordered_map<std::string, std::vector<double>> timings_;
};

struct ScopedTimer {
    ScopedTimer(const char* name) : name_(name), start_(now_us()) {}
    ~ScopedTimer() {
        double elapsed = now_us() - start_;
        Profiler::get().record(name_, elapsed);
    }
    const char* name_;
    int64_t start_;
};
```

### 5.3 QnnRuntime.cpp — Phase 1 (Sequential, No Async)
```cpp
#include "QnnRuntime.h"
#include "Profiler.h"
#include <dlfcn.h>
#include <android/log.h>

// QNN API function pointer types
typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_t)(
    const QnnInterface_t***, uint32_t*);

class QnnRuntime {
public:

    bool initialize(const std::string& model_so_path,
                    const std::string& backend_so_path) {
        PROFILE_SCOPE("initialize_total");

        // ── 1. Load HTP backend shared library ───────────────────────────
        {
            PROFILE_SCOPE("load_htp_backend_so");
            backend_handle_ = dlopen(backend_so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!backend_handle_) {
                LOGE("dlopen HTP backend failed: %s", dlerror());
                LOGE("Ensure libQnnHtp.so is in LD_LIBRARY_PATH");
                return false;
            }
            PROFILE_LOG("HTP backend loaded: %s", backend_so_path.c_str());
        }

        // ── 2. Get QNN interface ──────────────────────────────────────────
        {
            PROFILE_SCOPE("get_qnn_interface");
            auto getProviders = (QnnInterface_getProviders_t)
                dlsym(backend_handle_, "QnnInterface_getProviders");
            if (!getProviders) {
                LOGE("QnnInterface_getProviders not found in backend");
                return false;
            }
            const QnnInterface_t** providers = nullptr;
            uint32_t num_providers = 0;
            getProviders(&providers, &num_providers);
            if (num_providers == 0) {
                LOGE("No QNN providers found");
                return false;
            }
            qnn_ = providers[0]->QNN_INTERFACE_VER_NAME;
            PROFILE_LOG("QNN interface version: %u.%u",
                QNNN_INTERFACE_VER_MAJOR, QNNN_INTERFACE_VER_MINOR);
        }

        // ── 3. Create backend ─────────────────────────────────────────────
        {
            PROFILE_SCOPE("create_backend");
            // HTP performance config: sustained high performance mode
            QnnHtpDevice_PerfInfrastructure_t perf_infra = {};
            perf_infra.createPowerConfigId = 1;

            Qnn_BackendConfig_t backend_cfg = QNN_BACKEND_CONFIG_INIT;
            auto err = qnn_.backendCreate(
                nullptr, &backend_cfg, 1, &backend_handle_qnn_);
            if (err != QNN_SUCCESS) {
                LOGE("backendCreate failed: %d", err);
                return false;
            }
            PROFILE_LOG("QNN backend created");
        }

        // ── 4. Load model .so ─────────────────────────────────────────────
        {
            PROFILE_SCOPE("load_model_so");
            model_handle_ = dlopen(model_so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!model_handle_) {
                LOGE("dlopen model failed: %s", dlerror());
                return false;
            }
            PROFILE_LOG("Model loaded: %s", model_so_path.c_str());
        }

        // ── 5. Setup context and graph ────────────────────────────────────
        {
            PROFILE_SCOPE("setup_context_and_graph");
            // Context config with HTP optimisation level 3 (maximum)
            QnnHtpGraph_CustomConfig_t graph_cfg = {};
            graph_cfg.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
            graph_cfg.optimizationOption.type =
                QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
            graph_cfg.optimizationOption.floatValue = 3.0f;

            auto err = qnn_.contextCreate(
                backend_handle_qnn_, nullptr, 0, &context_handle_);
            if (err != QNN_SUCCESS) {
                LOGE("contextCreate failed: %d", err);
                return false;
            }
        }

        log_memory_post_init();
        is_initialized_ = true;
        LOGI("[PROFILE] ✓ QNN Runtime initialized successfully");
        return true;
    }

    // Single decode step — this is the inner loop
    // Profile every sub-operation
    bool decode_step(
        int32_t token_id,
        int     position,
        float*  logits_out,   // pre-allocated, vocab_size floats
        int     vocab_size)
    {
        PROFILE_SCOPE("decode_step_total");

        // Input: single token
        {
            PROFILE_SCOPE("decode_input_set");
            input_ids_buf_[0] = token_id;
            pos_ids_buf_[0]   = position;
            // attention_mask is always 1 for current token
            // KV cache pointers are updated externally before each call
            set_input_tensor("input_ids",      input_ids_buf_.data(), {1, 1});
            set_input_tensor("position_ids",   pos_ids_buf_.data(),   {1, 1});
            set_input_tensor("attention_mask", attn_mask_buf_.data(), {1, max_seq_len_});
        }

        // Execute on HTP — this is the number that matters
        {
            PROFILE_SCOPE("htp_graph_execute");
            int64_t t0 = now_us();

            auto err = qnn_.graphExecute(
                graph_handle_,
                input_tensors_.data(),  input_tensors_.size(),
                output_tensors_.data(), output_tensors_.size(),
                nullptr, nullptr
            );

            int64_t elapsed_us = now_us() - t0;
            PROFILE_LOG("htp_execute: %lld us → %.2f tok/s",
                        elapsed_us, 1e6 / elapsed_us);

            if (err != QNN_SUCCESS) {
                LOGE("[decode] graphExecute error: %d at position %d", err, position);
                return false;
            }
        }

        // Extract logits
        {
            PROFILE_SCOPE("decode_logits_extract");
            memcpy(logits_out, logits_tensor_ptr_, vocab_size * sizeof(float));
        }

        return true;
    }

    // Prefill: process full prompt
    // Separate profile target: time-to-first-token (TTFT)
    bool prefill(
        const std::vector<int32_t>& token_ids,
        float* logits_out,
        int    vocab_size)
    {
        PROFILE_SCOPE("prefill_total");
        int64_t t0 = now_us();
        int     n  = token_ids.size();

        PROFILE_LOG("Prefill start: %d tokens", n);

        {
            PROFILE_SCOPE("prefill_input_set");
            set_input_tensor("input_ids",      token_ids.data(),     {1, n});
            set_attention_mask_prefill(n);
        }

        {
            PROFILE_SCOPE("prefill_htp_execute");
            auto err = qnn_.graphExecute(
                graph_handle_,
                input_tensors_.data(),  input_tensors_.size(),
                output_tensors_.data(), output_tensors_.size(),
                nullptr, nullptr
            );
            if (err != QNN_SUCCESS) {
                LOGE("[prefill] graphExecute error: %d", err);
                return false;
            }
        }

        double ttft_ms = (now_us() - t0) / 1000.0;
        double prefill_tps = n / (ttft_ms / 1000.0);
        LOGI("[PROFILE] TTFT: %.1f ms | Prefill: %.1f tok/s (n=%d)",
             ttft_ms, prefill_tps, n);

        memcpy(logits_out, logits_tensor_ptr_, vocab_size * sizeof(float));
        return true;
    }

private:
    void log_memory_post_init() {
        FILE* f = fopen("/proc/self/status", "r");
        char line[256];
        while (f && fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0 ||
                strncmp(line, "VmPeak:", 7) == 0) {
                line[strcspn(line, "\n")] = 0;
                LOGI("[PROFILE][MEM] %s", line);
            }
        }
        if (f) fclose(f);
    }

    // ... handles, tensor buffers, etc.
    bool is_initialized_ = false;
    int  max_seq_len_    = 512;
    void* backend_handle_    = nullptr;
    void* model_handle_      = nullptr;
    Qnn_BackendHandle_t  backend_handle_qnn_ = nullptr;
    Qnn_ContextHandle_t  context_handle_     = nullptr;
    Qnn_GraphHandle_t    graph_handle_        = nullptr;
    QnnInterface_t       qnn_;

    float* logits_tensor_ptr_ = nullptr;
    std::vector<int32_t> input_ids_buf_  = std::vector<int32_t>(1);
    std::vector<int32_t> pos_ids_buf_    = std::vector<int32_t>(1);
    std::vector<int32_t> attn_mask_buf_;
    std::vector<Qnn_Tensor_t> input_tensors_;
    std::vector<Qnn_Tensor_t> output_tensors_;
};
```

---

## 6. Phase 1 Benchmark Harness

```cpp
// benchmark_phase1.cpp
// Simple, sequential, no tricks — just prove HTP runs faster than CPU

#include "QnnRuntime.h"
#include "Profiler.h"
#include "ThermalMonitor.h"
#include <string>

const char* MODEL_PATH   = "/data/local/tmp/vajra/libqwen3_int8.so";
const char* BACKEND_PATH = "/data/local/tmp/vajra/libQnnHtp.so";
const int   VOCAB_SIZE   = 151669;  // Qwen3 vocab size
const int   NUM_WARMUP   = 5;
const int   NUM_TIMED    = 100;

int main() {
    ThermalMonitor thermal;
    QnnRuntime     runtime;
    Profiler&      profiler = Profiler::get();

    LOGI("═══════════════════════════════════════════════");
    LOGI("VAJRA Phase 1 Benchmark — QNN INT8 on HTP V73");
    LOGI("═══════════════════════════════════════════════");

    // ── Load model ────────────────────────────────────────────────────────
    thermal.log_snapshot("pre_load");
    if (!runtime.initialize(MODEL_PATH, BACKEND_PATH)) {
        LOGE("Initialization failed — see above for details");
        return 1;
    }
    thermal.log_snapshot("post_load");

    std::vector<float> logits(VOCAB_SIZE);

    // ── Warmup (not counted) ──────────────────────────────────────────────
    LOGI("Warming up (%d iterations)...", NUM_WARMUP);
    profiler.reset();
    for (int i = 0; i < NUM_WARMUP; i++) {
        runtime.decode_step(1, i, logits.data(), VOCAB_SIZE);
    }
    profiler.reset();  // discard warmup timings

    // ── Thermal soak (60s steady state) ──────────────────────────────────
    LOGI("Thermal soak (60s)...");
    int64_t soak_end = now_us() + 60 * 1000000LL;
    int soak_tokens  = 0;
    while (now_us() < soak_end) {
        runtime.decode_step(42, soak_tokens % 512, logits.data(), VOCAB_SIZE);
        soak_tokens++;
    }
    thermal.log_snapshot("post_soak");
    profiler.reset();  // discard soak timings — fresh start

    // ── Timed benchmark ───────────────────────────────────────────────────
    LOGI("Starting timed benchmark (%d tokens)...", NUM_TIMED);
    bool throttle_observed = false;

    for (int i = 0; i < NUM_TIMED; i++) {
        if (thermal.is_throttling()) {
            if (!throttle_observed) {
                LOGW("[BENCHMARK] First throttle event at token %d", i);
                throttle_observed = true;
            }
        }
        runtime.decode_step(i % VOCAB_SIZE, i % 512, logits.data(), VOCAB_SIZE);

        // Log temperature every 25 tokens
        if (i % 25 == 0) {
            thermal.log_snapshot(("token_" + std::to_string(i)).c_str());
        }
    }

    // ── Sustained 120s benchmark ──────────────────────────────────────────
    LOGI("Starting sustained 120s benchmark...");
    profiler.reset();
    int64_t sustained_end = now_us() + 120 * 1000000LL;
    int sustained_tokens  = 0;

    while (now_us() < sustained_end) {
        runtime.decode_step(
            sustained_tokens % VOCAB_SIZE,
            sustained_tokens % 512,
            logits.data(), VOCAB_SIZE
        );
        sustained_tokens++;
    }

    double sustained_tps = sustained_tokens / 120.0;
    LOGI("SUSTAINED 120s RESULT: %.2f tok/s (%d tokens)", sustained_tps, sustained_tokens);

    // ── Results ───────────────────────────────────────────────────────────
    profiler.print_summary("Phase 1 Final");
    profiler.print_throughput();
    profiler.write_csv("/sdcard/vajra_phase1_profile.csv");

    thermal.log_snapshot("benchmark_complete");

    LOGI("══════════════════════════════════════════");
    LOGI("PHASE 1 COMPLETE");
    LOGI("Pull profile CSV: adb pull /sdcard/vajra_phase1_profile.csv");
    LOGI("Compare against baseline: baselines/phase1_baseline.txt");
    LOGI("Go/no-go: QNN INT8 decode rate > llama.cpp CPU baseline?");
    LOGI("══════════════════════════════════════════");

    return 0;
}
```

---

## 7. Phase 1 Completion Checklist

Before moving to Phase 2, every item must be true:

```
[ ] llama.cpp CPU baseline recorded (3 runs, median noted)
[ ] llama.cpp Vulkan baseline recorded (3 runs, median noted)
[ ] ONNX export succeeds with no errors
[ ] ONNX host validation passes (onnxruntime on x86 produces correct output shapes)
[ ] QNN INT8 conversion log has < 5 FALLBACK warnings
    (if ≥ 5: investigate each one before proceeding)
[ ] libqwen3_int8.so compiled and pushed to device
[ ] libQnnHtpV73Skel.so confirmed present on device
    (if missing: QNN will silently use CPU — your numbers will be wrong)
[ ] Phase 1 benchmark runs without crash for 120 sustained seconds
[ ] QNN INT8 decode rate > llama.cpp CPU baseline (median, post-thermal-soak)
[ ] Profile CSV pulled from device and analyzed with analyze_profile.py
[ ] results/comparison.md row 1 and row 3 filled in
[ ] No unexplained memory growth during 120s sustained run
    (check via: adb shell cat /proc/<pid>/status | grep VmRSS every 30s)
```

**If QNN INT8 is SLOWER than llama.cpp CPU:**
This should not happen. If it does, in priority order:
1. Verify `libQnnHtpV73Skel.so` is on device — without it, silent CPU fallback
2. Verify HTP power mode is set (see Section 1.3)
3. Check conversion log for excessive FALLBACK warnings
4. Verify you are measuring post-warmup (first run includes JIT overhead)
5. Verify device is not thermally throttling during measurement
