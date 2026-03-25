# QNN-Native LLM Inference Runtime — Technical Build Prompt
## Target: Qwen3-1.7B on Snapdragon 7s Gen 3 (Nothing Phone 3a Pro) via Hexagon HTP

> Use this document as your master specification. Every phase must be completed
> and profiled before moving to the next. No optimization without measurement.

---

## 0. Ground Rules (Read Before Writing a Single Line)

- **Measure before you optimize. Always.** Intuition about bottlenecks is almost always wrong at this layer.
- **Every function that touches model execution must emit timing.** No exceptions.
- **Establish a baseline llama.cpp number on the same device on Day 1.** That number is your enemy.
- **Target metric:** tokens/sec (decode phase, single-token generation, sustained over 120 seconds).
- **Secondary metric:** time-to-first-token (prefill latency for 512-token prompt).
- **Thermal constraint:** all benchmarks must be run after the device has been at load for 60 seconds. Cold numbers are lies.

---

## 1. Environment & Toolchain Setup

### 1.1 Host Machine
```
OS:           Ubuntu 22.04 LTS (native preferred, WSL2 acceptable)
Python:       3.10 (exact — QNN SDK has strict version pins)
CUDA:         Not required (CPU-only conversion pipeline)
Android NDK:  r26b (exact — later breaks QNN integration)
ADB:          Latest platform-tools
```

### 1.2 SDK Stack (install in this order, no shortcuts)
```bash
# Step 1: Qualcomm AI Engine Direct SDK (QNN SDK)
# Download from: https://qpm.qualcomm.com → QNN SDK ≥ v2.20
# This gives you: qnn-onnx-converter, qnn-net-run, libQnnHtp.so

# Step 2: Qualcomm Neural Processing SDK (optional but useful for SNPE tooling)
# Download: Qualcomm Developer Network → SNPE SDK

# Step 3: Python dependencies
pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.51.0 optimum onnx onnxruntime onnxsim
pip install numpy pandas matplotlib seaborn  # for profiling visualisation

# Step 4: Android deps
export ANDROID_NDK=/path/to/ndk/r26b
export QNN_SDK=/path/to/qnn-sdk
export PATH=$QNN_SDK/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
```

### 1.3 Device Setup
```bash
# Enable developer options on Nothing Phone 3a Pro
# USB debugging ON, Stay awake ON, Disable HW overlays OFF

adb root                          # must succeed — you need root for perf counters
adb shell setenforce 0            # temporarily disable SELinux for profiling
adb shell "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
adb shell "echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"

# Verify HTP device is visible to QNN
adb push $QNN_SDK/lib/aarch64-android/libQnnHtp.so /data/local/tmp/
adb push $QNN_SDK/lib/aarch64-android/libQnnHtpV73Skel.so /data/local/tmp/
adb push $QNN_SDK/bin/aarch64-android/qnn-net-run /data/local/tmp/
adb shell "chmod +x /data/local/tmp/qnn-net-run"
```

---

## 2. Baseline: llama.cpp Profiling (Day 1, Non-Negotiable)

Before writing any QNN code, establish your baseline. This number is sacred.

### 2.1 Build and Run llama.cpp
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# Build for Android with NEON only (no Vulkan yet — isolate CPU baseline)
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DGGML_NEON=ON \
  -DGGML_OPENMP=ON

cmake --build build-android --config Release -j8

# Push to device
adb push build-android/bin/llama-cli /data/local/tmp/
adb push /path/to/Qwen3-1.7B-Q4_0.gguf /data/local/tmp/
```

### 2.2 Baseline Profiling Script
```bash
# Run this EXACTLY — same prompt, same params, every time you compare
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
```

### 2.3 Record These Numbers
```
File: baselines/llamacpp_baseline.txt
Format:
  date: YYYY-MM-DD HH:MM
  device_temp_at_start: XX°C
  device_temp_at_end: XX°C
  prompt_eval_time: XX ms (YY tokens)
  prompt_eval_rate: XX tokens/sec
  eval_time: XX ms (YY tokens)           ← THIS IS YOUR TARGET TO BEAT
  eval_rate: XX tokens/sec               ← THIS IS YOUR TARGET TO BEAT
  sustained_60s_rate: XX tokens/sec      ← measure separately with longer output
```

Also run with Vulkan (`-ngl 99`) and record separately. This gives you two baselines.

---

## 3. Model Export Pipeline: HuggingFace → ONNX

### 3.1 Download Qwen3-1.7B Weights
```bash
# Use HF fp16 weights — NOT the GGUF. QNN needs the raw weights.
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-1.7B',
    local_dir='./models/qwen3-1.7b-hf',
    ignore_patterns=['*.gguf', '*.bin']   # get safetensors only
)
"
```

### 3.2 ONNX Export with Profiling Hooks
```python
# export_to_onnx.py
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.exporters.onnx import main_export
import os

MODEL_PATH = "./models/qwen3-1.7b-hf"
ONNX_PATH  = "./models/qwen3-1.7b-onnx"

print("[PROFILE] Starting model load...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
t1 = time.perf_counter()
print(f"[PROFILE] Model load: {t1-t0:.2f}s")
print(f"[PROFILE] Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"[PROFILE] Model size fp16: {sum(p.numel()*2 for p in model.parameters())/1e9:.2f}GB")

# Export — sequence length 512 is your target context
print("[PROFILE] Starting ONNX export...")
t0 = time.perf_counter()
main_export(
    model_name_or_path=MODEL_PATH,
    output=ONNX_PATH,
    task="text-generation-with-past",  # CRITICAL: with-past = KV cache support
    opset=17,
    device="cpu",
    fp16=False,     # keep fp32 for ONNX — QNN handles quantization separately
    optimize="O2",
)
t1 = time.perf_counter()
print(f"[PROFILE] ONNX export: {t1-t0:.2f}s")

# Profile the exported graph
import onnx
model_onnx = onnx.load(f"{ONNX_PATH}/model.onnx")
print(f"[PROFILE] ONNX graph nodes: {len(model_onnx.graph.node)}")
print(f"[PROFILE] ONNX inputs: {[i.name for i in model_onnx.graph.input]}")
print(f"[PROFILE] ONNX outputs: {[o.name for o in model_onnx.graph.output]}")

# Count op types — know what you're converting
from collections import Counter
op_counts = Counter(node.op_type for node in model_onnx.graph.node)
print("[PROFILE] Op type distribution:")
for op, count in op_counts.most_common(15):
    print(f"  {op:30s}: {count}")
```

### 3.3 ONNX Graph Simplification & Validation
```bash
# Simplify the graph — removes redundant ops before QNN conversion
python -m onnxsim \
  ./models/qwen3-1.7b-onnx/model.onnx \
  ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --overwrite-input-shape "input_ids:1,1" "attention_mask:1,512" \
  --check-n 3

# Validate with ONNX Runtime on host (before converting to QNN)
python -c "
import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession('./models/qwen3-1.7b-onnx/model_simplified.onnx')
inputs = {
    'input_ids': np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
    'attention_mask': np.ones((1, 5), dtype=np.int64),
}
# Warmup
for _ in range(3):
    outputs = sess.run(None, inputs)

# Profile
times = []
for _ in range(20):
    t0 = time.perf_counter()
    outputs = sess.run(None, inputs)
    times.append(time.perf_counter() - t0)

import numpy as np
print(f'[PROFILE] ONNX host inference:')
print(f'  mean: {np.mean(times)*1000:.1f}ms')
print(f'  p50:  {np.percentile(times, 50)*1000:.1f}ms')
print(f'  p95:  {np.percentile(times, 95)*1000:.1f}ms')
print(f'  p99:  {np.percentile(times, 99)*1000:.1f}ms')
"
```

---

## 4. QNN Conversion: ONNX → HTP-Optimized DLC

### 4.1 Generate INT8 Calibration Dataset
```python
# generate_calibration_data.py
# QNN quantization needs representative inputs — do NOT skip this
import numpy as np
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./models/qwen3-1.7b-hf")

# Diverse calibration prompts — cover your actual use cases
CALIBRATION_PROMPTS = [
    "Explain quantum entanglement simply.",
    "Write a Python function to sort a list.",
    "What is the capital of France and why is it significant?",
    "Summarize the French Revolution in 3 sentences.",
    "How does photosynthesis work at the molecular level?",
    "Translate 'Hello world' to 5 different languages.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of making bread from scratch.",
    "How do vaccines work in the human immune system?",
    "Explain recursion with a simple example.",
    # Add 40 more domain-specific prompts for your target use case
]

os.makedirs("./calibration_data", exist_ok=True)
for i, prompt in enumerate(CALIBRATION_PROMPTS):
    inputs = tokenizer(prompt, return_tensors="np", padding="max_length",
                       max_length=128, truncation=True)
    np.save(f"./calibration_data/input_ids_{i}.npy", inputs["input_ids"])
    np.save(f"./calibration_data/attention_mask_{i}.npy", inputs["attention_mask"])

print(f"[PROFILE] Generated {len(CALIBRATION_PROMPTS)} calibration samples")
```

### 4.2 Convert to QNN with INT8 Quantization
```bash
# This is the core conversion command
# Run on HOST machine (x86), generates HTP-optimized binary

qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --output_path ./models/qwen3-1.7b-qnn/model.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --quantization_overrides ./quantization_config.json \
  --act_bw 8 \
  --weight_bw 8 \
  --bias_bw 32 \
  --float_fallback \
  --keep_int64_inputs \
  --verbose \
  2>&1 | tee ./logs/qnn_conversion.log

# Compile to HTP-specific binary
qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn/model.cpp \
  -b ./models/qwen3-1.7b-qnn/model.bin \
  -o ./models/qwen3-1.7b-qnn/ \
  --lib_target aarch64-android \
  -t android_aarch64 \
  2>&1 | tee ./logs/qnn_compilation.log
```

### 4.3 Quantization Override Config for Transformer Precision
```json
// quantization_config.json
// Attention layers need higher precision — do NOT quantize uniformly
{
  "activation_encodings": {},
  "param_encodings": {
    // Keep embedding layer in fp16 — quantizing hurts quality severely
    "model.embed_tokens.weight": [{"bitwidth": 16, "dtype": "float"}],
    
    // Output LM head in fp16 — affects vocabulary distribution
    "lm_head.weight":            [{"bitwidth": 16, "dtype": "float"}],
    
    // Attention Q/K projections: INT8 is fine
    // (pattern: layers.N.self_attn.q_proj.weight)
    
    // Attention V/O projections: INT8 fine
    
    // FFN gate/up: INT8 fine
    // FFN down: consider INT16 if quality degrades
    
    // Layer norms: ALWAYS float — tiny params, critical for stability
    "model.norm.weight":         [{"bitwidth": 16, "dtype": "float"}]
  }
}
```

### 4.4 INT4 Conversion (After INT8 is Working)
```bash
# Only attempt after INT8 baseline is validated
# INT4 on HTP V73 uses dedicated dot-product engines — this is your biggest gain

qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --output_path ./models/qwen3-1.7b-qnn-int4/model.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --act_bw 8 \
  --weight_bw 4 \        # ← INT4 weights, INT8 activations (standard for HTP)
  --bias_bw 32 \
  --float_fallback \
  --quantization_overrides ./quantization_config_int4.json \
  --verbose \
  2>&1 | tee ./logs/qnn_int4_conversion.log
```

---

## 5. Android Runtime: Native Inference Engine

### 5.1 Project Structure
```
qnn_llm_runtime/
├── CMakeLists.txt
├── include/
│   ├── QnnRuntime.h
│   ├── Profiler.h
│   ├── KVCacheManager.h
│   ├── TokenizerWrapper.h
│   └── ThermalMonitor.h
├── src/
│   ├── QnnRuntime.cpp       ← core HTP execution
│   ├── Profiler.cpp         ← aggressive timing + logging
│   ├── KVCacheManager.cpp   ← pinned memory KV cache
│   ├── TokenizerWrapper.cpp ← sentencepiece JNI wrapper
│   ├── ThermalMonitor.cpp   ← throttle detection
│   └── main.cpp             ← CLI benchmark harness
├── jni/
│   └── InferenceJNI.cpp     ← Android JNI bridge
└── benchmarks/
    └── run_benchmarks.sh
```

### 5.2 Aggressive Profiler (Profiler.h / Profiler.cpp)
```cpp
// Profiler.h — embed this in EVERY inference function
#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <android/log.h>

#define LOG_TAG "QNN_RUNTIME"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// PROFILE_SCOPE: drop this at the start of any function you want timed
// It logs automatically on scope exit. Zero overhead in release if PROFILING=0
#ifdef PROFILING_ENABLED
#define PROFILE_SCOPE(name) ScopedTimer _timer_##__LINE__(name, g_profiler)
#define PROFILE_START(name) g_profiler.start(name)
#define PROFILE_END(name)   g_profiler.end(name)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_START(name)
#define PROFILE_END(name)
#endif

struct TimingEntry {
    std::string name;
    double duration_us;   // microseconds — ms is too coarse for layer-level work
    int64_t timestamp_us;
};

class Profiler {
public:
    void start(const std::string& name);
    void end(const std::string& name);

    // Call after every N tokens — dumps to logcat + file
    void dump_stats(const std::string& label);

    // Computes: mean, p50, p95, p99, min, max per named region
    void compute_percentiles();

    // Writes CSV to /sdcard/qnn_profile_TIMESTAMP.csv for offline analysis
    void write_to_file();

    // Resets all accumulators — call at start of each benchmark run
    void reset();

    // Per-layer breakdown — how many ms did each transformer layer take?
    void print_layer_breakdown();

private:
    std::unordered_map<std::string, std::vector<double>> timings_;
    std::unordered_map<std::string, 
        std::chrono::time_point<std::chrono::high_resolution_clock>> start_times_;
};

struct ScopedTimer {
    ScopedTimer(const std::string& name, Profiler& p) : name_(name), p_(p) {
        p_.start(name_);
    }
    ~ScopedTimer() { p_.end(name_); }
    std::string name_;
    Profiler& p_;
};

extern Profiler g_profiler;  // global — accessible from any file
```

### 5.3 Core QNN Runtime (QnnRuntime.cpp)
```cpp
// QnnRuntime.cpp — every QNN API call is profiled
#include "QnnRuntime.h"
#include "Profiler.h"
#include "QNN/QnnInterface.h"
#include "QNN/HTP/QnnHtpDevice.h"
#include "QNN/HTP/QnnHtpGraph.h"

class QnnRuntime {
public:
    bool initialize(const std::string& model_path) {
        PROFILE_SCOPE("QnnRuntime::initialize");

        // ── 1. Load QNN HTP backend ──────────────────────────────────────
        {
            PROFILE_SCOPE("load_htp_backend");
            backend_lib_handle_ = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
            if (!backend_lib_handle_) {
                LOGE("Failed to load libQnnHtp.so: %s", dlerror());
                return false;
            }
        }

        // ── 2. Create QNN Backend ────────────────────────────────────────
        {
            PROFILE_SCOPE("create_backend");
            QnnHtpDevice_CustomConfig_t device_config = {};
            device_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_PERFORMANCE_INFRASTRUCTURE;
            // Set HTP to sustained high performance mode
            device_config.performanceInfrastructure.powerConfigId = 1;

            Qnn_BackendConfig_t backend_cfg = QNN_BACKEND_CONFIG_INIT;
            auto status = qnn_interface_.backendCreate(
                log_handle_, nullptr, 0, &backend_handle_);
            LOG_QNN_STATUS("backendCreate", status);
        }

        // ── 3. Load Model ────────────────────────────────────────────────
        {
            PROFILE_SCOPE("load_model_from_file");
            // Load pre-compiled .so (the compiled QNN model binary)
            model_lib_handle_ = dlopen(model_path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!model_lib_handle_) {
                LOGE("Failed to load model: %s", dlerror());
                return false;
            }
            LOGI("[PROFILE] Model binary loaded successfully");
        }

        // ── 4. Create Graph ──────────────────────────────────────────────
        {
            PROFILE_SCOPE("create_graph");
            QnnHtpGraph_CustomConfig_t graph_config = {};
            graph_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
            // Enable all HTP V73 optimizations
            graph_config.optimizationOption.type = 
                QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
            graph_config.optimizationOption.floatValue = 3.0f;  // max opt level

            auto status = qnn_interface_.graphCreate(
                context_handle_, "qwen3_decode", nullptr, 0, &graph_handle_);
            LOG_QNN_STATUS("graphCreate", status);
        }

        log_memory_usage("post_initialize");
        return true;
    }

    // Single token decode — this is the hot path
    // Profile every microsecond here
    std::vector<float> decode_step(
        const std::vector<int32_t>& input_ids,
        const KVCacheState& kv_state,
        int position_id)
    {
        PROFILE_SCOPE("decode_step_total");

        // ── Input preparation ────────────────────────────────────────────
        {
            PROFILE_SCOPE("decode_input_prep");
            set_tensor("input_ids", input_ids);
            set_tensor("position_ids", {position_id});
            set_kv_cache_tensors(kv_state);  // pre-pinned memory — no copy
        }

        // ── HTP Graph Execute ────────────────────────────────────────────
        {
            PROFILE_SCOPE("htp_graph_execute");  // ← this is your core metric
            auto status = qnn_interface_.graphExecute(
                graph_handle_,
                input_tensors_.data(), input_tensors_.size(),
                output_tensors_.data(), output_tensors_.size(),
                nullptr, nullptr
            );
            if (status != QNN_SUCCESS) {
                LOGE("[PROFILE] graphExecute failed: %d", status);
                return {};
            }
        }

        // ── Output extraction ────────────────────────────────────────────
        {
            PROFILE_SCOPE("decode_output_extract");
            return extract_logits();
        }
    }

    // Prefill — process full prompt in one graph execute
    // Different profile target: time-to-first-token
    std::vector<float> prefill(
        const std::vector<int32_t>& prompt_ids)
    {
        PROFILE_SCOPE("prefill_total");
        LOGI("[PROFILE] Prefill: %zu tokens", prompt_ids.size());

        int64_t t0 = get_time_us();

        {
            PROFILE_SCOPE("prefill_input_prep");
            set_tensor("input_ids", prompt_ids);
            // attention_mask: all 1s for prompt
            std::vector<int32_t> mask(prompt_ids.size(), 1);
            set_tensor("attention_mask", mask);
        }

        {
            PROFILE_SCOPE("prefill_htp_execute");
            auto status = qnn_interface_.graphExecute(
                graph_handle_,
                input_tensors_.data(), input_tensors_.size(),
                output_tensors_.data(), output_tensors_.size(),
                nullptr, nullptr
            );
            LOG_QNN_STATUS("prefill_execute", status);
        }

        int64_t t1 = get_time_us();
        double prefill_ms = (t1 - t0) / 1000.0;
        double prefill_tps = prompt_ids.size() / (prefill_ms / 1000.0);
        LOGI("[PROFILE] Prefill: %.1fms | %.1f tokens/sec", prefill_ms, prefill_tps);

        return extract_logits();
    }

private:
    void log_memory_usage(const std::string& label) {
        // Read /proc/self/status for RSS
        FILE* f = fopen("/proc/self/status", "r");
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "VmRSS:", 6) == 0 ||
                strncmp(line, "VmPeak:", 7) == 0) {
                LOGI("[PROFILE][MEM][%s] %s", label.c_str(), line);
            }
        }
        fclose(f);
    }

    int64_t get_time_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
    }

    // ... QNN handles, tensor state, etc.
};
```

### 5.4 KV Cache Manager (KVCacheManager.h)
```cpp
// KVCacheManager.h
// CRITICAL: KV cache must NEVER be copied during decode loop
// Allocate once, reuse across all tokens via pointer passing

class KVCacheManager {
public:
    // Pre-allocate for max_seq_len at startup
    // Never reallocate during inference — this kills performance
    bool allocate(int num_layers, int num_kv_heads, int head_dim, int max_seq_len) {
        PROFILE_SCOPE("kv_cache_allocate");
        
        size_t bytes_per_layer = num_kv_heads * head_dim * max_seq_len * sizeof(uint16_t);
        size_t total_bytes = num_layers * 2 * bytes_per_layer;  // K and V
        
        LOGI("[PROFILE][KV] Allocating %.1f MB for KV cache", total_bytes / 1e6);
        LOGI("[PROFILE][KV] Config: layers=%d, kv_heads=%d, head_dim=%d, max_len=%d",
             num_layers, num_kv_heads, head_dim, max_seq_len);
        
        // Use mmap with MADV_HUGEPAGE for better TLB performance
        kv_buffer_ = mmap(nullptr, total_bytes,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (kv_buffer_ == MAP_FAILED) {
            LOGE("[PROFILE][KV] mmap failed!");
            return false;
        }
        
        // Lock pages in RAM — prevent Android from swapping KV cache
        if (mlock(kv_buffer_, total_bytes) != 0) {
            LOGW("[PROFILE][KV] mlock failed — KV cache may be paged out under pressure");
        }
        
        // Advise kernel: we access sequentially per token
        madvise(kv_buffer_, total_bytes, MADV_SEQUENTIAL);
        
        total_bytes_ = total_bytes;
        num_layers_ = num_layers;
        current_seq_len_ = 0;
        
        LOGI("[PROFILE][KV] Allocation complete. %.1f MB locked in RAM", total_bytes / 1e6);
        return true;
    }

    // Zero-copy pointer into pre-allocated buffer for layer N
    void* get_k_ptr(int layer, int seq_pos) { /* ... */ }
    void* get_v_ptr(int layer, int seq_pos) { /* ... */ }
    
    void log_cache_stats() {
        LOGI("[PROFILE][KV] Current seq len: %d / %d (%.1f%% full)",
             current_seq_len_, max_seq_len_,
             100.0f * current_seq_len_ / max_seq_len_);
    }

private:
    void*  kv_buffer_   = nullptr;
    size_t total_bytes_ = 0;
    int    num_layers_  = 0;
    int    max_seq_len_ = 0;
    int    current_seq_len_ = 0;
};
```

### 5.5 Thermal Monitor (ThermalMonitor.h)
```cpp
// ThermalMonitor.h
// Snapdragon 7s Gen 3 WILL throttle — detect it before it destroys your benchmark

class ThermalMonitor {
public:
    float get_cpu_temp() {
        // Read from thermal zone — zone 0 is typically CPU on SD 7s Gen 3
        FILE* f = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
        float temp_milli = 0;
        fscanf(f, "%f", &temp_milli);
        fclose(f);
        return temp_milli / 1000.0f;
    }
    
    bool is_throttling() {
        // Compare current CPU freq vs max freq
        FILE* f = fopen("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq", "r");
        int cur = 0; fscanf(f, "%d", &cur); fclose(f);
        
        f = fopen("/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq", "r");
        int max = 0; fscanf(f, "%d", &max); fclose(f);
        
        float ratio = (float)cur / max;
        if (ratio < 0.85f) {
            LOGW("[PROFILE][THERMAL] THROTTLING DETECTED: CPU at %.0f%% max freq (%.0f°C)",
                 ratio * 100, get_cpu_temp());
            return true;
        }
        return false;
    }
    
    void log_thermal_snapshot(const std::string& label) {
        LOGI("[PROFILE][THERMAL][%s] CPU: %.1f°C | Throttling: %s",
             label.c_str(), get_cpu_temp(), is_throttling() ? "YES" : "NO");
    }
};
```

---

## 6. Benchmark Harness

### 6.1 Main Benchmark Script (C++)
```cpp
// benchmarks/benchmark_main.cpp
int main() {
    QnnRuntime runtime;
    ThermalMonitor thermal;
    
    LOGI("═══════════════════════════════════════");
    LOGI("QNN HTP Runtime Benchmark");
    LOGI("Target: Qwen3-1.7B INT4 on Hexagon V73");
    LOGI("═══════════════════════════════════════");
    
    // ── Phase 1: Load model ──────────────────
    thermal.log_thermal_snapshot("pre_load");
    runtime.initialize("/data/local/tmp/qwen3_1.7b_int4.so");
    thermal.log_thermal_snapshot("post_load");
    
    // ── Phase 2: Warmup (3 runs, not counted) ─
    LOGI("[BENCHMARK] Warming up (3 runs)...");
    for (int i = 0; i < 3; i++) {
        run_decode_loop(runtime, TEST_PROMPT_SHORT, 20);
    }
    
    // ── Phase 3: Thermal soak (wait for steady state)
    LOGI("[BENCHMARK] Thermal soak: running for 60 seconds...");
    auto soak_end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < soak_end) {
        run_decode_loop(runtime, TEST_PROMPT_SHORT, 10);
    }
    thermal.log_thermal_snapshot("post_soak");
    
    // ── Phase 4: Actual benchmark ────────────
    g_profiler.reset();
    LOGI("[BENCHMARK] Starting timed benchmark...");
    
    // Test 1: Short prompt decode speed
    {
        auto result = run_decode_loop(runtime, TEST_PROMPT_SHORT, 100);
        LOGI("[RESULT] Short prompt | 100 tokens | %.2f tok/s | ttft=%.1fms",
             result.tokens_per_sec, result.time_to_first_token_ms);
    }
    
    // Test 2: Long context (near your max)
    {
        auto result = run_decode_loop(runtime, TEST_PROMPT_LONG_400_TOKENS, 100);
        LOGI("[RESULT] Long context | 100 tokens | %.2f tok/s | ttft=%.1fms",
             result.tokens_per_sec, result.time_to_first_token_ms);
    }
    
    // Test 3: Sustained 120-second throughput
    {
        int total_tokens = 0;
        auto start = std::chrono::steady_clock::now();
        auto end   = start + std::chrono::seconds(120);
        while (std::chrono::steady_clock::now() < end) {
            auto result = run_decode_loop(runtime, TEST_PROMPT_SHORT, 50);
            total_tokens += 50;
            if (thermal.is_throttling()) {
                LOGW("[RESULT] Throttle event at token %d", total_tokens);
            }
        }
        double sustained_tps = total_tokens / 120.0;
        LOGI("[RESULT] SUSTAINED 120s: %.2f tok/s (total: %d tokens)", 
             sustained_tps, total_tokens);
    }
    
    // ── Phase 5: Dump full profiler report ───
    g_profiler.dump_stats("final");
    g_profiler.print_layer_breakdown();
    g_profiler.write_to_file();
    
    thermal.log_thermal_snapshot("benchmark_complete");
    
    return 0;
}
```

### 6.2 ADB Benchmark Runner Script
```bash
#!/bin/bash
# benchmarks/run_benchmarks.sh
# Run this script — it handles everything

set -e
DEVICE_DIR=/data/local/tmp/qnn_runtime
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE=./logs/benchmark_${DATE}.log

echo "═══════════════════════════════════════"
echo "QNN HTP Benchmark Runner"
echo "Date: $(date)"
echo "═══════════════════════════════════════"

# Push latest binary
adb push ./build-android/qnn_benchmark $DEVICE_DIR/
adb push ./models/qwen3-1.7b-qnn-int4/libqwen3_model.so $DEVICE_DIR/

# Set performance mode
adb shell "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
adb shell "echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"

# Log device state before benchmark
echo "--- Device State ---" | tee -a $LOG_FILE
adb shell cat /sys/class/thermal/thermal_zone0/temp | tee -a $LOG_FILE
adb shell cat /proc/meminfo | grep -E "MemFree|MemAvailable" | tee -a $LOG_FILE

# Run benchmark
echo "--- Benchmark Output ---" | tee -a $LOG_FILE
adb shell "cd $DEVICE_DIR && \
  LD_LIBRARY_PATH=$DEVICE_DIR \
  ./qnn_benchmark \
  --model libqwen3_model.so \
  --backend libQnnHtp.so \
  --iterations 5 \
  --warmup 3 \
  --output-tokens 100 \
  --log-level verbose \
  2>&1" | tee -a $LOG_FILE

# Pull profiler CSV off device
adb pull /sdcard/qnn_profile_*.csv ./logs/ 2>/dev/null || true

echo "═══════════════════════════════════════"
echo "Benchmark complete. Log: $LOG_FILE"
echo "═══════════════════════════════════════"
```

---

## 7. Profiling Data Analysis

### 7.1 Parse & Visualize Profile CSV
```python
# analyze_profile.py
import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])  # the CSV written by Profiler::write_to_file()

# ── Layer latency breakdown ──────────────────────────────────────────────────
layer_data = df[df['name'].str.contains('layer_')]
layer_avg = layer_data.groupby('name')['duration_us'].mean().sort_values()

plt.figure(figsize=(12, 6))
layer_avg.plot(kind='barh')
plt.title('Per-layer Latency Breakdown (μs)')
plt.xlabel('Average duration (μs)')
plt.tight_layout()
plt.savefig('./logs/layer_breakdown.png', dpi=150)
print("Saved layer_breakdown.png")

# ── Token throughput over time (shows thermal throttle) ─────────────────────
decode_times = df[df['name'] == 'decode_step_total']['duration_us']
tps_series = 1e6 / decode_times  # convert μs/token → tokens/sec

plt.figure(figsize=(14, 4))
plt.plot(tps_series.values)
plt.axhline(y=tps_series.mean(), color='r', linestyle='--', label=f'Mean: {tps_series.mean():.1f} tok/s')
plt.axhline(y=tps_series.quantile(0.05), color='orange', linestyle='--', label=f'P5: {tps_series.quantile(0.05):.1f} tok/s')
plt.title('Token Generation Speed Over Time (tok/s)')
plt.xlabel('Token index')
plt.ylabel('Tokens/sec')
plt.legend()
plt.tight_layout()
plt.savefig('./logs/throughput_timeline.png', dpi=150)
print("Saved throughput_timeline.png")

# ── Print summary stats ──────────────────────────────────────────────────────
print("\n═══════════════════════════════════════════")
print("INFERENCE PERFORMANCE SUMMARY")
print("═══════════════════════════════════════════")
print(f"Mean tok/s:    {tps_series.mean():.2f}")
print(f"Median tok/s:  {tps_series.median():.2f}")
print(f"P95 tok/s:     {tps_series.quantile(0.95):.2f}")
print(f"Min tok/s:     {tps_series.min():.2f}  ← throttle floor")
print(f"Max tok/s:     {tps_series.max():.2f}  ← peak burst")

htp_time = df[df['name'] == 'htp_graph_execute']['duration_us'].mean()
prep_time = df[df['name'] == 'decode_input_prep']['duration_us'].mean()
extract_time = df[df['name'] == 'decode_output_extract']['duration_us'].mean()
total_time = df[df['name'] == 'decode_step_total']['duration_us'].mean()

print(f"\nTime breakdown per token (mean):")
print(f"  HTP execute:    {htp_time:.0f} μs  ({100*htp_time/total_time:.1f}%)")
print(f"  Input prep:     {prep_time:.0f} μs  ({100*prep_time/total_time:.1f}%)")
print(f"  Output extract: {extract_time:.0f} μs  ({100*extract_time/total_time:.1f}%)")
print(f"  Total:          {total_time:.0f} μs")
print("═══════════════════════════════════════════")
```

---

## 8. Comparison Table (Fill This In As You Progress)

```
File: results/comparison.md — update after every milestone

| Config                        | Prefill (tok/s) | Decode (tok/s) | Sustained 120s | TTFT (512 tok) |
|-------------------------------|-----------------|----------------|----------------|----------------|
| llama.cpp Q4_0 CPU baseline   |                 |                |                |                |
| llama.cpp Q4_0 Vulkan         |                 |                |                |                |
| QNN INT8 (ONNX → HTP)         |                 |                |                |                |
| QNN INT4 (ONNX → HTP)         |                 |                |                |                |
| QNN INT4 + KV cache pinned    |                 |                |                |                |
| QNN INT4 + async pipeline     |                 |                |                |                |
| QNN INT4 + speculative decode |                 |                |                |                |
```

---

## 9. Red Flags & Debug Checklist

If decode speed is slower than llama.cpp after QNN conversion, check these in order:

- [ ] Is `libQnnHtpV73Skel.so` present on device? Without it, QNN silently falls back to CPU.
- [ ] Is the HTP performance mode actually set? Check via `adb shell cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq`
- [ ] Are you profiling after warmup? First 2–3 runs include JIT compilation overhead.
- [ ] Is your quantization calibration dataset representative? Bad calibration → bad INT8 accuracy → you compensate with higher bitwidth → slower.
- [ ] Is the KV cache getting paged out? Check `mlock` return value explicitly.
- [ ] Are non-HTP-compatible ops falling back to CPU? Check conversion log for `[FALLBACK]` warnings.
- [ ] Is the device thermally throttled? Check `is_throttling()` output.
- [ ] Are you running on big cores? `taskset 0xF0` for cores 4–7 (Cortex-A715 cluster).

---

## 10. The Demo That Gets You Hired

Your final artifact must include all of the following:

1. **GitHub repo** with this full project, clean README, build instructions
2. **Benchmark video** on device — show `adb logcat` output streaming with tok/s numbers
3. **Comparison table** — llama.cpp baseline vs your QNN runtime, same device, same model
4. **Profile visualisation** — the layer breakdown PNG and throughput timeline PNG
5. **One blog post** — title format: *"Running Qwen3-1.7B at Xtok/s on a ₹20,000 Android phone: a QNN HTP deep dive"*
6. **Reproducible** — anyone with the same hardware must be able to reproduce your numbers

Post the blog to: Hacker News, r/LocalLLaMA, r/androiddev, X/Twitter with #EdgeAI #OnDeviceAI #QNN
```
