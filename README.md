# HalfHex — QNN-Native LLM Inference Runtime

A high-performance inference runtime that executes Qwen3-1.7B directly on the
Hexagon Tensor Processor (HTP) of the Qualcomm Snapdragon 7s Gen 3, bypassing
CPU and GPU entirely. Built for the Nothing Phone 3a Pro, with production-grade
OOM protection, thermal safety, and microsecond-level profiling.

---

## Target Hardware

| Component       | Specification                                          |
|-----------------|--------------------------------------------------------|
| Device          | Nothing Phone 3a Pro (A059P)                           |
| SoC             | Qualcomm Snapdragon 7s Gen 3 (codename "volcano")     |
| NPU             | Hexagon V73 with HTP (Tensor Processor)                |
| CPU             | 4x Cortex-A510 @1.8GHz + 4x Cortex-A715 @2.4GHz      |
| RAM             | 12 GB LPDDR5 (~7 GB available under normal use)        |
| Storage         | 256 GB UFS 2.2 (132 GB free)                           |
| OS              | Android 15                                             |
| Model           | Qwen3-1.7B (28 layers, 4 KV heads, 128 head_dim)      |
| Quantization    | INT4 weights / INT8 activations (W4A8) on HTP V73      |
| Context Window  | 512 tokens                                             |

---

## Architecture

```
Host Machine                          Nothing Phone 3a Pro
+------------------+                  +----------------------------------+
|                  |                  |  /data/local/tmp/halfhex/        |
| HuggingFace      |                  |  +----------------------------+  |
| Qwen3-1.7B (fp16)|                  |  | OOM Guard (oom_guard.sh)   |  |
|       |          |                  |  |  - RSS limit: 3 GB         |  |
|       v          |                  |  |  - Temp limit: 55°C        |  |
| export_to_onnx.py|                  |  |  - Sys reserve: 1 GB       |  |
|       |          |                  |  +------|---------------------+  |
|       v          |                  |         |                        |
| ONNX (fp32)      |                  |         v                        |
|       |          |                  |  +----------------------------+  |
|       v          |    adb push      |  | qnn_benchmark              |  |
| qnn-onnx-converter| ------------->  |  |  Profiler (ring buffer)    |  |
|       |          |                  |  |  MemoryGuard (/proc)       |  |
|       v          |                  |  |  KVCacheManager (mmap)     |  |
| QNN INT4 (.so)   |                  |  |  ThermalMonitor (sysfs)    |  |
+------------------+                  |  +------|---------------------+  |
                                      |         |                        |
                                      |         v                        |
                                      |  +----------------------------+  |
                                      |  | libQnnHtp.so → Hexagon V73 |  |
                                      |  | INT4 dot-product engines   |  |
                                      |  +----------------------------+  |
                                      +----------------------------------+
```

---

## Project Structure

```
HalfHex/
├── CLAUDE.md                          # Project instructions for dev sessions
├── README.md                          # This file
├── .claude/settings.json              # No AI attribution in commits
├── .gitignore                         # Excludes models, builds, logs
│
├── qnn_llm_runtime/                   # C++ native runtime
│   ├── CMakeLists.txt                 # Android NDK cross-compilation config
│   ├── include/
│   │   ├── Profiler.h                 # Zero-overhead profiling macros (PROFILE_SCOPE)
│   │   ├── MemoryGuard.h             # OOM protection + memory budget enforcement
│   │   ├── QnnRuntime.h              # QNN HTP backend interface
│   │   ├── KVCacheManager.h          # mmap+mlock pinned KV cache
│   │   ├── ThermalMonitor.h          # Thermal zones + throttle detection
│   │   └── TokenizerWrapper.h        # SentencePiece tokenizer
│   ├── src/
│   │   ├── main.cpp                   # 5-phase benchmark harness
│   │   ├── Profiler.cpp              # Ring buffer profiler + CSV export
│   │   ├── MemoryGuard.cpp           # /proc parsing + budget enforcement
│   │   ├── QnnRuntime.cpp            # QNN SDK dlopen + graph execute
│   │   ├── KVCacheManager.cpp        # mmap/mlock/madvise + zero-copy
│   │   ├── ThermalMonitor.cpp        # sysfs thermal zone reading
│   │   └── TokenizerWrapper.cpp      # SentencePiece integration scaffold
│   ├── jni/
│   │   └── InferenceJNI.cpp          # Android app JNI bridge
│   └── benchmarks/
│       └── run_benchmarks.sh          # ADB benchmark runner
│
├── device/                            # Device-side scripts
│   └── sandbox_init.sh               # Creates isolated /data/local/tmp/halfhex/
│                                      # with OOM guard + health monitor + cleanup
│
├── docs/                              # Documentation
│   ├── BUILD_GUIDE.md                # 14-step build pipeline
│   ├── ARCHITECTURE.md               # Internal architecture reference
│   ├── SECURITY.md                   # Security properties
│   └── DEVICE_SAFETY.md             # Phone protection measures
│
├── export_to_onnx.py                  # HuggingFace → ONNX export
├── generate_calibration_data.py       # INT8/INT4 quantization calibration
├── analyze_profile.py                 # Profile CSV → matplotlib visualization
│
├── quantization_config.json           # INT8 precision overrides
├── quantization_config_int4.json      # INT4 precision overrides
│
├── setup_env.sh                       # Host environment setup
├── setup_device.sh                    # ADB device preparation
├── build_android.sh                   # CMake cross-compilation wrapper
├── convert_to_qnn.sh                 # Full ONNX → QNN conversion pipeline
├── run_baseline.sh                    # llama.cpp Day 1 baseline
│
├── baselines/
│   └── llamacpp_baseline.txt          # Template for baseline measurements
├── results/
│   └── comparison.md                  # Performance comparison table
├── models/                            # Model files (gitignored)
├── calibration_data/                  # Calibration .npy files (gitignored)
├── logs/                              # Profiling output (gitignored)
└── qnn_htp_runtime_prompt.md          # Original specification document
```

---

## Quick Start

```bash
# 1. Clone
git clone git@github.com:ramaseshanms/HalfHex.git
cd HalfHex

# 2. Set up host environment
source setup_env.sh

# 3. Export model and convert to QNN
python export_to_onnx.py
python generate_calibration_data.py
bash convert_to_qnn.sh

# 4. Build runtime for Android
bash build_android.sh

# 5. Deploy and benchmark
bash setup_device.sh
bash qnn_llm_runtime/benchmarks/run_benchmarks.sh
```

See [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) for detailed instructions.

---

## Safety Features

HalfHex is designed to run on a daily-driver phone without affecting
normal use. Three independent safety systems protect the device:

### 1. OOM Guard (MemoryGuard + oom_guard.sh)
- Sets `oom_score_adj=800` — Android kills our process before any user app
- Hard RSS limit: 3 GB (of 12 GB total, ~7 GB available)
- System reserve: refuses allocation if MemAvailable would drop below 1 GB
- Memory tracked by category: KV cache, model weights, activations

### 2. Thermal Kill-Switch (ThermalMonitor)
- Normal: < 40°C (no action)
- Warning: 40-50°C (logged, benchmark data flagged)
- Critical: 50-55°C (benchmark aborted)
- Emergency: > 55°C (process terminated immediately)
- Throttle detection via CPU frequency ratio (big cores at cpu4-7)

### 3. Device Sandbox (sandbox_init.sh)
- All files confined to `/data/local/tmp/halfhex/`
- Never modifies /system, /vendor, /product
- Never installs APKs or touches app data
- Complete cleanup script with safety verification
- Health monitor logs memory/thermal/frequency timeline to CSV

---

## Performance Targets

Fill in as you progress. All numbers measured after 60-second thermal soak.

| Configuration                 | Prefill (tok/s) | Decode (tok/s) | Sustained 120s | TTFT (512 tok) |
|-------------------------------|-----------------|----------------|----------------|----------------|
| llama.cpp Q4_0 CPU baseline   |                 |                |                |                |
| llama.cpp Q4_0 Vulkan         |                 |                |                |                |
| QNN INT8 (ONNX → HTP)        |                 |                |                |                |
| QNN INT4 (ONNX → HTP)        |                 |                |                |                |
| QNN INT4 + KV cache pinned   |                 |                |                |                |
| QNN INT4 + async pipeline     |                 |                |                |                |

---

## Troubleshooting

If decode speed is slower than llama.cpp after QNN conversion:

1. Is `libQnnHtpV73Skel.so` on the device? Without it, QNN silently falls back to CPU.
2. Is HTP performance mode set? Check `adb shell cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq`
3. Are you profiling after warmup? First 2-3 runs include JIT compilation.
4. Is calibration data representative? Bad calibration → bad quantization → slower.
5. Is the KV cache mlocked? Check logcat for "mlock succeeded".
6. Are non-HTP ops falling back to CPU? Check conversion log for `[FALLBACK]`.
7. Is the device thermally throttled? Check `[THERMAL]` output.
8. Running on big cores? Use `taskset 0xF0` for cores 4-7.
9. System memory pressure? Check `[MEMGUARD]` output for available RAM.

---

## Documentation

| Document | Description |
|----------|-------------|
| [BUILD_GUIDE.md](docs/BUILD_GUIDE.md) | Step-by-step build from source |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Internal design and data flow |
| [SECURITY.md](docs/SECURITY.md) | Security properties of the codebase |
| [DEVICE_SAFETY.md](docs/DEVICE_SAFETY.md) | How the phone is protected |

---

## License

To be determined.
