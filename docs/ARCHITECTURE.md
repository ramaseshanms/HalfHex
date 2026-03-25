# Architecture — HalfHex QNN LLM Runtime

## System Overview

```
                            INFERENCE PIPELINE
                            ==================

  User Prompt                                              Generated Text
       |                                                        ^
       v                                                        |
  +------------+     +----------+     +-----------+     +------------+
  | Tokenizer  | --> | Prefill  | --> | Decode    | --> | Detokenize |
  | (CPU)      |     | (HTP)    |     | Loop (HTP)|     | (CPU)      |
  +------------+     +----------+     +-----------+     +------------+
                          |               |   ^
                          v               v   |
                     +---------------------------+
                     | KV Cache (mmap + mlock)    |
                     | 28 layers x 4 heads x 128d |
                     | ~57 MB pinned in RAM       |
                     +---------------------------+

  Monitored by:
    [MemoryGuard] /proc/self/status, /proc/meminfo
    [ThermalMonitor] /sys/class/thermal/thermal_zone*
    [Profiler] PROFILE_SCOPE on every function
```

---

## Namespace: halfhex

All C++ code lives under `namespace halfhex {}`. This prevents symbol
collisions when linking against QNN SDK libraries (which use C-style global
symbols) and SentencePiece (which uses `sentencepiece::` namespace).

Global instances:
- `halfhex::g_profiler` — Profiler singleton
- `halfhex::g_memory_guard` — MemoryGuard singleton

---

## Component Design

### 1. Profiler (Profiler.h / Profiler.cpp)

**Purpose:** Capture microsecond timing for every inference function.

**Design:**
- Ring buffer pre-allocated to 1M entries (~48 MB) to avoid `malloc` on hot path
- `PROFILE_SCOPE("name")` macro creates a `ScopedTimer` that records on scope exit
- Compile-time disable: when `PROFILING_ENABLED` is not defined, all macros expand to `(void)0`
- Thread-safe via `std::mutex` (uncontended in single-threaded decode)

**Data flow:**
```
PROFILE_SCOPE("decode_step")
  → ScopedTimer constructor → g_profiler.start("decode_step")
  → ... inference work ...
  → ScopedTimer destructor → g_profiler.end("decode_step")
      → TimingEntry { name, duration_us, timestamp_us } appended to entries_
      → duration appended to timings_["decode_step"]
```

**Output formats:**
- `dump_stats()` → logcat with percentile table
- `write_to_file()` → CSV for pandas/matplotlib analysis
- `print_layer_breakdown()` → sorted by duration, percentage of total

### 2. MemoryGuard (MemoryGuard.h / MemoryGuard.cpp)

**Purpose:** Prevent inference from triggering Android's Low Memory Killer.

**Budget (Qwen3-1.7B):**

| Category       | Budget   | Actual   | Notes                          |
|----------------|----------|----------|--------------------------------|
| Model weights  | 1200 MB  | ~1000 MB | INT4 weights loaded via dlopen |
| KV cache       | 256 MB   | ~57 MB   | 28L x 4KV x 128d x 512 x fp16|
| Activations    | 128 MB   | ~100 MB  | Intermediate tensors           |
| Runtime        | ~200 MB  | ~150 MB  | Profiler ring + overhead       |
| **Total RSS**  | **3 GB** | ~1.3 GB  | Conservative for safety        |
| **Sys reserve**| **1 GB** |          | Never let MemAvailable drop    |

**How it works:**
1. At startup: reads `/proc/meminfo` for MemAvailable, sets oom_score_adj=800
2. Before allocation: `can_allocate()` checks both RSS budget and system available
3. During inference: `should_emergency_stop()` checked every 50 tokens
4. Atomic counters track allocations by category (relaxed ordering — sufficient)

### 3. KVCacheManager (KVCacheManager.h / KVCacheManager.cpp)

**Purpose:** Zero-copy pinned memory for transformer KV cache.

**Memory layout (contiguous):**
```
Buffer start → [L0_K][L0_V][L1_K][L1_V]...[L27_K][L27_V] ← Buffer end

Each K or V block:
  num_kv_heads (4) × head_dim (128) × max_seq_len (512) × sizeof(uint16_t) (2)
  = 4 × 128 × 512 × 2 = 524,288 bytes per block

Total: 28 layers × 2 (K+V) × 524,288 = 29,360,128 bytes ≈ 28 MB
(Header says ~57 MB because it includes alignment padding)
```

**Memory strategy:**
1. `mmap(MAP_PRIVATE | MAP_ANONYMOUS)` — no file backing, kernel zero-fills
2. `mlock()` — pin in physical RAM, prevent zRAM swap (critical on Android)
3. `madvise(MADV_SEQUENTIAL)` — hint kernel for sequential prefetch
4. `memset(0)` before `munmap()` — security: prevent data leakage

**Access pattern:**
```
get_k_ptr(layer=5, seq_pos=42)
  → base + (5 × 2 × kv_block) + (42 × 4 × 128 × 2)
  → Direct pointer passed to QNN graph execute (ZERO COPY)
```

### 4. ThermalMonitor (ThermalMonitor.h / ThermalMonitor.cpp)

**Purpose:** Detect thermal throttling before it corrupts benchmark data.

**Nothing Phone 3a Pro thermal zones:**
| Zone | Name           | Measures           | Typical |
|------|----------------|--------------------|---------|
| 0    | shell_front    | Front panel temp   | 33.8°C  |
| 1    | shell_frame    | Metal frame temp   | 33.7°C  |
| 2    | shell_back     | Back panel temp    | 33.2°C  |
| 3    | shell_max      | Max of all shells  | 33.7°C  |
| 4    | pmxr2230-bcl-0 | Battery current    | 0       |

**Throttle detection:**
```
cpu4_current_freq / cpu4_max_freq < 0.85 → THROTTLED
```
Reads `/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq` and `cpuinfo_max_freq`.

**Threshold tiers:**
- < 40°C: NORMAL — proceed
- 40-50°C: WARNING — log, flag benchmark data
- 50-55°C: CRITICAL — abort benchmark
- > 55°C: EMERGENCY — terminate process

### 5. QnnRuntime (QnnRuntime.h / QnnRuntime.cpp)

**Purpose:** Interface to Qualcomm QNN SDK for Hexagon HTP execution.

**Initialization sequence:**
```
1. dlopen("libQnnHtp.so")           → backend_lib_handle_
2. QnnInterface_getProviders()      → function table
3. backendCreate() + deviceCreate() → backend_handle_, device_handle_
4. dlopen(model_path)               → model_lib_handle_
5. QnnModel_composeGraphs()         → graph_handle_
6. graphFinalize()                  → ready for execute
```

**Hot path (decode_step):**
```
decode_step_total
  ├── decode_input_prep     (bind tensors: input_ids, position_ids, KV cache ptrs)
  ├── htp_graph_execute     (QNN graphExecute → Hexagon V73 → result)
  └── decode_output_extract (read logits from output tensor)
```

### 6. TokenizerWrapper (TokenizerWrapper.h / TokenizerWrapper.cpp)

**Purpose:** SentencePiece BPE tokenizer for Qwen3-1.7B.

NOT on the critical path — runs once for prefill, once per token for decode output.

---

## Benchmark Protocol (5 Phases)

| Phase | Duration | Purpose                                         |
|-------|----------|-------------------------------------------------|
| 1     | ~5s      | Initialize runtime, allocate KV cache           |
| 2     | ~10s     | Warmup (3 runs) — JIT compilation, cache warm   |
| 3     | 60s      | Thermal soak — reach steady-state temperature   |
| 4     | ~5min    | Timed benchmarks (short, long, sustained 120s)  |
| 5     | ~1s      | Dump profiler stats, write CSV, log final state |

**Why thermal soak?** Mobile SoCs boost to high clocks for brief periods
then throttle. Cold benchmarks show 2x the actual sustained performance.
The 60-second soak ensures we measure real-world throughput.

---

## Thread Model

**Single-threaded inference.** The HTP graph execute call is synchronous
and blocks until the Hexagon DSP completes. There is no benefit to
multi-threading the decode loop because:
1. HTP V73 parallelizes internally across HVX threads
2. KV cache access is sequential (one position at a time)
3. The profiler mutex is uncontended in single-threaded mode

The JNI bridge documents that callers must ensure single-threaded access.

---

## Build Configuration

| CMake Flag          | Default | Effect                                |
|---------------------|---------|---------------------------------------|
| PROFILING_ENABLED   | ON      | Enable PROFILE_SCOPE macros (+48MB)   |
| BUILD_JNI           | OFF     | Build JNI .so for Android app         |
| QNN_SDK_ROOT        | (none)  | Path to QNN SDK headers               |
| CMAKE_BUILD_TYPE    | Release | -O3 -ffast-math -s for Android        |

Compiler flags:
- `-Wall -Wextra -Wpedantic` — comprehensive warnings
- `-Werror=return-type` — missing return is always a bug
- `-Werror=uninitialized` — prevents silent memory corruption
- `-O3 -ffast-math` — maximum optimization for tensor prep code
