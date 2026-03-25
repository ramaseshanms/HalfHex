# Device Safety — Protecting the Nothing Phone 3a Pro

This document explains how HalfHex protects the device during LLM inference.
Read this before running anything on the phone.

---

## Device Specifications

| Property     | Value                                         |
|--------------|-----------------------------------------------|
| Model        | Nothing Phone 3a Pro (A059P)                  |
| Serial       | 00178353A000566                                |
| SoC          | Snapdragon 7s Gen 3 ("volcano")               |
| RAM          | 12 GB (11,711,820 kB)                          |
| Available    | ~7 GB under normal use                         |
| Swap (zRAM)  | 9.4 GB                                         |
| Storage      | 225 GB (132 GB free)                           |
| Android      | 15                                             |
| Baseline temp| 33.2-33.8°C                                    |
| Battery      | 90% at initial setup                           |

---

## Sandbox Directory Layout

All HalfHex files are confined to one directory tree:

```
/data/local/tmp/halfhex/           ← SANDBOX ROOT
├── .sandbox_meta                   ← Identity file (required for cleanup)
├── .pid                            ← PID of running process (if any)
├── sandbox_init.log               ← Baseline device state at creation
├── cleanup.sh                     ← Remove everything safely
│
├── bin/                            ← Executables
│   ├── qnn_benchmark              ← Main benchmark binary
│   ├── oom_guard.sh               ← Memory/thermal watchdog wrapper
│   └── health_monitor.sh          ← Continuous health logger
│
├── lib/                            ← QNN shared libraries
│   ├── libQnnHtp.so               ← HTP backend
│   └── libQnnHtpV73Skel.so       ← Hexagon V73 skeleton
│
├── models/                         ← Compiled model files
│   └── libqwen3_model.so          ← QNN-compiled INT4 model
│
├── tokenizer/                      ← Tokenizer files
│   └── tokenizer.model            ← SentencePiece model
│
├── logs/                           ← Runtime logs
│   └── health_timeline.csv        ← Health monitor output
│
├── profiles/                       ← Profiler CSV dumps
│   └── profile_YYYYMMDD_HHMMSS.csv
│
└── tmp/                            ← Scratch space (cleaned each run)
```

---

## What We NEVER Touch

The following exist in `/data/local/tmp/` and are **not ours**:

| File            | Owner   | Purpose       |
|-----------------|---------|---------------|
| bifrost_agent   | Unknown | Pre-existing  |

HalfHex scripts explicitly filter this file from all operations.
The `sandbox_init.sh` script logs existing files as a record.

---

## Three Independent Safety Systems

### 1. OOM Guard Wrapper (oom_guard.sh)

A shell script that wraps any binary with hardware limits:

```bash
./oom_guard.sh 3072 55 ./qnn_benchmark --model libqwen3_model.so
#              ^^^^  ^^
#              |     |
#              |     Max temperature (°C)
#              Max RSS (MB)
```

**Monitoring loop (every 2 seconds):**

| Check                | Threshold          | Action              |
|----------------------|--------------------|---------------------|
| Process RSS          | > 3072 MB          | SIGTERM → SIGKILL   |
| CPU temperature      | > 55°C             | SIGTERM → SIGKILL   |
| System MemAvailable  | < 1024 MB          | SIGTERM → SIGKILL   |
| Process alive        | Process exited     | Clean up PID file   |

**Kill sequence:** SIGTERM (graceful) → 2 second wait → SIGKILL (force).

### 2. MemoryGuard (C++ runtime)

Built into the benchmark binary itself:

- `oom_score_adj=800` set at process start
- Pre-flight `can_allocate()` before every large allocation
- Runtime `should_emergency_stop()` every 50 generated tokens
- Atomic allocation tracking by category

### 3. ThermalMonitor (C++ runtime)

- `is_safe_to_continue()` gate checked before each benchmark phase
- `wait_for_cooldown()` can pause until device reaches safe temperature
- Throttle events logged with timestamp for correlation with tok/s data

---

## Emergency Stop Conditions

The process will terminate if ANY of these three conditions are met:

| Condition                  | Source          | Exit Code | Recovery         |
|---------------------------|-----------------|-----------|------------------|
| RSS > 3072 MB             | oom_guard.sh    | 137       | Automatic        |
| CPU temp > 55°C           | oom_guard.sh    | 138       | Wait for cooldown|
| MemAvailable < 1024 MB    | oom_guard.sh    | 139       | Automatic        |
| oom_score_adj triggered    | Android LMK     | SIGKILL   | Automatic        |
| KV cache mmap fails       | KVCacheManager  | 1         | Check free RAM   |
| MemoryGuard budget denied  | MemoryGuard     | 1         | Reduce workload  |

---

## Benchmark Thermal Protocol

**Why thermal soak matters:**
Mobile SoCs boost to maximum frequency for brief periods (~10s), then
throttle down to sustainable levels. A benchmark that only runs for 5
seconds will show 2x the actual sustained performance.

**Protocol:**
1. Phase 2 (warmup): 3 short runs to warm JIT and caches
2. Phase 3 (thermal soak): 60 seconds of continuous inference
3. Phase 4 (benchmark): Measurements taken only after soak

This ensures benchmark numbers reflect real-world sustained performance.

---

## Verifying Device Is Clean

After running HalfHex, verify the device is clean:

```bash
# Option 1: Use the cleanup script
adb shell "sh /data/local/tmp/halfhex/cleanup.sh"

# Option 2: Manual verification
adb shell "ls -la /data/local/tmp/"
# Should show only: bifrost_agent (and no halfhex directory)

# Verify no HalfHex processes running
adb shell "ps -A | grep -i halfhex"
# Should return nothing

# Verify thermal is normal
adb shell "cat /sys/class/thermal/thermal_zone0/temp"
# Should be < 40000 (40°C)
```

---

## What If Something Goes Wrong

| Scenario                        | What Happens                     | How to Fix                          |
|---------------------------------|----------------------------------|-------------------------------------|
| Process uses too much RAM       | OOM guard kills it               | PID file cleaned, device unaffected |
| Device overheats                | OOM guard kills process          | Wait 5 min for cooldown             |
| System RAM critically low       | OOM guard emergency kill         | Device recovers automatically       |
| ADB disconnects during run      | Process continues, eventually killed by Android | Reconnect, run cleanup.sh  |
| Power loss during run           | mmap pages lost (no disk writes) | Sandbox stays, cleanup on next run  |
| cleanup.sh run in wrong dir     | Refuses — no .sandbox_meta found | Safe, nothing deleted               |
