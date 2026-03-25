# Security Properties — HalfHex Runtime

This document describes the security properties of the HalfHex codebase
and the threat model it operates under.

---

## Threat Model

HalfHex runs on a **daily-driver phone** — the user's primary device with
personal data, messaging apps, banking apps, and photos. The threat is not
external attackers, but **the inference workload itself**:

- A runaway process could exhaust RAM and crash the phone
- A thermal runaway could damage the battery or SoC
- Model data could leak into freed memory pages
- A poorly written cleanup could delete user files
- Unchecked allocations could trigger Android's OOM killer on user apps

Every security measure in this codebase is designed to prevent these failures.

---

## Memory Safety

### 1. KV Cache Zeroing Before Release

The KV cache contains transformer activations derived from user prompts.
These could theoretically reveal prompt content if another process reads
the same physical pages after deallocation.

**Mitigation:** `memset(buffer_, 0, total_bytes_)` is called before
`munmap()` in `KVCacheManager::release()`. This zeroes every byte of the
KV cache before the pages are returned to the kernel.

### 2. No File-Backed Mappings

All memory allocations use `mmap(MAP_PRIVATE | MAP_ANONYMOUS)`:
- `MAP_ANONYMOUS`: No file descriptor, no disk backing
- `MAP_PRIVATE`: Changes are not visible to other processes
- Model data never touches the filesystem (loaded via `dlopen` from .so)

This means inference data exists only in RAM and is never written to flash.

### 3. Bounds Checking on KV Cache Access

Every `get_k_ptr()` and `get_v_ptr()` call validates:
```cpp
if (!buffer_ || layer < 0 || layer >= config_.num_layers ||
    seq_pos < 0 || seq_pos >= config_.max_seq_len) {
    return nullptr;
}
```

Out-of-bounds access returns `nullptr` instead of a corrupted pointer.

### 4. No Unchecked String Formatting

The codebase uses `snprintf()` exclusively. There are zero uses of
`sprintf()`, `gets()`, or unbounded string operations.

JNI string handling always releases via `ReleaseStringUTFChars()` in every
code path, including error returns.

---

## OOM Protection

### Process Expendability

At startup, MemoryGuard writes `800` to `/proc/self/oom_score_adj`.
Android's Low Memory Killer uses this score to decide kill order:

| Score  | Process Type              | Kill Priority |
|--------|---------------------------|---------------|
| -1000  | System critical           | Never         |
| -900   | Persistent services       | Last resort   |
| 0      | Foreground app            | Only in crisis|
| 200    | Visible app               | Under pressure|
| **800**| **HalfHex inference**     | **Early kill** |
| 906    | Cached empty process      | First to die  |

Our process is killed before any visible user app.

### Budget Enforcement

Three independent checks:
1. **Pre-allocation:** `can_allocate()` rejects if RSS would exceed 3 GB
2. **System-wide:** Refuses if MemAvailable would drop below 1 GB
3. **Runtime:** `should_emergency_stop()` checked every 50 tokens

---

## Thermal Protection

The ThermalMonitor reads hardware sensors and enforces:
- **55°C:** Process termination (protects battery and SoC)
- **Frequency ratio < 85%:** Logged as throttle event

Thermal checks happen:
- Before each benchmark phase
- During sustained throughput tests
- Via the external `oom_guard.sh` wrapper (independent monitoring)

---

## Process Isolation

### Filesystem Confinement

Every file created by HalfHex lives under `/data/local/tmp/halfhex/`.
The codebase never:
- Writes to /system, /vendor, or /product
- Installs APKs or modifies app data directories
- Creates files in /sdcard (except profiler CSV, which is optional)
- Modifies /etc or any system configuration

### Cleanup Verification

The cleanup script (`cleanup.sh`) requires `.sandbox_meta` to exist
before deleting anything:

```sh
if [ -f "${SANDBOX_ROOT}/.sandbox_meta" ]; then
    rm -rf "${SANDBOX_ROOT}"
else
    echo "Refusing to remove — this may not be a HalfHex sandbox"
    exit 1
fi
```

This prevents accidental deletion if the script is run from the wrong directory.

### No Persistent System Changes

- SELinux policy is never modified persistently
- CPU governor is only changed for benchmark duration (not by default)
- No kernel modules loaded
- No sysctl modifications

---

## Input Validation

### JNI Bridge

- JNI string pointers checked for null after `GetStringUTFChars()`
- All `new` allocations use `std::nothrow` (JNI must never throw)
- Runtime state checked before every operation (`if (!g_runtime)`)

### QNN API

- Every QNN API call return code is checked via `LOG_QNN_STATUS` macro
- Failed `dlopen()` and `dlsym()` report `dlerror()` details
- Error counter incremented on any graph execute failure

### Model Configuration

- `KVCacheManager::allocate()` validates all dimensions are positive
- Context window overflow is detected and logged
