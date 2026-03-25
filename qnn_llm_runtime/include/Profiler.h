// ============================================================================
// Profiler.h — Microsecond-Resolution Inference Profiler
// ============================================================================
//
// PURPOSE:
//   Provides zero-overhead (when disabled) profiling macros that capture
//   timing data for every function in the inference pipeline. When enabled,
//   every function call is timed with std::chrono::high_resolution_clock
//   (typically nanosecond resolution on ARM64) and stored in a pre-allocated
//   ring buffer to avoid malloc during hot-path execution.
//
// USAGE:
//   // At function scope — automatically times until scope exit:
//   void my_function() {
//       PROFILE_SCOPE("my_function");
//       // ... work ...
//   }  // timing recorded here
//
//   // For sub-regions within a function:
//   PROFILE_START("tensor_copy");
//   memcpy(dst, src, size);
//   PROFILE_END("tensor_copy");
//
// DESIGN DECISIONS:
//   - Pre-allocated ring buffer (default 1M entries) to avoid heap
//     allocation during inference hot path
//   - Thread-local start times to avoid mutex contention
//   - Compile-time disable via PROFILING_ENABLED=0 for release builds
//   - CSV output for offline pandas/matplotlib analysis
//   - Per-region percentile computation (p50/p95/p99) for latency analysis
//
// MEMORY FOOTPRINT:
//   Each TimingEntry = 48 bytes (name ptr + double + int64_t + padding)
//   Default ring: 1M entries = ~48 MB (acceptable for profiling builds)
//   Release build (PROFILING_ENABLED=0): zero bytes, zero instructions
//
// ============================================================================

#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cstdio>

// ── Android Logging Macros ──────────────────────────────────────────────────
// These are used throughout the entire runtime, not just the profiler.
// On non-Android builds, they fall back to fprintf(stderr, ...).
// ────────────────────────────────────────────────────────────────────────────
#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "HALFHEX"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) do { fprintf(stderr, "[INFO]  "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define LOGW(...) do { fprintf(stderr, "[WARN]  "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define LOGE(...) do { fprintf(stderr, "[ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define LOGD(...) do { fprintf(stderr, "[DEBUG] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

// ── Profiling Macros ────────────────────────────────────────────────────────
// PROFILE_SCOPE:  Drop at function start. Logs duration on scope exit.
// PROFILE_START:  Manual region start.
// PROFILE_END:    Manual region end. Must pair with PROFILE_START.
//
// When PROFILING_ENABLED is not defined or 0, all macros expand to nothing.
// The compiler eliminates them entirely — no branches, no function calls.
// ────────────────────────────────────────────────────────────────────────────
#ifdef PROFILING_ENABLED
#define HALFHEX_CONCAT_INNER(a, b) a##b
#define HALFHEX_CONCAT(a, b) HALFHEX_CONCAT_INNER(a, b)
#define PROFILE_SCOPE(name) \
    halfhex::ScopedTimer HALFHEX_CONCAT(_hh_timer_, __LINE__)(name, halfhex::g_profiler)
#define PROFILE_START(name) halfhex::g_profiler.start(name)
#define PROFILE_END(name)   halfhex::g_profiler.end(name)
#else
#define PROFILE_SCOPE(name) (void)0
#define PROFILE_START(name) (void)0
#define PROFILE_END(name)   (void)0
#endif

namespace halfhex {

// ── TimingEntry ─────────────────────────────────────────────────────────────
// A single profiling measurement. Stored in the ring buffer.
// ────────────────────────────────────────────────────────────────────────────
struct TimingEntry {
    const char* name;       // Pointer to string literal (no allocation)
    double duration_us;     // Microseconds — ms is too coarse for layer-level
    int64_t timestamp_us;   // Wall-clock timestamp for timeline analysis
};

// ── PercentileStats ─────────────────────────────────────────────────────────
// Computed statistics for a named profiling region.
// ────────────────────────────────────────────────────────────────────────────
struct PercentileStats {
    double mean_us   = 0.0;
    double median_us = 0.0;
    double p95_us    = 0.0;
    double p99_us    = 0.0;
    double min_us    = 0.0;
    double max_us    = 0.0;
    size_t count     = 0;
};

// ── Profiler ────────────────────────────────────────────────────────────────
// Thread-safe profiling accumulator.
//
// Concurrency model:
//   - start() and end() use a mutex to protect start_times_ map.
//     In production, the inference pipeline is single-threaded on the
//     HTP hot path, so contention is near-zero.
//   - timings_ vector uses reserve() at construction to avoid realloc.
// ────────────────────────────────────────────────────────────────────────────
class Profiler {
public:
    // Pre-allocate storage for expected number of timing entries.
    // Default 1M entries covers ~1000 tokens * ~30 layers * ~30 ops/layer.
    explicit Profiler(size_t reserve_entries = 1'000'000);

    // Record the start of a named region.
    void start(const char* name);

    // Record the end of a named region. Must pair with start().
    // Returns the duration in microseconds for immediate use.
    double end(const char* name);

    // Dump all statistics to logcat (Android) or stderr (host).
    // label: human-readable context for this dump (e.g., "final", "warmup").
    void dump_stats(const char* label);

    // Compute percentile statistics for all named regions.
    // Returns a map from region name to PercentileStats.
    std::unordered_map<std::string, PercentileStats> compute_percentiles() const;

    // Write all raw timing entries to a CSV file.
    // On Android: writes to /sdcard/qnn_profile_TIMESTAMP.csv
    // On host: writes to ./logs/profile_TIMESTAMP.csv
    // Returns the path written, or empty string on failure.
    std::string write_to_file(const char* directory = nullptr);

    // Clear all accumulated data. Call at start of each benchmark run.
    void reset();

    // Print per-transformer-layer breakdown (entries matching "layer_*").
    void print_layer_breakdown();

    // Get total number of recorded entries.
    size_t entry_count() const;

private:
    // Raw timing entries (append-only during profiling).
    std::vector<TimingEntry> entries_;

    // In-flight timers: name -> start time.
    std::unordered_map<std::string,
        std::chrono::time_point<std::chrono::high_resolution_clock>> start_times_;

    // Per-region accumulated durations for statistics.
    std::unordered_map<std::string, std::vector<double>> timings_;

    // Protects start_times_ and timings_ maps.
    mutable std::mutex mutex_;
};

// ── ScopedTimer ─────────────────────────────────────────────────────────────
// RAII timer. Created by PROFILE_SCOPE macro. Records duration on destruction.
// ────────────────────────────────────────────────────────────────────────────
struct ScopedTimer {
    ScopedTimer(const char* name, Profiler& p) : name_(name), p_(p) {
        p_.start(name_);
    }
    ~ScopedTimer() {
        p_.end(name_);
    }

    // Non-copyable, non-movable (always stack-allocated by macro).
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    const char* name_;
    Profiler& p_;
};

// Global profiler instance. Accessible from any translation unit.
extern Profiler g_profiler;

} // namespace halfhex
