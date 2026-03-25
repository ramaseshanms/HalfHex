// ============================================================================
// MemoryGuard.h — Android OOM Protection & Memory Budget Enforcement
// ============================================================================
//
// PURPOSE:
//   Prevents the QNN inference process from triggering Android's Low Memory
//   Killer (LMK), which would kill user-facing apps. This is critical on a
//   daily-driver phone like the Nothing 3a Pro where the user has messaging,
//   email, and other apps in the background.
//
// HOW IT WORKS:
//   1. At startup, queries system memory state via /proc/meminfo
//   2. Computes a safe memory budget (never exceed X% of MemAvailable)
//   3. Tracks all allocations made by the runtime
//   4. Refuses allocations that would exceed the budget
//   5. Sets oom_score_adj=800 so Android kills US first, not user apps
//
// MEMORY BUDGET CALCULATION:
//   Qwen3-1.7B memory requirements (approximate):
//   - Model weights (INT4):  ~1.0 GB
//   - KV cache (512 ctx):    ~0.2 GB (28 layers * 4 KV heads * 128 dim * 512 * fp16)
//   - Activation buffers:    ~0.1 GB
//   - Runtime overhead:      ~0.1 GB
//   - Safety margin (20%):   ~0.3 GB
//   Total budget:            ~1.7 GB (of 7GB available on this device)
//
// NOTHING PHONE 3a PRO SPECIFICS:
//   - 12GB total RAM, ~7GB available under normal use
//   - Android 15's LMK is aggressive — it watches MemAvailable closely
//   - SwapTotal = 9.4GB (zRAM) — we do NOT want our data swapped
//   - mlock() is essential for KV cache and model weights
//
// ============================================================================

#pragma once

#include "Profiler.h"
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <string>

namespace halfhex {

// ── MemoryBudget ────────────────────────────────────────────────────────────
// Configuration for memory limits. All values in bytes.
// ────────────────────────────────────────────────────────────────────────────
struct MemoryBudget {
    size_t max_rss_bytes;            // Hard RSS limit for this process
    size_t min_system_available;     // Refuse alloc if system drops below this
    size_t kv_cache_budget;          // Max bytes for KV cache
    size_t model_weight_budget;      // Max bytes for model weights
    size_t activation_budget;        // Max bytes for activation buffers
    int    oom_score_adj;            // oom_score_adj value (higher = killed first)
};

// ── Default budgets for Qwen3-1.7B on Nothing Phone 3a Pro ─────────────────
constexpr MemoryBudget QWEN3_1_7B_BUDGET = {
    .max_rss_bytes        = 3ULL * 1024 * 1024 * 1024,  // 3 GB hard limit
    .min_system_available = 1ULL * 1024 * 1024 * 1024,  // Keep 1 GB for system
    .kv_cache_budget      = 256ULL * 1024 * 1024,       // 256 MB for KV cache
    .model_weight_budget  = 1200ULL * 1024 * 1024,      // 1.2 GB for weights
    .activation_budget    = 128ULL * 1024 * 1024,        // 128 MB for activations
    .oom_score_adj        = 800,                         // Kill before user apps
};

// ── MemoryGuard ─────────────────────────────────────────────────────────────
// Singleton that tracks and enforces memory usage across the runtime.
//
// Thread safety: All methods are safe to call from any thread.
// The atomic counters use relaxed ordering (sufficient for monotonic tracking).
// ────────────────────────────────────────────────────────────────────────────
class MemoryGuard {
public:
    // Initialize with a memory budget. Call once at startup.
    // Returns false if the device doesn't have enough memory.
    bool initialize(const MemoryBudget& budget);

    // Check if an allocation of `bytes` would be safe.
    // Does NOT actually allocate — just checks the budget.
    // Returns true if safe, false if would exceed limits.
    bool can_allocate(size_t bytes, const char* label) const;

    // Record an allocation. Call after successful malloc/mmap.
    // category: "kv_cache", "model_weights", "activations", "other"
    void record_allocation(size_t bytes, const char* category);

    // Record a deallocation. Call before free/munmap.
    void record_deallocation(size_t bytes, const char* category);

    // Get current RSS of this process from /proc/self/status.
    // Returns 0 on failure.
    size_t get_current_rss_bytes() const;

    // Get system-wide MemAvailable from /proc/meminfo.
    // Returns 0 on failure.
    size_t get_system_available_bytes() const;

    // Check if we're approaching memory limits.
    // Returns true if RSS > 80% of max_rss_bytes.
    bool is_memory_pressure() const;

    // Emergency check: should we abort to protect the device?
    // Returns true if system MemAvailable < min_system_available.
    bool should_emergency_stop() const;

    // Log current memory state to logcat.
    void log_memory_state(const char* label) const;

    // Get total tracked allocations.
    size_t total_allocated() const { return total_allocated_.load(std::memory_order_relaxed); }

    // Set oom_score_adj for this process.
    // On failure, logs a warning but continues.
    static void set_oom_score(int score);

private:
    MemoryBudget budget_ = {};
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> kv_cache_allocated_{0};
    std::atomic<size_t> model_weight_allocated_{0};
    std::atomic<size_t> activation_allocated_{0};
    bool initialized_ = false;

    // Read a value from /proc by key (e.g., "VmRSS:" from /proc/self/status).
    size_t read_proc_kb(const char* path, const char* key) const;
};

// Global instance.
extern MemoryGuard g_memory_guard;

} // namespace halfhex
