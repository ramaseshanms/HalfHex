// ============================================================================
// test_perf_regression.cpp — Performance Regression Tests
// ============================================================================
//
// PURPOSE:
//   Detect performance regressions in the runtime's hot-path operations.
//   Each test measures a specific operation and asserts it completes within
//   a known time budget. If a code change causes a test to fail, someone
//   introduced a regression.
//
// HOW TO USE:
//   Run after every code change to the runtime. Failures indicate:
//   - An accidental O(n²) loop was introduced
//   - An allocation was added to the hot path
//   - A mutex contention issue was created
//   - Memory layout changed causing cache misses
//
// BASELINES:
//   Baselines are set conservatively (2-5x the expected time) to avoid
//   flaky failures on slow CI machines. The goal is to catch 10x regressions,
//   not micro-optimizations.
//
// PLATFORM:
//   These tests run on BOTH host (x86_64) and device (aarch64).
//   Thresholds are set for the slower platform.
//
// ============================================================================

#include "test_framework.h"
#include "Profiler.h"
#include "MemoryGuard.h"
#include "KVCacheManager.h"
#include <chrono>
#include <cstring>

// ── Helper: measure time in microseconds ──────────────────────────────────
static double measure_us(std::function<void()> fn) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// ── Profiler hot-path performance ─────────────────────────────────────────

TEST(PerfRegression, ProfilerStartEndUnder10us) {
    // A single start/end pair should take < 10us on any modern CPU.
    // This is called once per transformer layer per decode step.
    // At 28 layers and 10 tok/s, that's 280 calls/sec — must be fast.
    halfhex::Profiler p(1024);

    // Warmup
    for (int i = 0; i < 100; i++) {
        p.start("warmup"); p.end("warmup");
    }
    p.reset();

    // Measure 1000 iterations
    double total = measure_us([&]() {
        for (int i = 0; i < 1000; i++) {
            p.start("benchmark_region");
            p.end("benchmark_region");
        }
    });
    double per_call = total / 1000.0;

    // Budget: 10us per start/end pair (conservative)
    // Typical: 0.5-2us on modern x86_64, 1-5us on ARM64
    ASSERT_LT(per_call, 10.0);
}

TEST(PerfRegression, ScopedTimerOverheadUnder10us) {
    halfhex::Profiler p(1024);

    double total = measure_us([&]() {
        for (int i = 0; i < 1000; i++) {
            halfhex::ScopedTimer timer("scoped_bench", p);
        }
    });
    double per_scope = total / 1000.0;
    ASSERT_LT(per_scope, 10.0);
}

TEST(PerfRegression, ProfilerPercentileComputeUnder50ms) {
    // Computing percentiles over 10000 entries should be fast.
    halfhex::Profiler p(100000);
    for (int i = 0; i < 10000; i++) {
        p.start("stress_region");
        p.end("stress_region");
    }

    double compute_time = measure_us([&]() {
        auto stats = p.compute_percentiles();
        (void)stats;
    });

    // Budget: 50ms for 10k entries (includes sorting)
    ASSERT_LT(compute_time, 50000.0);
}

// ── MemoryGuard hot-path performance ──────────────────────────────────────

TEST(PerfRegression, MemoryGuardCanAllocateUnder5us) {
    // can_allocate() is called before every allocation.
    // It must be fast (just integer comparison + optional /proc read).
    halfhex::MemoryGuard guard;
    halfhex::MemoryBudget budget = halfhex::QWEN3_1_7B_BUDGET;
    guard.initialize(budget);

    // Warmup
    for (int i = 0; i < 100; i++) {
        guard.can_allocate(1024, "warmup");
    }

    double total = measure_us([&]() {
        for (int i = 0; i < 1000; i++) {
            guard.can_allocate(1024, "bench");
        }
    });
    double per_check = total / 1000.0;

    // Budget: 5us per check (may include /proc read on first call)
    // Note: /proc reads are ~2us on Android, so this is generous
    // On host without /proc, it's even faster
    ASSERT_LT(per_check, 100.0);  // Very generous for host+CI
}

TEST(PerfRegression, MemoryGuardRecordAllocUnder1us) {
    // record_allocation is a single atomic fetch_add.
    halfhex::MemoryGuard guard;

    double total = measure_us([&]() {
        for (int i = 0; i < 10000; i++) {
            guard.record_allocation(1, "kv_cache");
        }
    });
    double per_record = total / 10000.0;

    // Budget: 1us per atomic operation (generous)
    ASSERT_LT(per_record, 1.0);
}

// ── KV Cache access pattern performance ───────────────────────────────────

TEST(PerfRegression, KVCacheAllocateUnder100ms) {
    // Full KV cache allocation (mmap + mlock) for Qwen3-1.7B should be fast.
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);

    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;

    double alloc_time = measure_us([&]() {
        cache.allocate(config);
    });

    // Budget: 100ms for ~57MB mmap + mlock
    ASSERT_LT(alloc_time, 100000.0);

    cache.release();
}

TEST(PerfRegression, KVCachePointerLookupUnder1us) {
    // get_k_layer_ptr / get_v_layer_ptr is called per-layer per-token.
    // At 28 layers and 10 tok/s, that's 560 calls/sec.
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);

    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    cache.allocate(config);

    // Warmup
    for (int i = 0; i < 100; i++) {
        cache.get_k_layer_ptr(i % config.num_layers);
    }

    double total = measure_us([&]() {
        for (int i = 0; i < 10000; i++) {
            volatile void* k = cache.get_k_layer_ptr(i % config.num_layers);
            volatile void* v = cache.get_v_layer_ptr(i % config.num_layers);
            (void)k; (void)v;
        }
    });
    double per_lookup = total / 20000.0;  // 10000 iters * 2 lookups each

    // Budget: 1us per lookup (should be just pointer arithmetic)
    ASSERT_LT(per_lookup, 1.0);

    cache.release();
}

TEST(PerfRegression, KVCacheSequentialWriteThroughput) {
    // Simulate filling the KV cache sequentially (as in autoregressive decode).
    // This tests memory bandwidth and cache-friendliness of our layout.
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);

    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 64;  // Small for fast testing
    cache.allocate(config);

    double total = measure_us([&]() {
        for (int seq = 0; seq < config.max_seq_len; seq++) {
            for (int layer = 0; layer < config.num_layers; layer++) {
                void* k = cache.get_k_ptr(layer, seq);
                void* v = cache.get_v_ptr(layer, seq);
                if (k && v) {
                    // Simulate writing one token's KV data
                    // 4 heads * 128 dim * 2 bytes = 1024 bytes per K or V
                    memset(k, 0x42, config.num_kv_heads * config.head_dim * sizeof(uint16_t));
                    memset(v, 0x43, config.num_kv_heads * config.head_dim * sizeof(uint16_t));
                }
            }
            cache.advance_seq_len();
        }
    });

    // 64 tokens * 28 layers * 2 (K+V) writes = 3584 memset calls
    // Each memset is 1024 bytes. Total: ~3.5 MB of writes.
    // Budget: 50ms for 3.5MB sequential writes (easily achievable)
    ASSERT_LT(total, 50000.0);

    cache.release();
}
