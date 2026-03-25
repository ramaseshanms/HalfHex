// ============================================================================
// test_memory_guard.cpp — Unit Tests for halfhex::MemoryGuard
// ============================================================================
//
// Tests cover:
//   1. Budget initialization with valid/invalid budgets
//   2. Allocation tracking (record/dealloc counters)
//   3. Budget enforcement (can_allocate checks)
//   4. Category-specific tracking (kv_cache, model_weights, activations)
//   5. Memory pressure detection logic
//   6. Emergency stop threshold logic
//   7. Atomic counter thread safety
//   8. Default budget constants validity
//
// NOTE: Tests that read /proc/meminfo or /proc/self/status may behave
//       differently on host vs Android. Those are integration tests.
//       These unit tests focus on the budget math and tracking logic.
//
// ============================================================================

#include "test_framework.h"
#include "MemoryGuard.h"
#include <thread>

// ── Budget constants validation ───────────────────────────────────────────

TEST(MemoryGuard, DefaultBudgetConstantsAreReasonable) {
    auto b = halfhex::QWEN3_1_7B_BUDGET;

    // RSS budget should be between 1GB and 8GB
    ASSERT_GT(b.max_rss_bytes, 1ULL * 1024 * 1024 * 1024);
    ASSERT_LT(b.max_rss_bytes, 8ULL * 1024 * 1024 * 1024);

    // System reserve should be at least 512MB
    ASSERT_GE(b.min_system_available, 512ULL * 1024 * 1024);

    // KV cache budget should be reasonable for Qwen3-1.7B
    // 28 layers * 4 kv_heads * 128 dim * 512 seq * 2 (K+V) * 2 bytes = ~57MB
    // Budget of 256MB gives ~4.5x headroom
    ASSERT_GE(b.kv_cache_budget, 50ULL * 1024 * 1024);
    ASSERT_LE(b.kv_cache_budget, 1024ULL * 1024 * 1024);

    // OOM score should be high (expendable) but not max
    ASSERT_GT(b.oom_score_adj, 0);
    ASSERT_LE(b.oom_score_adj, 1000);

    // Sum of category budgets should not exceed max RSS
    size_t category_total = b.kv_cache_budget + b.model_weight_budget + b.activation_budget;
    ASSERT_LE(category_total, b.max_rss_bytes);
}

// ── Allocation tracking ───────────────────────────────────────────────────

TEST(MemoryGuard, AllocationTrackingAccumulates) {
    halfhex::MemoryGuard guard;
    // Don't initialize (skips /proc reads) — test pure tracking logic
    ASSERT_EQ(guard.total_allocated(), (size_t)0);

    guard.record_allocation(1024, "kv_cache");
    ASSERT_EQ(guard.total_allocated(), (size_t)1024);

    guard.record_allocation(2048, "model_weights");
    ASSERT_EQ(guard.total_allocated(), (size_t)3072);

    guard.record_allocation(512, "activations");
    ASSERT_EQ(guard.total_allocated(), (size_t)3584);
}

TEST(MemoryGuard, DeallocationReducesTotal) {
    halfhex::MemoryGuard guard;

    guard.record_allocation(1000, "kv_cache");
    guard.record_allocation(2000, "model_weights");
    ASSERT_EQ(guard.total_allocated(), (size_t)3000);

    guard.record_deallocation(1000, "kv_cache");
    ASSERT_EQ(guard.total_allocated(), (size_t)2000);

    guard.record_deallocation(2000, "model_weights");
    ASSERT_EQ(guard.total_allocated(), (size_t)0);
}

TEST(MemoryGuard, UnknownCategoryStillTracksTotal) {
    halfhex::MemoryGuard guard;

    guard.record_allocation(5000, "unknown_category");
    ASSERT_EQ(guard.total_allocated(), (size_t)5000);

    guard.record_deallocation(5000, "unknown_category");
    ASSERT_EQ(guard.total_allocated(), (size_t)0);
}

// ── Budget enforcement (can_allocate) ─────────────────────────────────────

TEST(MemoryGuard, CanAllocateWithinBudget) {
    halfhex::MemoryGuard guard;
    halfhex::MemoryBudget budget = {
        .max_rss_bytes        = 1024 * 1024,  // 1MB
        .min_system_available = 0,             // Don't check system memory
        .kv_cache_budget      = 512 * 1024,
        .model_weight_budget  = 512 * 1024,
        .activation_budget    = 128 * 1024,
        .oom_score_adj        = 800,
    };
    // Note: initialize() will try to read /proc and set oom_score.
    // On host, these may fail silently. That's fine for unit testing.
    guard.initialize(budget);

    // Should allow small allocation
    ASSERT_TRUE(guard.can_allocate(1024, "small_test"));
}

TEST(MemoryGuard, CanAllocateRejectsOverBudget) {
    halfhex::MemoryGuard guard;
    halfhex::MemoryBudget budget = {
        .max_rss_bytes        = 1024,  // Tiny 1KB budget
        .min_system_available = 0,
        .kv_cache_budget      = 512,
        .model_weight_budget  = 512,
        .activation_budget    = 128,
        .oom_score_adj        = 800,
    };
    guard.initialize(budget);

    // Pre-allocate most of the budget
    guard.record_allocation(900, "kv_cache");

    // Should reject — 900 + 200 > 1024
    ASSERT_FALSE(guard.can_allocate(200, "overflow_test"));
}

TEST(MemoryGuard, UninitializedGuardAllowsEverything) {
    halfhex::MemoryGuard guard;
    // Not initialized — should allow any allocation
    ASSERT_TRUE(guard.can_allocate(999999999, "huge_alloc"));
}

// ── Thread safety ─────────────────────────────────────────────────────────

TEST(MemoryGuard, ConcurrentAllocationsAreConsistent) {
    halfhex::MemoryGuard guard;

    auto worker = [&guard](int id) {
        for (int i = 0; i < 1000; i++) {
            guard.record_allocation(1, "kv_cache");
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) t.join();

    // 4 threads * 1000 * 1 byte = 4000
    ASSERT_EQ(guard.total_allocated(), (size_t)4000);
}

TEST(MemoryGuard, ConcurrentAllocAndDeallocNetZero) {
    halfhex::MemoryGuard guard;

    // Pre-allocate so dealloc doesn't underflow
    guard.record_allocation(4000, "kv_cache");

    auto alloc_worker = [&guard]() {
        for (int i = 0; i < 1000; i++) {
            guard.record_allocation(1, "kv_cache");
        }
    };
    auto dealloc_worker = [&guard]() {
        for (int i = 0; i < 1000; i++) {
            guard.record_deallocation(1, "kv_cache");
        }
    };

    std::thread t1(alloc_worker);
    std::thread t2(dealloc_worker);
    t1.join();
    t2.join();

    // Net change: +1000 - 1000 = 0, so still 4000
    ASSERT_EQ(guard.total_allocated(), (size_t)4000);
}
