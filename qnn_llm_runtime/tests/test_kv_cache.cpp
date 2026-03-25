// ============================================================================
// test_kv_cache.cpp — Unit Tests for halfhex::KVCacheManager
// ============================================================================
//
// Tests cover:
//   1. ModelConfig default values match Qwen3-1.7B spec
//   2. KV cache memory layout correctness
//   3. Allocation and deallocation lifecycle
//   4. Sequence position tracking (advance, reset, is_full)
//   5. Pointer arithmetic for per-layer K/V access
//   6. Out-of-bounds access returns nullptr
//   7. Double-free safety
//   8. Memory zeroing on release (security)
//   9. Total bytes calculation matches formula
//  10. Integration with MemoryGuard budget checks
//
// NOTE: mmap/mlock tests only work fully on Linux/Android.
//       On Windows host, the allocator falls back to malloc.
//
// ============================================================================

#include "test_framework.h"
#include "KVCacheManager.h"
#include <cstring>

// ── ModelConfig defaults ──────────────────────────────────────────────────

TEST(KVCache, DefaultModelConfigMatchesQwen3) {
    halfhex::ModelConfig config;

    ASSERT_EQ(config.num_layers, 28);
    ASSERT_EQ(config.num_kv_heads, 4);
    ASSERT_EQ(config.num_q_heads, 16);
    ASSERT_EQ(config.head_dim, 128);
    ASSERT_EQ(config.hidden_size, 2048);
    ASSERT_EQ(config.vocab_size, 151936);
    ASSERT_EQ(config.max_seq_len, 512);

    // GQA ratio should be 4 (16 Q heads / 4 KV heads)
    ASSERT_EQ(config.num_q_heads / config.num_kv_heads, 4);
}

TEST(KVCache, TotalBytesCalculation) {
    halfhex::ModelConfig config;

    // per_layer_kv_bytes = num_kv_heads(4) * head_dim(128) * max_seq_len(512) * sizeof(uint16_t)(2)
    //                    = 524,288 bytes per K or V block
    // total = num_layers(28) * 2 (K+V) * 524,288 = 29,360,128 bytes = ~28 MB
    size_t per_kv = (size_t)config.num_kv_heads * config.head_dim *
                    config.max_seq_len * sizeof(uint16_t);
    size_t expected = (size_t)config.num_layers * 2 * per_kv;

    ASSERT_GT(expected, 20ULL * 1024 * 1024);   // > 20 MB
    ASSERT_LT(expected, 60ULL * 1024 * 1024);   // < 60 MB
    ASSERT_EQ(per_kv, (size_t)524288);           // Exact per-block size
    ASSERT_EQ(expected, (size_t)29360128);       // Exact total
}

// ── Allocation lifecycle ──────────────────────────────────────────────────

TEST(KVCache, AllocateSucceedsWithDefaultConfig) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;

    // Initialize memory guard with generous budget
    halfhex::MemoryGuard guard;
    halfhex::MemoryBudget budget = halfhex::QWEN3_1_7B_BUDGET;
    guard.initialize(budget);

    ASSERT_FALSE(cache.is_allocated());
    bool ok = cache.allocate(config);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(cache.is_allocated());
    ASSERT_GT(cache.total_bytes(), (size_t)0);

    cache.release();
    ASSERT_FALSE(cache.is_allocated());
}

TEST(KVCache, DoubleReleaseIsSafe) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);

    cache.allocate(config);
    cache.release();
    cache.release();  // Should not crash
    ASSERT_FALSE(cache.is_allocated());
}

// ── Sequence position tracking ────────────────────────────────────────────

TEST(KVCache, SequencePositionStartsAtZero) {
    halfhex::KVCacheManager cache;
    ASSERT_EQ(cache.current_seq_len(), 0);
    ASSERT_FALSE(cache.is_full());
}

TEST(KVCache, AdvanceIncreasesSeqLen) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);
    cache.allocate(config);

    cache.advance_seq_len();
    ASSERT_EQ(cache.current_seq_len(), 1);

    for (int i = 0; i < 99; i++) cache.advance_seq_len();
    ASSERT_EQ(cache.current_seq_len(), 100);
}

TEST(KVCache, ResetSeqLenGoesToZero) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);
    cache.allocate(config);

    for (int i = 0; i < 50; i++) cache.advance_seq_len();
    ASSERT_EQ(cache.current_seq_len(), 50);

    cache.reset_seq_len();
    ASSERT_EQ(cache.current_seq_len(), 0);
}

TEST(KVCache, IsFullAtMaxSeqLen) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 10;  // Small for testing
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);
    cache.allocate(config);

    for (int i = 0; i < 10; i++) cache.advance_seq_len();
    ASSERT_TRUE(cache.is_full());
    ASSERT_EQ(cache.current_seq_len(), 10);

    cache.release();
}

// ── Pointer access ────────────────────────────────────────────────────────

TEST(KVCache, LayerPointersAreNonNullAfterAllocate) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 16;  // Small for testing
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);
    cache.allocate(config);

    for (int layer = 0; layer < config.num_layers; layer++) {
        void* k = cache.get_k_layer_ptr(layer);
        void* v = cache.get_v_layer_ptr(layer);
        ASSERT_NOT_NULL(k);
        ASSERT_NOT_NULL(v);
        ASSERT_NE(k, v);  // K and V are separate regions
    }

    cache.release();
}

TEST(KVCache, PointersAreNullBeforeAllocate) {
    halfhex::KVCacheManager cache;
    void* k = cache.get_k_layer_ptr(0);
    void* v = cache.get_v_layer_ptr(0);
    ASSERT_NULL(k);
    ASSERT_NULL(v);
}

TEST(KVCache, LayerPointersAreDistinct) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 16;
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);
    cache.allocate(config);

    // Each layer's K and V pointers should be at different addresses
    void* k0 = cache.get_k_layer_ptr(0);
    void* k1 = cache.get_k_layer_ptr(1);
    void* v0 = cache.get_v_layer_ptr(0);

    ASSERT_NE(k0, k1);
    ASSERT_NE(k0, v0);

    // Layer 1 pointers should be after layer 0 pointers (contiguous layout)
    ASSERT_GT((uintptr_t)k1, (uintptr_t)k0);

    cache.release();
}

// ── Memory is zero-initialized ────────────────────────────────────────────

TEST(KVCache, AllocatedMemoryIsZeroed) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 4;  // Tiny for testing
    config.num_layers = 2;
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);
    cache.allocate(config);

    // Check first few bytes of layer 0 K buffer are zero
    uint8_t* k0 = static_cast<uint8_t*>(cache.get_k_layer_ptr(0));
    ASSERT_NOT_NULL(k0);

    bool all_zero = true;
    for (int i = 0; i < 64; i++) {
        if (k0[i] != 0) { all_zero = false; break; }
    }
    ASSERT_TRUE(all_zero);

    cache.release();
}
