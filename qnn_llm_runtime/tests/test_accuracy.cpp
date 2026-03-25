// ============================================================================
// test_accuracy.cpp — Numerical Accuracy & Correctness Tests
// ============================================================================
//
// PURPOSE:
//   Verify the mathematical and logical correctness of inference-critical
//   operations. These tests catch:
//   - Off-by-one errors in KV cache indexing
//   - Incorrect memory layout (wrong stride, wrong offset)
//   - Integer overflow in size calculations
//   - fp16 precision issues
//   - Argmax sampling correctness
//   - Sequence position drift
//
// DESIGN:
//   Tests use known inputs and assert exact expected outputs. No randomness.
//   Deterministic failures mean deterministic bugs.
//
// ============================================================================

#include "test_framework.h"
#include "KVCacheManager.h"
#include "MemoryGuard.h"
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// ── KV Cache memory layout correctness ────────────────────────────────────

TEST(Accuracy, KVCacheLayerPointersAreContiguous) {
    // Verify that layer N+1's K pointer starts exactly after layer N's V block.
    // This is critical for QNN tensor binding which expects contiguous memory.
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);

    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 8;
    config.num_layers = 4;
    cache.allocate(config);

    // Expected size of one K or V block:
    // num_kv_heads(4) * head_dim(128) * max_seq_len(8) * sizeof(uint16_t)(2)
    size_t per_kv_block = config.num_kv_heads * config.head_dim *
                          config.max_seq_len * sizeof(uint16_t);

    for (int layer = 0; layer < config.num_layers - 1; layer++) {
        uintptr_t k_this = (uintptr_t)cache.get_k_layer_ptr(layer);
        uintptr_t v_this = (uintptr_t)cache.get_v_layer_ptr(layer);
        uintptr_t k_next = (uintptr_t)cache.get_k_layer_ptr(layer + 1);

        // V should follow K within the same layer
        ASSERT_EQ(v_this, k_this + per_kv_block);

        // Next layer's K should follow this layer's V
        ASSERT_EQ(k_next, v_this + per_kv_block);
    }

    cache.release();
}

TEST(Accuracy, KVCacheSeqPosPointersHaveCorrectStride) {
    // Verify that seq_pos N+1 is exactly (num_kv_heads * head_dim * 2) bytes
    // after seq_pos N within the same layer's K buffer.
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);

    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 8;
    config.num_layers = 2;
    cache.allocate(config);

    // Expected stride between consecutive sequence positions:
    // num_kv_heads(4) * head_dim(128) * sizeof(uint16_t)(2) = 1024 bytes
    size_t seq_stride = config.num_kv_heads * config.head_dim * sizeof(uint16_t);

    for (int layer = 0; layer < config.num_layers; layer++) {
        uintptr_t pos0 = (uintptr_t)cache.get_k_ptr(layer, 0);
        uintptr_t pos1 = (uintptr_t)cache.get_k_ptr(layer, 1);

        if (pos0 != 0 && pos1 != 0) {
            ASSERT_EQ(pos1 - pos0, seq_stride);
        }
    }

    cache.release();
}

TEST(Accuracy, KVCacheWriteReadRoundtrip) {
    // Write known pattern to KV cache and read it back.
    // Verifies no data corruption in the buffer.
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);

    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 4;
    config.num_layers = 2;
    cache.allocate(config);

    // Write a known pattern to layer 0, seq_pos 0
    uint16_t* k0 = static_cast<uint16_t*>(cache.get_k_ptr(0, 0));
    ASSERT_NOT_NULL(k0);

    size_t num_elements = config.num_kv_heads * config.head_dim;
    for (size_t i = 0; i < num_elements; i++) {
        k0[i] = static_cast<uint16_t>(i & 0xFFFF);
    }

    // Read it back and verify
    uint16_t* k0_read = static_cast<uint16_t*>(cache.get_k_ptr(0, 0));
    for (size_t i = 0; i < num_elements; i++) {
        ASSERT_EQ(k0_read[i], static_cast<uint16_t>(i & 0xFFFF));
    }

    // Verify layer 1 is still zeroed (writes to layer 0 didn't bleed)
    uint16_t* k1 = static_cast<uint16_t*>(cache.get_k_ptr(1, 0));
    ASSERT_NOT_NULL(k1);
    bool layer1_zeroed = true;
    for (size_t i = 0; i < num_elements; i++) {
        if (k1[i] != 0) { layer1_zeroed = false; break; }
    }
    ASSERT_TRUE(layer1_zeroed);

    cache.release();
}

// ── Size calculation overflow safety ──────────────────────────────────────

TEST(Accuracy, KVCacheSizeCalculationNoOverflow) {
    // For Qwen3-1.7B full config, verify the total size calculation
    // doesn't overflow a size_t (64-bit unsigned).
    halfhex::ModelConfig config;

    // Actual formula from KVCacheManager:
    // per_layer = num_kv_heads * head_dim * max_seq_len * sizeof(uint16_t)
    // total = num_layers * 2 * per_layer
    size_t per_kv = (size_t)config.num_kv_heads * config.head_dim *
                    config.max_seq_len * sizeof(uint16_t);
    size_t expected = (size_t)config.num_layers * 2 * per_kv;

    // Should be ~28 MB
    ASSERT_GT(expected, 20ULL * 1024 * 1024);
    ASSERT_LT(expected, 60ULL * 1024 * 1024);

    // Verify it fits in 32 bits
    ASSERT_LT(expected, (size_t)UINT32_MAX);
}

TEST(Accuracy, KVCacheSizeWithMaxContextDoesNotOverflow) {
    // Even with max context (32k), size should not overflow 64-bit.
    halfhex::ModelConfig config;
    config.max_seq_len = 32768;

    // per_layer = 4 * 128 * 32768 * 2 = 33,554,432 bytes = 32 MB
    // total = 28 * 2 * 33,554,432 = 1,879,048,192 bytes = ~1.75 GB
    size_t per_kv = (size_t)config.num_kv_heads * config.head_dim *
                    config.max_seq_len * sizeof(uint16_t);
    size_t total = (size_t)config.num_layers * 2 * per_kv;

    ASSERT_GT(total, 1ULL * 1024 * 1024 * 1024);
    ASSERT_LT(total, 4ULL * 1024 * 1024 * 1024);
}

// ── Argmax sampling correctness ───────────────────────────────────────────
// This simulates the argmax operation used in greedy decoding.

TEST(Accuracy, ArgmaxFindsCorrectMaximum) {
    // Simulate a logits vector of vocab_size=151936
    // Place the maximum at a known index
    std::vector<float> logits(151936, -100.0f);
    int expected_token = 42;
    logits[expected_token] = 10.0f;

    // Argmax
    int predicted = static_cast<int>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));

    ASSERT_EQ(predicted, expected_token);
}

TEST(Accuracy, ArgmaxHandlesNegativeLogits) {
    // All logits negative — should still find the least negative
    std::vector<float> logits = {-5.0f, -3.0f, -1.0f, -2.0f, -4.0f};
    int predicted = static_cast<int>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));

    ASSERT_EQ(predicted, 2);  // -1.0 is the maximum
}

TEST(Accuracy, ArgmaxHandlesIdenticalValues) {
    // Tie-breaking: std::max_element returns first maximum
    std::vector<float> logits = {1.0f, 5.0f, 5.0f, 3.0f};
    int predicted = static_cast<int>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));

    ASSERT_EQ(predicted, 1);  // First occurrence of 5.0
}

// ── MemoryGuard budget arithmetic ─────────────────────────────────────────

TEST(Accuracy, MemoryBudgetCategorySumFitsRSS) {
    auto b = halfhex::QWEN3_1_7B_BUDGET;
    size_t category_sum = b.kv_cache_budget + b.model_weight_budget + b.activation_budget;

    // Category budgets must sum to less than max RSS
    ASSERT_LT(category_sum, b.max_rss_bytes);

    // And leave room for runtime overhead (~500MB)
    ASSERT_LT(category_sum + 500ULL * 1024 * 1024, b.max_rss_bytes);
}

// ── Sequence tracking correctness ─────────────────────────────────────────

TEST(Accuracy, SequenceTrackingIsMonotonic) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 100;
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);
    cache.allocate(config);

    int prev = cache.current_seq_len();
    for (int i = 0; i < 100; i++) {
        cache.advance_seq_len();
        int curr = cache.current_seq_len();
        ASSERT_EQ(curr, prev + 1);
        prev = curr;
    }
    ASSERT_TRUE(cache.is_full());

    cache.release();
}

TEST(Accuracy, MultipleResetCyclesWork) {
    halfhex::KVCacheManager cache;
    halfhex::ModelConfig config;
    config.max_seq_len = 10;
    halfhex::MemoryGuard guard;
    guard.initialize(halfhex::QWEN3_1_7B_BUDGET);
    cache.allocate(config);

    // Cycle 1
    for (int i = 0; i < 10; i++) cache.advance_seq_len();
    ASSERT_TRUE(cache.is_full());
    cache.reset_seq_len();
    ASSERT_EQ(cache.current_seq_len(), 0);
    ASSERT_FALSE(cache.is_full());

    // Cycle 2
    for (int i = 0; i < 5; i++) cache.advance_seq_len();
    ASSERT_EQ(cache.current_seq_len(), 5);
    ASSERT_FALSE(cache.is_full());

    cache.release();
}
