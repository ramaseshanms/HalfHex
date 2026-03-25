// ============================================================================
// KVCacheManager.h — Zero-Copy Pinned KV Cache for Hexagon HTP
// ============================================================================
//
// PURPOSE:
//   Manages the Key-Value cache used by the transformer's attention layers
//   during autoregressive decoding. The KV cache stores previously computed
//   key and value projections so they don't need to be recomputed for each
//   new token.
//
// WHY THIS IS CRITICAL:
//   In autoregressive LLM inference, the KV cache is the single largest
//   data structure accessed on every decode step. For Qwen3-1.7B with
//   512 context length:
//
//     28 layers × 4 KV heads × 128 head_dim × 512 seq_len × 2 (K+V) × 2 bytes
//     = 28 × 4 × 128 × 512 × 2 × 2 = ~57 MB (fp16)
//
//   This must NEVER be:
//   - Copied during the decode loop (kills throughput)
//   - Swapped to zRAM by Android (kills latency with 10-100x penalty)
//   - Reallocated (causes fragmentation + latency spikes)
//
// MEMORY STRATEGY:
//   1. mmap(MAP_PRIVATE | MAP_ANONYMOUS) — anonymous mapping, no file backing
//   2. mlock() — pin pages in physical RAM, prevent swap
//   3. madvise(MADV_SEQUENTIAL) — hint kernel for prefetch pattern
//   4. Pre-allocate for max_seq_len at startup — never grow during inference
//   5. Pass raw pointers to QNN graph execute — zero copy
//
// NOTHING PHONE 3a PRO SPECIFICS:
//   - 12GB RAM with ~7GB available → 57MB KV cache is <1% of available
//   - mlock() requires either root or CAP_IPC_LOCK (available via adb shell)
//   - If mlock() fails, we continue with a warning (may cause latency spikes)
//   - zRAM swap is 9.4GB — without mlock, Android WILL swap idle KV pages
//
// SECURITY:
//   - Memory is zeroed on allocation (MAP_ANONYMOUS guarantees this)
//   - Memory is zeroed on release (explicit memset before munmap)
//   - No file-backed mapping — KV cache never touches disk
//
// ============================================================================

#pragma once

#include "Profiler.h"
#include "MemoryGuard.h"

#include <cstdint>
#include <cstddef>
#include <cstring>

#ifdef __ANDROID__
#include <sys/mman.h>
#endif

namespace halfhex {

// ── Qwen3-1.7B Model Dimensions ────────────────────────────────────────────
// These are architectural constants from the model config.
// Source: Qwen/Qwen3-1.7B/config.json
// ────────────────────────────────────────────────────────────────────────────
struct ModelConfig {
    int num_layers     = 28;    // transformer decoder layers
    int num_kv_heads   = 4;     // GQA: grouped query attention KV heads
    int num_q_heads    = 16;    // query heads (GQA ratio = 16/4 = 4)
    int head_dim       = 128;   // dimension per head
    int hidden_size    = 2048;  // model hidden dimension
    int vocab_size     = 151936;// vocabulary size
    int max_seq_len    = 512;   // context window for our deployment
    int intermediate_size = 8960; // FFN intermediate dimension
};

// ── KVCacheManager ──────────────────────────────────────────────────────────
class KVCacheManager {
public:
    KVCacheManager() = default;
    ~KVCacheManager();

    // Non-copyable, non-movable (owns mmap'd memory).
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    // ── Lifecycle ────────────────────────────────────────────────────────

    // Allocate the KV cache buffer for the full context window.
    // This MUST be called once at startup, before any inference.
    //
    // Returns false if:
    //   - MemoryGuard denies the allocation (insufficient system memory)
    //   - mmap() fails
    //   - Config is invalid (zero dimensions)
    //
    // What it does:
    //   1. Checks MemoryGuard budget
    //   2. mmap() anonymous pages
    //   3. mlock() to pin in RAM (warns on failure, continues)
    //   4. madvise(MADV_SEQUENTIAL) for prefetch
    //   5. Records allocation with MemoryGuard
    bool allocate(const ModelConfig& config);

    // Release all memory. Called on shutdown.
    // Zeroes memory before release (security: prevent data leakage).
    void release();

    // ── Per-Token Access ─────────────────────────────────────────────────

    // Get raw pointer to key buffer for a specific layer and sequence position.
    // Returns nullptr if not allocated or position out of bounds.
    //
    // Memory layout (contiguous per layer):
    //   [layer 0 K] [layer 0 V] [layer 1 K] [layer 1 V] ... [layer N K] [layer N V]
    //
    // Each K/V block is: num_kv_heads × head_dim × max_seq_len × sizeof(uint16_t)
    void* get_k_ptr(int layer, int seq_pos);
    void* get_v_ptr(int layer, int seq_pos);

    // Get pointer to the full K or V buffer for a layer (for QNN tensor binding).
    void* get_k_layer_ptr(int layer);
    void* get_v_layer_ptr(int layer);

    // ── Sequence Position Tracking ───────────────────────────────────────

    // Advance the current sequence position by one token.
    void advance_seq_len();

    // Reset sequence position to zero (start a new generation).
    void reset_seq_len();

    // Get current sequence length.
    int current_seq_len() const { return current_seq_len_; }

    // Check if context window is full.
    bool is_full() const { return current_seq_len_ >= config_.max_seq_len; }

    // ── Diagnostics ──────────────────────────────────────────────────────

    // Log cache utilization stats.
    void log_cache_stats() const;

    // Get total allocated bytes.
    size_t total_bytes() const { return total_bytes_; }

    // Check if the cache is allocated.
    bool is_allocated() const { return buffer_ != nullptr; }

private:
    // Compute byte offset for a given (layer, is_value, seq_pos) coordinate.
    size_t compute_offset(int layer, bool is_value, int seq_pos) const;

    // Compute the size of one K or V block for a single layer.
    size_t per_layer_kv_bytes() const;

    void*       buffer_          = nullptr;
    size_t      total_bytes_     = 0;
    ModelConfig config_          = {};
    int         current_seq_len_ = 0;
    bool        mlocked_         = false;
};

} // namespace halfhex
