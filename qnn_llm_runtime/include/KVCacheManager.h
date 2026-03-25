// KVCacheManager.h
// CRITICAL: KV cache must NEVER be copied during decode loop
// Allocate once, reuse across all tokens via pointer passing
#pragma once

#include "Profiler.h"
#include <cstdint>
#include <cstddef>
#include <sys/mman.h>

class KVCacheManager {
public:
    KVCacheManager() = default;
    ~KVCacheManager();

    // Pre-allocate for max_seq_len at startup
    // Never reallocate during inference — this kills performance
    bool allocate(int num_layers, int num_kv_heads, int head_dim, int max_seq_len) {
        PROFILE_SCOPE("kv_cache_allocate");

        num_layers_   = num_layers;
        num_kv_heads_ = num_kv_heads;
        head_dim_     = head_dim;
        max_seq_len_  = max_seq_len;

        size_t bytes_per_layer = (size_t)num_kv_heads * head_dim * max_seq_len * sizeof(uint16_t);
        total_bytes_ = num_layers * 2 * bytes_per_layer;  // K and V

        LOGI("[PROFILE][KV] Allocating %.1f MB for KV cache", total_bytes_ / 1e6);
        LOGI("[PROFILE][KV] Config: layers=%d, kv_heads=%d, head_dim=%d, max_len=%d",
             num_layers, num_kv_heads, head_dim, max_seq_len);

        // Use mmap with MADV_HUGEPAGE for better TLB performance
        kv_buffer_ = mmap(nullptr, total_bytes_,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (kv_buffer_ == MAP_FAILED) {
            LOGE("[PROFILE][KV] mmap failed!");
            kv_buffer_ = nullptr;
            return false;
        }

        // Lock pages in RAM — prevent Android from swapping KV cache
        if (mlock(kv_buffer_, total_bytes_) != 0) {
            LOGW("[PROFILE][KV] mlock failed — KV cache may be paged out under pressure");
        }

        // Advise kernel: we access sequentially per token
        madvise(kv_buffer_, total_bytes_, MADV_SEQUENTIAL);

        current_seq_len_ = 0;

        LOGI("[PROFILE][KV] Allocation complete. %.1f MB locked in RAM", total_bytes_ / 1e6);
        return true;
    }

    // Zero-copy pointer into pre-allocated buffer for layer N
    void* get_k_ptr(int layer, int seq_pos) {
        size_t bytes_per_layer = (size_t)num_kv_heads_ * head_dim_ * max_seq_len_ * sizeof(uint16_t);
        size_t offset = (size_t)layer * 2 * bytes_per_layer +
                        (size_t)seq_pos * num_kv_heads_ * head_dim_ * sizeof(uint16_t);
        return static_cast<uint8_t*>(kv_buffer_) + offset;
    }

    void* get_v_ptr(int layer, int seq_pos) {
        size_t bytes_per_layer = (size_t)num_kv_heads_ * head_dim_ * max_seq_len_ * sizeof(uint16_t);
        size_t offset = (size_t)layer * 2 * bytes_per_layer + bytes_per_layer +
                        (size_t)seq_pos * num_kv_heads_ * head_dim_ * sizeof(uint16_t);
        return static_cast<uint8_t*>(kv_buffer_) + offset;
    }

    void log_cache_stats() {
        LOGI("[PROFILE][KV] Current seq len: %d / %d (%.1f%% full)",
             current_seq_len_, max_seq_len_,
             100.0f * current_seq_len_ / max_seq_len_);
    }

    void advance_seq_len() { current_seq_len_++; }
    int get_current_seq_len() const { return current_seq_len_; }
    void reset_seq_len() { current_seq_len_ = 0; }

    void release() {
        if (kv_buffer_ && kv_buffer_ != MAP_FAILED) {
            munlock(kv_buffer_, total_bytes_);
            munmap(kv_buffer_, total_bytes_);
            kv_buffer_ = nullptr;
        }
    }

private:
    void*  kv_buffer_     = nullptr;
    size_t total_bytes_   = 0;
    int    num_layers_    = 0;
    int    num_kv_heads_  = 0;
    int    head_dim_      = 0;
    int    max_seq_len_   = 0;
    int    current_seq_len_ = 0;
};
