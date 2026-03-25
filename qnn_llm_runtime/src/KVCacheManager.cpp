// ============================================================================
// KVCacheManager.cpp — Zero-Copy Pinned KV Cache Implementation
// ============================================================================
//
// See KVCacheManager.h for design rationale and memory layout.
//
// ============================================================================

#include "KVCacheManager.h"

#ifdef __ANDROID__
#include <sys/mman.h>
#else
// Stubs for host-side compilation (testing only).
#include <cstdlib>
#define MAP_FAILED ((void*)-1)
#define PROT_READ  0x1
#define PROT_WRITE 0x2
#define MAP_PRIVATE  0x02
#define MAP_ANONYMOUS 0x20
#define MADV_SEQUENTIAL 2
inline void* mmap(void*, size_t sz, int, int, int, int) { return calloc(1, sz); }
inline int munmap(void* p, size_t) { free(p); return 0; }
inline int mlock(void*, size_t) { return 0; }
inline int munlock(void*, size_t) { return 0; }
inline int madvise(void*, size_t, int) { return 0; }
#endif

namespace halfhex {

// ── Destructor ──────────────────────────────────────────────────────────────
KVCacheManager::~KVCacheManager() {
    release();
}

// ── allocate ────────────────────────────────────────────────────────────────
bool KVCacheManager::allocate(const ModelConfig& config) {
    PROFILE_SCOPE("kv_cache_allocate");

    // Validate config.
    if (config.num_layers <= 0 || config.num_kv_heads <= 0 ||
        config.head_dim <= 0 || config.max_seq_len <= 0) {
        LOGE("[KV] Invalid config: layers=%d, kv_heads=%d, head_dim=%d, max_len=%d",
             config.num_layers, config.num_kv_heads, config.head_dim, config.max_seq_len);
        return false;
    }

    // Release any existing allocation.
    if (buffer_) release();

    config_ = config;

    // Compute total memory requirement.
    // Layout: [L0_K][L0_V][L1_K][L1_V]...[LN_K][LN_V]
    // Each K/V = num_kv_heads × head_dim × max_seq_len × sizeof(uint16_t)
    size_t kv_block = per_layer_kv_bytes();
    total_bytes_ = (size_t)config_.num_layers * 2 * kv_block;

    LOGI("[KV] Configuration:");
    LOGI("[KV]   Layers:     %d", config_.num_layers);
    LOGI("[KV]   KV heads:   %d", config_.num_kv_heads);
    LOGI("[KV]   Head dim:   %d", config_.head_dim);
    LOGI("[KV]   Max seq:    %d", config_.max_seq_len);
    LOGI("[KV]   Per-layer:  %.2f MB (K) + %.2f MB (V)",
         kv_block / 1e6, kv_block / 1e6);
    LOGI("[KV]   Total:      %.2f MB", total_bytes_ / 1e6);

    // Check memory budget BEFORE allocating.
    if (!g_memory_guard.can_allocate(total_bytes_, "kv_cache")) {
        LOGE("[KV] MemoryGuard denied KV cache allocation of %.2f MB", total_bytes_ / 1e6);
        return false;
    }

    // Allocate via mmap (anonymous, private).
    // MAP_ANONYMOUS guarantees zero-initialized pages.
    buffer_ = mmap(nullptr, total_bytes_,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (buffer_ == MAP_FAILED) {
        LOGE("[KV] mmap failed for %zu bytes", total_bytes_);
        buffer_ = nullptr;
        total_bytes_ = 0;
        return false;
    }

    // Pin pages in physical RAM — prevent Android from swapping to zRAM.
    if (mlock(buffer_, total_bytes_) == 0) {
        mlocked_ = true;
        LOGI("[KV] mlock succeeded — %zu MB pinned in RAM", total_bytes_ / (1024 * 1024));
    } else {
        mlocked_ = false;
        LOGW("[KV] mlock FAILED — KV cache may be swapped to zRAM under memory pressure");
        LOGW("[KV] This will cause latency spikes during decode. Consider running as root.");
    }

    // Hint kernel about our access pattern.
    madvise(buffer_, total_bytes_, MADV_SEQUENTIAL);

    // Record with MemoryGuard.
    g_memory_guard.record_allocation(total_bytes_, "kv_cache");

    current_seq_len_ = 0;

    LOGI("[KV] Allocation complete: %.2f MB at %p (mlocked=%s)",
         total_bytes_ / 1e6, buffer_, mlocked_ ? "yes" : "NO");

    return true;
}

// ── release ─────────────────────────────────────────────────────────────────
void KVCacheManager::release() {
    if (!buffer_) return;

    PROFILE_SCOPE("kv_cache_release");

    // Security: zero memory before release to prevent data leakage.
    // The KV cache contains model activations which could theoretically
    // leak prompt content if another process reads the physical pages.
    memset(buffer_, 0, total_bytes_);

    if (mlocked_) {
        munlock(buffer_, total_bytes_);
        mlocked_ = false;
    }

    munmap(buffer_, total_bytes_);

    g_memory_guard.record_deallocation(total_bytes_, "kv_cache");

    LOGI("[KV] Released %.2f MB", total_bytes_ / 1e6);

    buffer_          = nullptr;
    total_bytes_     = 0;
    current_seq_len_ = 0;
}

// ── get_k_ptr / get_v_ptr ───────────────────────────────────────────────────
void* KVCacheManager::get_k_ptr(int layer, int seq_pos) {
    if (!buffer_ || layer < 0 || layer >= config_.num_layers ||
        seq_pos < 0 || seq_pos >= config_.max_seq_len) {
        return nullptr;
    }
    return static_cast<uint8_t*>(buffer_) + compute_offset(layer, false, seq_pos);
}

void* KVCacheManager::get_v_ptr(int layer, int seq_pos) {
    if (!buffer_ || layer < 0 || layer >= config_.num_layers ||
        seq_pos < 0 || seq_pos >= config_.max_seq_len) {
        return nullptr;
    }
    return static_cast<uint8_t*>(buffer_) + compute_offset(layer, true, seq_pos);
}

void* KVCacheManager::get_k_layer_ptr(int layer) {
    return get_k_ptr(layer, 0);
}

void* KVCacheManager::get_v_layer_ptr(int layer) {
    return get_v_ptr(layer, 0);
}

// ── advance_seq_len / reset_seq_len ─────────────────────────────────────────
void KVCacheManager::advance_seq_len() {
    if (current_seq_len_ < config_.max_seq_len) {
        current_seq_len_++;
    } else {
        LOGW("[KV] Context window full (%d tokens) — cannot advance", config_.max_seq_len);
    }
}

void KVCacheManager::reset_seq_len() {
    current_seq_len_ = 0;
}

// ── log_cache_stats ─────────────────────────────────────────────────────────
void KVCacheManager::log_cache_stats() const {
    if (!buffer_) {
        LOGI("[KV] Not allocated");
        return;
    }

    float utilization = (config_.max_seq_len > 0)
        ? 100.0f * current_seq_len_ / config_.max_seq_len
        : 0.0f;

    LOGI("[KV] Seq: %d / %d (%.1f%% full) | Size: %.2f MB | mlocked: %s",
         current_seq_len_, config_.max_seq_len, utilization,
         total_bytes_ / 1e6, mlocked_ ? "yes" : "no");
}

// ── compute_offset ──────────────────────────────────────────────────────────
size_t KVCacheManager::compute_offset(int layer, bool is_value, int seq_pos) const {
    size_t kv_block = per_layer_kv_bytes();

    // Layout: [L0_K][L0_V][L1_K][L1_V]...[LN_K][LN_V]
    size_t layer_offset = (size_t)layer * 2 * kv_block;
    size_t kv_offset    = is_value ? kv_block : 0;
    size_t pos_offset   = (size_t)seq_pos * config_.num_kv_heads * config_.head_dim * sizeof(uint16_t);

    return layer_offset + kv_offset + pos_offset;
}

// ── per_layer_kv_bytes ──────────────────────────────────────────────────────
size_t KVCacheManager::per_layer_kv_bytes() const {
    return (size_t)config_.num_kv_heads * config_.head_dim *
           config_.max_seq_len * sizeof(uint16_t);
}

} // namespace halfhex
