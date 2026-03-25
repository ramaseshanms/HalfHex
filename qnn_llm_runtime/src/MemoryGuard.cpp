// ============================================================================
// MemoryGuard.cpp — Android OOM Protection Implementation
// ============================================================================
//
// See MemoryGuard.h for design rationale.
//
// /proc PARSING NOTES:
//   - /proc/self/status fields are tab-separated: "VmRSS:\t12345 kB"
//   - /proc/meminfo fields are: "MemAvailable:   12345 kB"
//   - Values are in kB (1024 bytes), NOT KiB — Linux convention
//   - These reads are ~2μs on this device (negligible for 2-second checks)
//
// ============================================================================

#include "MemoryGuard.h"
#include <cstdio>
#include <cstring>

namespace halfhex {

MemoryGuard g_memory_guard;

// ── initialize ──────────────────────────────────────────────────────────────
bool MemoryGuard::initialize(const MemoryBudget& budget) {
    budget_ = budget;

    // Set OOM score immediately — this is the most important safety measure.
    set_oom_score(budget_.oom_score_adj);

    // Check if the device has enough memory for our budget.
    size_t avail = get_system_available_bytes();
    if (avail == 0) {
        LOGW("[MEMGUARD] Cannot read MemAvailable — proceeding without pre-check");
    } else if (avail < budget_.max_rss_bytes + budget_.min_system_available) {
        LOGE("[MEMGUARD] INSUFFICIENT MEMORY");
        LOGE("[MEMGUARD]   System available: %zu MB", avail / (1024 * 1024));
        LOGE("[MEMGUARD]   Our budget:       %zu MB", budget_.max_rss_bytes / (1024 * 1024));
        LOGE("[MEMGUARD]   System reserve:   %zu MB", budget_.min_system_available / (1024 * 1024));
        LOGE("[MEMGUARD]   Need at least:    %zu MB",
             (budget_.max_rss_bytes + budget_.min_system_available) / (1024 * 1024));
        return false;
    }

    initialized_ = true;
    log_memory_state("init");
    LOGI("[MEMGUARD] Initialized with budget: RSS=%zu MB, reserve=%zu MB, oom_adj=%d",
         budget_.max_rss_bytes / (1024 * 1024),
         budget_.min_system_available / (1024 * 1024),
         budget_.oom_score_adj);
    return true;
}

// ── can_allocate ────────────────────────────────────────────────────────────
bool MemoryGuard::can_allocate(size_t bytes, const char* label) const {
    if (!initialized_) return true;  // Not initialized = no enforcement

    // Check 1: Would this exceed our total RSS budget?
    size_t projected = total_allocated_.load(std::memory_order_relaxed) + bytes;
    if (projected > budget_.max_rss_bytes) {
        LOGE("[MEMGUARD] DENIED allocation of %zu MB for '%s' — would exceed RSS budget "
             "(%zu + %zu > %zu MB)",
             bytes / (1024 * 1024), label,
             total_allocated_.load(std::memory_order_relaxed) / (1024 * 1024),
             bytes / (1024 * 1024),
             budget_.max_rss_bytes / (1024 * 1024));
        return false;
    }

    // Check 2: Would this leave the system with too little memory?
    size_t avail = get_system_available_bytes();
    if (avail > 0 && avail < bytes + budget_.min_system_available) {
        LOGE("[MEMGUARD] DENIED allocation of %zu MB for '%s' — system would drop below "
             "reserve (avail=%zu MB, need=%zu MB free)",
             bytes / (1024 * 1024), label,
             avail / (1024 * 1024),
             budget_.min_system_available / (1024 * 1024));
        return false;
    }

    return true;
}

// ── record_allocation / record_deallocation ─────────────────────────────────
void MemoryGuard::record_allocation(size_t bytes, const char* category) {
    total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    if (strcmp(category, "kv_cache") == 0) {
        kv_cache_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    } else if (strcmp(category, "model_weights") == 0) {
        model_weight_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    } else if (strcmp(category, "activations") == 0) {
        activation_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    }
}

void MemoryGuard::record_deallocation(size_t bytes, const char* category) {
    total_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    if (strcmp(category, "kv_cache") == 0) {
        kv_cache_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    } else if (strcmp(category, "model_weights") == 0) {
        model_weight_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    } else if (strcmp(category, "activations") == 0) {
        activation_allocated_.fetch_sub(bytes, std::memory_order_relaxed);
    }
}

// ── get_current_rss_bytes ───────────────────────────────────────────────────
size_t MemoryGuard::get_current_rss_bytes() const {
    return read_proc_kb("/proc/self/status", "VmRSS:") * 1024;
}

// ── get_system_available_bytes ──────────────────────────────────────────────
size_t MemoryGuard::get_system_available_bytes() const {
    return read_proc_kb("/proc/meminfo", "MemAvailable:") * 1024;
}

// ── is_memory_pressure ──────────────────────────────────────────────────────
bool MemoryGuard::is_memory_pressure() const {
    size_t rss = get_current_rss_bytes();
    return rss > (budget_.max_rss_bytes * 80 / 100);
}

// ── should_emergency_stop ───────────────────────────────────────────────────
bool MemoryGuard::should_emergency_stop() const {
    size_t avail = get_system_available_bytes();
    return avail > 0 && avail < budget_.min_system_available;
}

// ── log_memory_state ────────────────────────────────────────────────────────
void MemoryGuard::log_memory_state(const char* label) const {
    size_t rss   = get_current_rss_bytes();
    size_t avail = get_system_available_bytes();

    LOGI("[MEMGUARD][%s] RSS=%zu MB | Tracked=%zu MB (KV=%zu, Weights=%zu, Act=%zu) | "
         "SysAvail=%zu MB | Pressure=%s",
         label,
         rss / (1024 * 1024),
         total_allocated_.load(std::memory_order_relaxed) / (1024 * 1024),
         kv_cache_allocated_.load(std::memory_order_relaxed) / (1024 * 1024),
         model_weight_allocated_.load(std::memory_order_relaxed) / (1024 * 1024),
         activation_allocated_.load(std::memory_order_relaxed) / (1024 * 1024),
         avail / (1024 * 1024),
         is_memory_pressure() ? "YES" : "no");
}

// ── set_oom_score ───────────────────────────────────────────────────────────
void MemoryGuard::set_oom_score(int score) {
    FILE* f = fopen("/proc/self/oom_score_adj", "w");
    if (!f) {
        LOGW("[MEMGUARD] Cannot set oom_score_adj (not root?) — device apps are at risk");
        return;
    }
    fprintf(f, "%d", score);
    fclose(f);
    LOGI("[MEMGUARD] oom_score_adj set to %d (higher = killed first)", score);
}

// ── read_proc_kb ────────────────────────────────────────────────────────────
size_t MemoryGuard::read_proc_kb(const char* path, const char* key) const {
    FILE* f = fopen(path, "r");
    if (!f) return 0;

    char line[256];
    size_t value_kb = 0;
    size_t key_len = strlen(key);

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, key_len) == 0) {
            // Parse: "VmRSS:\t12345 kB\n"
            const char* p = line + key_len;
            while (*p == ' ' || *p == '\t') p++;
            value_kb = (size_t)strtoull(p, nullptr, 10);
            break;
        }
    }
    fclose(f);
    return value_kb;
}

} // namespace halfhex
