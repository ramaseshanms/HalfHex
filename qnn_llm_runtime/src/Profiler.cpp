// ============================================================================
// Profiler.cpp — Microsecond-Resolution Inference Profiler Implementation
// ============================================================================
//
// See Profiler.h for design rationale and usage.
//
// IMPLEMENTATION NOTES:
//   - std::chrono::high_resolution_clock on ARM64 Android maps to
//     clock_gettime(CLOCK_MONOTONIC), which reads the ARM generic timer
//     register (CNTVCT_EL0) — ~52ns overhead per call.
//   - The mutex in start()/end() is uncontended in the typical single-
//     threaded inference path. If profiling multi-threaded prefill, the
//     contention is bounded by layer count (~28 layers for Qwen3-1.7B).
//   - CSV output uses fprintf (not iostream) for minimal overhead and
//     deterministic formatting on Android.
//
// ============================================================================

#include "Profiler.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <ctime>
#include <numeric>

namespace halfhex {

// ── Global instance ─────────────────────────────────────────────────────────
Profiler g_profiler;

// ── Constructor ─────────────────────────────────────────────────────────────
Profiler::Profiler(size_t reserve_entries) {
    entries_.reserve(reserve_entries);
    timings_.reserve(64);      // Expect ~30 layers + ~30 named regions
    start_times_.reserve(64);
}

// ── start ───────────────────────────────────────────────────────────────────
void Profiler::start(const char* name) {
    auto now = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    start_times_[name] = now;
}

// ── end ─────────────────────────────────────────────────────────────────────
double Profiler::end(const char* name) {
    auto end_time = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = start_times_.find(name);
    if (it == start_times_.end()) {
        LOGW("[PROFILER] end() called for '%s' without matching start()", name);
        return 0.0;
    }

    double duration_us = std::chrono::duration<double, std::micro>(
        end_time - it->second).count();

    int64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time.time_since_epoch()).count();

    // Append to raw entries (for CSV export).
    entries_.push_back({name, duration_us, timestamp_us});

    // Append to per-region accumulator (for percentile stats).
    timings_[name].push_back(duration_us);

    return duration_us;
}

// ── dump_stats ──────────────────────────────────────────────────────────────
void Profiler::dump_stats(const char* label) {
    std::lock_guard<std::mutex> lock(mutex_);

    LOGI("============================================================");
    LOGI("[PROFILER] Statistics: %s (%zu total entries)", label, entries_.size());
    LOGI("============================================================");
    LOGI("%-35s | %6s | %10s | %10s | %10s | %10s | %10s | %10s",
         "Region", "Count", "Mean(us)", "P50(us)", "P95(us)", "P99(us)", "Min(us)", "Max(us)");
    LOGI("------------------------------------------------------------");

    for (auto& [name, times] : timings_) {
        if (times.empty()) continue;

        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());

        double sum  = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        double mean = sum / sorted.size();
        size_t n    = sorted.size();

        double p50 = sorted[n / 2];
        double p95 = sorted[std::min(n - 1, (size_t)(n * 0.95))];
        double p99 = sorted[std::min(n - 1, (size_t)(n * 0.99))];

        LOGI("%-35s | %6zu | %10.0f | %10.0f | %10.0f | %10.0f | %10.0f | %10.0f",
             name.c_str(), n, mean, p50, p95, p99, sorted.front(), sorted.back());
    }
    LOGI("============================================================");
}

// ── compute_percentiles ─────────────────────────────────────────────────────
std::unordered_map<std::string, PercentileStats> Profiler::compute_percentiles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, PercentileStats> result;

    for (auto& [name, times] : timings_) {
        if (times.empty()) continue;

        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();

        PercentileStats stats;
        stats.count     = n;
        stats.mean_us   = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;
        stats.median_us = sorted[n / 2];
        stats.p95_us    = sorted[std::min(n - 1, (size_t)(n * 0.95))];
        stats.p99_us    = sorted[std::min(n - 1, (size_t)(n * 0.99))];
        stats.min_us    = sorted.front();
        stats.max_us    = sorted.back();

        result[name] = stats;
    }
    return result;
}

// ── write_to_file ───────────────────────────────────────────────────────────
std::string Profiler::write_to_file(const char* directory) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Generate timestamped filename.
    time_t now = time(nullptr);
    struct tm* t = localtime(&now);
    char filename[512];

    const char* dir = directory;
#ifdef __ANDROID__
    if (!dir) dir = "/data/local/tmp/halfhex/profiles";
#else
    if (!dir) dir = "./logs";
#endif

    snprintf(filename, sizeof(filename),
             "%s/profile_%04d%02d%02d_%02d%02d%02d.csv",
             dir,
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);

    FILE* f = fopen(filename, "w");
    if (!f) {
        LOGE("[PROFILER] Failed to open '%s' for writing", filename);
        return "";
    }

    // Header.
    fprintf(f, "name,duration_us,timestamp_us,index\n");

    // Write all entries.
    // Track per-name index for timeline analysis.
    std::unordered_map<std::string, size_t> name_index;
    for (const auto& entry : entries_) {
        size_t idx = name_index[entry.name]++;
        fprintf(f, "%s,%.2f,%" PRId64 ",%zu\n",
                entry.name, entry.duration_us, entry.timestamp_us, idx);
    }

    fclose(f);
    LOGI("[PROFILER] Wrote %zu entries to %s", entries_.size(), filename);
    return std::string(filename);
}

// ── reset ───────────────────────────────────────────────────────────────────
void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    start_times_.clear();
    timings_.clear();
    LOGI("[PROFILER] Reset — all accumulators cleared");
}

// ── print_layer_breakdown ───────────────────────────────────────────────────
void Profiler::print_layer_breakdown() {
    std::lock_guard<std::mutex> lock(mutex_);

    LOGI("============================================================");
    LOGI("[PROFILER] Transformer Layer Breakdown");
    LOGI("============================================================");

    // Collect all entries with "layer_" or "htp_" prefix.
    std::vector<std::pair<std::string, double>> layer_times;
    double total_layer_time = 0;

    for (auto& [name, times] : timings_) {
        if (name.find("layer_") == 0 || name.find("htp_") == 0 ||
            name.find("prefill_") == 0 || name.find("decode_") == 0) {
            double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            layer_times.push_back({name, mean});
            total_layer_time += mean;
        }
    }

    // Sort by mean duration (descending) to show hottest regions first.
    std::sort(layer_times.begin(), layer_times.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (auto& [name, mean_us] : layer_times) {
        double pct = (total_layer_time > 0) ? 100.0 * mean_us / total_layer_time : 0;
        LOGI("  %-35s | %10.0f us (%5.1f%%)", name.c_str(), mean_us, pct);
    }

    if (total_layer_time > 0) {
        LOGI("  %-35s | %10.0f us (100.0%%)", "TOTAL", total_layer_time);
    }
    LOGI("============================================================");
}

// ── entry_count ─────────────────────────────────────────────────────────────
size_t Profiler::entry_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

} // namespace halfhex
