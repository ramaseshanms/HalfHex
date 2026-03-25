// Profiler.cpp — aggressive timing + logging
#include "Profiler.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <numeric>

Profiler g_profiler;  // global instance

void Profiler::start(const std::string& name) {
    start_times_[name] = std::chrono::high_resolution_clock::now();
}

void Profiler::end(const std::string& name) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto it = start_times_.find(name);
    if (it == start_times_.end()) {
        LOGW("[PROFILE] end() called for '%s' without matching start()", name.c_str());
        return;
    }
    double duration_us = std::chrono::duration<double, std::micro>(
        end_time - it->second).count();

    timings_[name].push_back(duration_us);
}

void Profiler::dump_stats(const std::string& label) {
    LOGI("═══════════════════════════════════════");
    LOGI("[PROFILE] Stats dump: %s", label.c_str());
    LOGI("═══════════════════════════════════════");

    for (auto& [name, times] : timings_) {
        if (times.empty()) continue;

        std::vector<double> sorted = times;
        std::sort(sorted.begin(), sorted.end());

        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        double mean = sum / sorted.size();
        size_t n = sorted.size();

        double p50 = sorted[n / 2];
        double p95 = sorted[(size_t)(n * 0.95)];
        double p99 = sorted[(size_t)(n * 0.99)];

        LOGI("[PROFILE] %-35s | n=%5zu | mean=%.0f μs | p50=%.0f | p95=%.0f | p99=%.0f | min=%.0f | max=%.0f",
             name.c_str(), n, mean, p50, p95, p99, sorted.front(), sorted.back());
    }
    LOGI("═══════════════════════════════════════");
}

void Profiler::compute_percentiles() {
    // Computed inline in dump_stats — this is a hook for external consumers
    dump_stats("percentiles");
}

void Profiler::write_to_file() {
    // Generate timestamped filename
    time_t now = time(nullptr);
    struct tm* t = localtime(&now);
    char filename[256];
    snprintf(filename, sizeof(filename),
             "/sdcard/qnn_profile_%04d%02d%02d_%02d%02d%02d.csv",
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);

    FILE* f = fopen(filename, "w");
    if (!f) {
        LOGE("[PROFILE] Failed to open %s for writing", filename);
        return;
    }

    fprintf(f, "name,duration_us,index\n");
    for (auto& [name, times] : timings_) {
        for (size_t i = 0; i < times.size(); i++) {
            fprintf(f, "%s,%.2f,%zu\n", name.c_str(), times[i], i);
        }
    }

    fclose(f);
    LOGI("[PROFILE] Wrote profiler data to %s", filename);
}

void Profiler::reset() {
    timings_.clear();
    start_times_.clear();
    LOGI("[PROFILE] Profiler reset");
}

void Profiler::print_layer_breakdown() {
    LOGI("═══════════════════════════════════════");
    LOGI("[PROFILE] Layer Breakdown");
    LOGI("═══════════════════════════════════════");

    // Collect all layer-prefixed entries
    std::vector<std::pair<std::string, double>> layer_times;
    double total_layer_time = 0;

    for (auto& [name, times] : timings_) {
        if (name.find("layer_") == 0 || name.find("htp_") == 0) {
            double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            layer_times.push_back({name, mean});
            total_layer_time += mean;
        }
    }

    std::sort(layer_times.begin(), layer_times.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (auto& [name, mean_us] : layer_times) {
        double pct = (total_layer_time > 0) ? 100.0 * mean_us / total_layer_time : 0;
        LOGI("[PROFILE]   %-35s | %.0f μs (%.1f%%)", name.c_str(), mean_us, pct);
    }

    LOGI("[PROFILE] Total layer time: %.0f μs (%.2f ms)", total_layer_time, total_layer_time / 1000.0);
    LOGI("═══════════════════════════════════════");
}
