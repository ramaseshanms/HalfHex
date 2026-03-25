// Profiler.h — embed this in EVERY inference function
#pragma once
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <android/log.h>

#define LOG_TAG "QNN_RUNTIME"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// PROFILE_SCOPE: drop this at the start of any function you want timed
// It logs automatically on scope exit. Zero overhead in release if PROFILING=0
#ifdef PROFILING_ENABLED
#define PROFILE_SCOPE(name) ScopedTimer _timer_##__LINE__(name, g_profiler)
#define PROFILE_START(name) g_profiler.start(name)
#define PROFILE_END(name)   g_profiler.end(name)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_START(name)
#define PROFILE_END(name)
#endif

struct TimingEntry {
    std::string name;
    double duration_us;   // microseconds — ms is too coarse for layer-level work
    int64_t timestamp_us;
};

class Profiler {
public:
    void start(const std::string& name);
    void end(const std::string& name);

    // Call after every N tokens — dumps to logcat + file
    void dump_stats(const std::string& label);

    // Computes: mean, p50, p95, p99, min, max per named region
    void compute_percentiles();

    // Writes CSV to /sdcard/qnn_profile_TIMESTAMP.csv for offline analysis
    void write_to_file();

    // Resets all accumulators — call at start of each benchmark run
    void reset();

    // Per-layer breakdown — how many ms did each transformer layer take?
    void print_layer_breakdown();

private:
    std::unordered_map<std::string, std::vector<double>> timings_;
    std::unordered_map<std::string,
        std::chrono::time_point<std::chrono::high_resolution_clock>> start_times_;
};

struct ScopedTimer {
    ScopedTimer(const std::string& name, Profiler& p) : name_(name), p_(p) {
        p_.start(name_);
    }
    ~ScopedTimer() { p_.end(name_); }
    std::string name_;
    Profiler& p_;
};

extern Profiler g_profiler;  // global — accessible from any file
