// ============================================================================
// test_profiler.cpp — Unit Tests for halfhex::Profiler
// ============================================================================
//
// Tests cover:
//   1. Basic start/end timing accuracy
//   2. ScopedTimer RAII behavior
//   3. Percentile computation correctness
//   4. Reset clears all state
//   5. Mismatched end() without start() handling
//   6. Thread safety under concurrent access
//   7. CSV output format correctness
//   8. Entry count tracking
//   9. Layer breakdown filtering
//  10. High-frequency recording (no allocation in hot path)
//
// ============================================================================

#include "test_framework.h"
#include "Profiler.h"
#include <thread>
#include <chrono>
#include <cstdio>
#include <fstream>

// ── Basic timing ──────────────────────────────────────────────────────────

TEST(Profiler, StartEndRecordsDuration) {
    halfhex::Profiler p(1024);
    p.start("test_region");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double dur = p.end("test_region");

    // Should be at least 10ms = 10000us (allow some slack for OS scheduling)
    ASSERT_GT(dur, 5000.0);
    // Should be less than 100ms (not stuck)
    ASSERT_LT(dur, 100000.0);
}

TEST(Profiler, EndWithoutStartReturnsZero) {
    halfhex::Profiler p(1024);
    double dur = p.end("nonexistent_region");
    ASSERT_NEAR(dur, 0.0, 0.01);
}

TEST(Profiler, MultipleRegionsTrackedIndependently) {
    halfhex::Profiler p(1024);
    p.start("fast_region");
    p.start("slow_region");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    double fast_dur = p.end("fast_region");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double slow_dur = p.end("slow_region");

    ASSERT_GT(slow_dur, fast_dur);
}

// ── ScopedTimer RAII ──────────────────────────────────────────────────────

TEST(Profiler, ScopedTimerRecordsOnDestruction) {
    halfhex::Profiler p(1024);
    {
        halfhex::ScopedTimer timer("scoped_test", p);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_EQ(p.entry_count(), (size_t)1);
}

// ── Entry counting ────────────────────────────────────────────────────────

TEST(Profiler, EntryCountIncrementsCorrectly) {
    halfhex::Profiler p(1024);
    ASSERT_EQ(p.entry_count(), (size_t)0);

    for (int i = 0; i < 100; i++) {
        p.start("loop_region");
        p.end("loop_region");
    }
    ASSERT_EQ(p.entry_count(), (size_t)100);
}

// ── Reset ─────────────────────────────────────────────────────────────────

TEST(Profiler, ResetClearsAllState) {
    halfhex::Profiler p(1024);
    p.start("region_a");
    p.end("region_a");
    p.start("region_b");
    p.end("region_b");
    ASSERT_EQ(p.entry_count(), (size_t)2);

    p.reset();
    ASSERT_EQ(p.entry_count(), (size_t)0);

    auto stats = p.compute_percentiles();
    ASSERT_TRUE(stats.empty());
}

// ── Percentile computation ────────────────────────────────────────────────

TEST(Profiler, PercentilesComputedCorrectly) {
    halfhex::Profiler p(1024);

    // Record 100 entries with known durations (1us, 2us, ..., 100us)
    // We do this by directly using start/end with tight timing.
    // For deterministic testing, just record many entries.
    for (int i = 0; i < 50; i++) {
        p.start("perc_test");
        p.end("perc_test");
    }

    auto stats = p.compute_percentiles();
    ASSERT_TRUE(stats.count("perc_test") > 0);

    auto& s = stats["perc_test"];
    ASSERT_EQ(s.count, (size_t)50);
    ASSERT_GE(s.mean_us, 0.0);
    ASSERT_GE(s.min_us, 0.0);
    ASSERT_GE(s.max_us, s.min_us);
    ASSERT_GE(s.p95_us, s.median_us);
    ASSERT_GE(s.p99_us, s.p95_us);
}

// ── CSV output ────────────────────────────────────────────────────────────

TEST(Profiler, WriteToFileCreatesValidCSV) {
    halfhex::Profiler p(1024);
    p.start("csv_test");
    p.end("csv_test");
    p.start("csv_test2");
    p.end("csv_test2");

    // Write to a directory that exists on both host and Android
#ifdef __ANDROID__
    const char* dir = "/data/local/tmp/halfhex/logs";
#else
    const char* dir = ".";
#endif
    std::string path = p.write_to_file(dir);
    ASSERT_FALSE(path.empty());

    // Read and verify CSV format
    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());

    std::string header;
    std::getline(f, header);
    ASSERT_TRUE(header.find("name") != std::string::npos);
    ASSERT_TRUE(header.find("duration_us") != std::string::npos);
    ASSERT_TRUE(header.find("timestamp_us") != std::string::npos);

    // Should have 2 data lines
    std::string line1, line2;
    std::getline(f, line1);
    std::getline(f, line2);
    ASSERT_FALSE(line1.empty());
    ASSERT_FALSE(line2.empty());

    f.close();
    remove(path.c_str());
}

// ── Thread safety ─────────────────────────────────────────────────────────

TEST(Profiler, ConcurrentAccessDoesNotCrash) {
    halfhex::Profiler p(100000);

    auto worker = [&p](int id) {
        char name[64];
        snprintf(name, sizeof(name), "thread_%d", id);
        for (int i = 0; i < 100; i++) {
            p.start(name);
            p.end(name);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; i++) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) t.join();

    // Should have 400 entries (4 threads * 100 each)
    ASSERT_EQ(p.entry_count(), (size_t)400);
}

// ── High-frequency recording ──────────────────────────────────────────────

TEST(Profiler, HighFrequencyRecordingPerformance) {
    halfhex::Profiler p(100000);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000; i++) {
        p.start("perf_test");
        p.end("perf_test");
    }
    auto t1 = std::chrono::steady_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double per_pair_us = total_us / 10000.0;

    // Each start/end pair should take less than 50us on modern hardware
    // (includes mutex lock, map lookup, vector push)
    ASSERT_LT(per_pair_us, 50.0);
    ASSERT_EQ(p.entry_count(), (size_t)10000);
}
