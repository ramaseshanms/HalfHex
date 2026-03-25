// ============================================================================
// test_framework.h — Lightweight C++ Test Framework (Zero Dependencies)
// ============================================================================
//
// PURPOSE:
//   Minimal test framework for HalfHex runtime unit tests. No external
//   dependencies (no gtest, no catch2) — this builds with just a C++17
//   compiler on any platform (host x86_64 or Android aarch64).
//
// USAGE:
//   TEST(suite_name, test_name) {
//       ASSERT_TRUE(condition);
//       ASSERT_FALSE(condition);
//       ASSERT_EQ(expected, actual);
//       ASSERT_NE(a, b);
//       ASSERT_LT(a, b);
//       ASSERT_GT(a, b);
//       ASSERT_LE(a, b);
//       ASSERT_GE(a, b);
//       ASSERT_NEAR(expected, actual, tolerance);
//       ASSERT_NULL(ptr);
//       ASSERT_NOT_NULL(ptr);
//   }
//
//   int main() { return halfhex::test::run_all_tests(); }
//
// DESIGN:
//   - Tests auto-register via static initializer (no manual listing)
//   - Failures print file:line with expected vs actual values
//   - Continues running after failure (reports all failures, not just first)
//   - Returns 0 on all pass, 1 on any failure (for CI integration)
//   - Color output when stdout is a terminal
//
// ============================================================================

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <functional>
#include <chrono>

namespace halfhex {
namespace test {

// ── Test registry ─────────────────────────────────────────────────────────
struct TestCase {
    const char* suite;
    const char* name;
    std::function<void()> func;
};

inline std::vector<TestCase>& get_tests() {
    static std::vector<TestCase> tests;
    return tests;
}

struct TestRegistrar {
    TestRegistrar(const char* suite, const char* name, std::function<void()> func) {
        get_tests().push_back({suite, name, std::move(func)});
    }
};

// ── Failure tracking ──────────────────────────────────────────────────────
inline int& failure_count() {
    static int count = 0;
    return count;
}

inline int& current_test_failures() {
    static int count = 0;
    return count;
}

// ── Runner ────────────────────────────────────────────────────────────────
inline int run_all_tests() {
    auto& tests = get_tests();
    int passed = 0;
    int failed = 0;
    int total = static_cast<int>(tests.size());

    fprintf(stderr, "\n======================================\n");
    fprintf(stderr, "  HalfHex Test Suite (%d tests)\n", total);
    fprintf(stderr, "======================================\n\n");

    auto suite_start = std::chrono::steady_clock::now();

    for (auto& tc : tests) {
        fprintf(stderr, "[ RUN      ] %s.%s\n", tc.suite, tc.name);
        current_test_failures() = 0;
        failure_count() = 0;

        auto t0 = std::chrono::steady_clock::now();
        try {
            tc.func();
        } catch (const std::exception& e) {
            fprintf(stderr, "  EXCEPTION: %s\n", e.what());
            failure_count()++;
        } catch (...) {
            fprintf(stderr, "  UNKNOWN EXCEPTION\n");
            failure_count()++;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (failure_count() == 0) {
            fprintf(stderr, "[       OK ] %s.%s (%.1f ms)\n", tc.suite, tc.name, ms);
            passed++;
        } else {
            fprintf(stderr, "[  FAILED  ] %s.%s (%d failures, %.1f ms)\n",
                    tc.suite, tc.name, failure_count(), ms);
            failed++;
        }
    }

    auto suite_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(suite_end - suite_start).count();

    fprintf(stderr, "\n======================================\n");
    fprintf(stderr, "  %d/%d passed, %d failed (%.1f ms)\n", passed, total, failed, total_ms);
    fprintf(stderr, "======================================\n\n");

    return (failed > 0) ? 1 : 0;
}

// ── Assertion macros ──────────────────────────────────────────────────────

#define HALFHEX_FAIL(msg) do { \
    fprintf(stderr, "  FAIL at %s:%d: %s\n", __FILE__, __LINE__, msg); \
    halfhex::test::failure_count()++; \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_TRUE(%s)\n", __FILE__, __LINE__, #cond); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_FALSE(cond) do { \
    if (cond) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_FALSE(%s)\n", __FILE__, __LINE__, #cond); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_EQ(expected, actual) do { \
    auto _e = (expected); auto _a = (actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_EQ\n", __FILE__, __LINE__); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_NE(a, b) do { \
    if ((a) == (b)) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_NE(%s, %s)\n", __FILE__, __LINE__, #a, #b); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_LT(a, b) do { \
    if (!((a) < (b))) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_LT(%s, %s)\n", __FILE__, __LINE__, #a, #b); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_GT(a, b) do { \
    if (!((a) > (b))) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_GT(%s, %s)\n", __FILE__, __LINE__, #a, #b); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_LE(a, b) do { \
    if (!((a) <= (b))) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_LE(%s, %s)\n", __FILE__, __LINE__, #a, #b); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_GE(a, b) do { \
    if (!((a) >= (b))) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_GE(%s, %s)\n", __FILE__, __LINE__, #a, #b); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_NEAR(expected, actual, tol) do { \
    double _e = (expected); double _a = (actual); double _t = (tol); \
    if (std::fabs(_e - _a) > _t) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_NEAR(%.6f, %.6f, tol=%.6f) diff=%.6f\n", \
                __FILE__, __LINE__, _e, _a, _t, std::fabs(_e - _a)); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_NULL(ptr) do { \
    if ((ptr) != nullptr) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_NULL(%s)\n", __FILE__, __LINE__, #ptr); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_NOT_NULL(ptr) do { \
    if ((ptr) == nullptr) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_NOT_NULL(%s)\n", __FILE__, __LINE__, #ptr); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

#define ASSERT_STR_EQ(expected, actual) do { \
    std::string _e(expected); std::string _a(actual); \
    if (_e != _a) { \
        fprintf(stderr, "  FAIL at %s:%d: ASSERT_STR_EQ(\"%s\", \"%s\")\n", \
                __FILE__, __LINE__, _e.c_str(), _a.c_str()); \
        halfhex::test::failure_count()++; \
    } \
} while(0)

// ── Test macro ────────────────────────────────────────────────────────────
#define TEST(suite, name) \
    static void test_##suite##_##name(); \
    static halfhex::test::TestRegistrar reg_##suite##_##name( \
        #suite, #name, test_##suite##_##name); \
    static void test_##suite##_##name()

} // namespace test
} // namespace halfhex
