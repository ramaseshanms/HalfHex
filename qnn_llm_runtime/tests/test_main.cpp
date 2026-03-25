// ============================================================================
// test_main.cpp — HalfHex Test Suite Entry Point
// ============================================================================
//
// Runs all registered tests from:
//   - test_profiler.cpp        (unit tests for Profiler)
//   - test_memory_guard.cpp    (unit tests for MemoryGuard)
//   - test_kv_cache.cpp        (unit tests for KVCacheManager)
//   - test_thermal_monitor.cpp (unit tests for ThermalMonitor)
//   - test_accuracy.cpp        (numerical accuracy tests)
//   - test_perf_regression.cpp (performance regression tests)
//
// BUILD:
//   Host (Linux/macOS/Windows):
//     g++ -std=c++17 -DPROFILING_ENABLED -I../include \
//         test_main.cpp test_profiler.cpp test_memory_guard.cpp \
//         test_kv_cache.cpp test_thermal_monitor.cpp \
//         test_accuracy.cpp test_perf_regression.cpp \
//         ../src/Profiler.cpp ../src/MemoryGuard.cpp \
//         ../src/KVCacheManager.cpp ../src/ThermalMonitor.cpp \
//         -lpthread -o halfhex_tests
//     ./halfhex_tests
//
//   Android (via CMake + NDK):
//     cmake --build build-android --target halfhex_tests
//     adb push build-android/halfhex_tests /data/local/tmp/halfhex/bin/
//     adb shell "/data/local/tmp/halfhex/bin/halfhex_tests"
//
// EXIT CODE:
//   0 = all tests passed
//   1 = one or more tests failed
//
// ============================================================================

#include "test_framework.h"

int main() {
    return halfhex::test::run_all_tests();
}
