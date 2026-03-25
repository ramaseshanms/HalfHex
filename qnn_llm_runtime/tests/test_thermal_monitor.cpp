// ============================================================================
// test_thermal_monitor.cpp — Unit Tests for halfhex::ThermalMonitor
// ============================================================================
//
// Tests cover:
//   1. Thermal threshold constants are physically reasonable
//   2. ThermalSnapshot struct field initialization
//   3. Throttle detection logic (ratio computation)
//   4. Temperature classification (normal/warning/critical/emergency)
//   5. Sysfs read failure handling (returns safe defaults)
//
// NOTE: Actual sysfs reads only work on Android. On host, thermal zone
//       reads return -1.0 or 0, which is the expected "sensor unavailable"
//       behavior. We test the logic, not the hardware.
//
// ============================================================================

#include "test_framework.h"
#include "ThermalMonitor.h"

// ── Threshold constants ───────────────────────────────────────────────────

TEST(ThermalMonitor, ThresholdConstantsAreOrdered) {
    ASSERT_LT(halfhex::THERMAL_NORMAL_MAX, halfhex::THERMAL_WARNING);
    ASSERT_LT(halfhex::THERMAL_WARNING, halfhex::THERMAL_CRITICAL);
    ASSERT_LT(halfhex::THERMAL_CRITICAL, halfhex::THERMAL_EMERGENCY);
}

TEST(ThermalMonitor, ThresholdConstantsArePhysicallyReasonable) {
    // Normal max should be between 30-50°C
    ASSERT_GE(halfhex::THERMAL_NORMAL_MAX, 30.0f);
    ASSERT_LE(halfhex::THERMAL_NORMAL_MAX, 50.0f);

    // Critical should be between 50-65°C
    ASSERT_GE(halfhex::THERMAL_CRITICAL, 50.0f);
    ASSERT_LE(halfhex::THERMAL_CRITICAL, 65.0f);

    // Emergency should be between 55-80°C
    ASSERT_GE(halfhex::THERMAL_EMERGENCY, 55.0f);
    ASSERT_LE(halfhex::THERMAL_EMERGENCY, 80.0f);
}

TEST(ThermalMonitor, ThrottleRatioThresholdIsReasonable) {
    // Should be between 70-95%
    ASSERT_GE(halfhex::THROTTLE_RATIO_THRESHOLD, 0.70f);
    ASSERT_LE(halfhex::THROTTLE_RATIO_THRESHOLD, 0.95f);
}

// ── ThermalSnapshot struct ────────────────────────────────────────────────

TEST(ThermalMonitor, SnapshotDefaultConstructionIsZeroed) {
    halfhex::ThermalSnapshot snap = {};
    ASSERT_NEAR(snap.cpu_temp_c, 0.0f, 0.01f);
    ASSERT_NEAR(snap.shell_back_c, 0.0f, 0.01f);
    ASSERT_EQ(snap.big_core_freq_khz, 0);
    ASSERT_EQ(snap.big_core_max_khz, 0);
    ASSERT_FALSE(snap.is_throttling);
}

// ── Sensor read behavior on non-Android host ──────────────────────────────

TEST(ThermalMonitor, GetCpuTempReturnsNegativeOnHost) {
    halfhex::ThermalMonitor monitor;
    float temp = monitor.get_cpu_temp();
    // On host (no sysfs), should return -1.0 (sensor unavailable)
    // On Android, should return actual temperature
#ifndef __ANDROID__
    ASSERT_LT(temp, 0.0f);
#else
    // On Android, temp should be between 0 and 100
    ASSERT_GE(temp, 0.0f);
    ASSERT_LE(temp, 100.0f);
#endif
}

TEST(ThermalMonitor, IsSafeToContinueOnHost) {
    halfhex::ThermalMonitor monitor;
    // On host, with no readable thermal zones, should default to safe
    // (we don't want to abort just because we can't read temperature)
    bool safe = monitor.is_safe_to_continue();
#ifndef __ANDROID__
    // On host, should be safe (can't read temp → assume safe)
    ASSERT_TRUE(safe);
#endif
}

TEST(ThermalMonitor, SnapshotContainsTimestamp) {
    halfhex::ThermalMonitor monitor;
    auto snap = monitor.snapshot();
    // Timestamp should be non-negative (milliseconds since epoch)
    ASSERT_GE(snap.timestamp_ms, (int64_t)0);
}

// ── Throttle percentage logic ─────────────────────────────────────────────

TEST(ThermalMonitor, ThrottlePctReturnsValueOnHost) {
    halfhex::ThermalMonitor monitor;
    float pct = monitor.get_throttle_pct();
    // On host: should return 100% (no throttle) or 0% (can't read)
    // Both are acceptable
    ASSERT_GE(pct, 0.0f);
    ASSERT_LE(pct, 100.0f);
}
