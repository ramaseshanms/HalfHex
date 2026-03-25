// ============================================================================
// ThermalMonitor.h — Real-Time Thermal & Throttle Detection
// ============================================================================
//
// PURPOSE:
//   Detects thermal throttling on the Snapdragon 7s Gen 3 before it
//   corrupts benchmark data. Mobile SoCs aggressively reduce clock speeds
//   when junction temperature exceeds safe limits. If you don't detect this,
//   your "30 tok/s" benchmark is actually "30 tok/s for 10 seconds then
//   15 tok/s for the rest" — and you won't know which number is real.
//
// NOTHING PHONE 3a PRO THERMAL ZONES:
//   Zone 0 (shell_front): Front panel temperature sensor
//   Zone 1 (shell_frame): Metal frame temperature
//   Zone 2 (shell_back):  Back panel temperature
//   Zone 3 (shell_max):   Max of all shell sensors
//   Zone 4 (pmxr2230-bcl-lvl0): Battery Current Limit sensor
//
//   Values are in millidegrees Celsius (e.g., 33796 = 33.8°C).
//
// SNAPDRAGON 7s GEN 3 CPU TOPOLOGY:
//   cpu0-cpu3: Cortex-A510 "LITTLE" cores @ 1.8 GHz max
//   cpu4-cpu7: Cortex-A715 "BIG" cores @ 2.4 GHz max
//
//   We run inference on BIG cores (taskset 0xF0). Monitor cpu4's frequency
//   to detect throttling — if it drops below 85% of max, we're throttled.
//
// SAFE OPERATING LIMITS:
//   < 40°C: Normal operation, no throttling expected
//   40-50°C: Sustained load zone, may see brief throttle events
//   50-55°C: Extended throttling likely, benchmark data questionable
//   > 55°C: ABORT — prolonged operation risks hardware damage
//
// ============================================================================

#pragma once

#include "Profiler.h"
#include <cstdio>
#include <string>
#include <cstdint>

namespace halfhex {

// ── Thermal thresholds (degrees Celsius) ────────────────────────────────────
constexpr float THERMAL_NORMAL_MAX    = 40.0f;  // No throttling expected
constexpr float THERMAL_WARNING       = 50.0f;  // Benchmark data may be affected
constexpr float THERMAL_CRITICAL      = 55.0f;  // Must stop inference
constexpr float THERMAL_EMERGENCY     = 60.0f;  // Hardware damage risk

// ── Throttle detection threshold ────────────────────────────────────────────
constexpr float THROTTLE_RATIO_THRESHOLD = 0.85f; // Current/Max frequency

// ── ThermalSnapshot ─────────────────────────────────────────────────────────
// A point-in-time reading of all thermal/frequency sensors.
// ────────────────────────────────────────────────────────────────────────────
struct ThermalSnapshot {
    float cpu_temp_c;         // shell_front temperature (°C)
    float shell_back_c;       // shell_back temperature (°C)
    float battery_temp_c;     // Battery temperature (°C, from dumpsys)
    int   big_core_freq_khz;  // cpu4 current frequency (kHz)
    int   big_core_max_khz;   // cpu4 max frequency (kHz)
    float throttle_pct;       // Current/Max as percentage (100% = no throttle)
    bool  is_throttling;      // True if throttle_pct < 85%
    int64_t timestamp_ms;     // Wall clock timestamp
};

// ── ThermalMonitor ──────────────────────────────────────────────────────────
class ThermalMonitor {
public:
    // Read CPU temperature from thermal zone 0 (shell_front).
    // Returns -1.0 if the sensor is unreadable.
    float get_cpu_temp();

    // Read back panel temperature from thermal zone 2.
    float get_back_temp();

    // Check if the big core cluster is throttled.
    // Compares cpu4's current frequency against its max.
    bool is_throttling();

    // Get the current throttle percentage (100% = no throttle, 0% = fully throttled).
    float get_throttle_pct();

    // Take a complete thermal snapshot (all sensors + frequency).
    ThermalSnapshot snapshot();

    // Log thermal state to logcat with a human-readable label.
    void log_thermal_snapshot(const char* label);

    // Check if temperature is safe for continued operation.
    // Returns false if temperature exceeds THERMAL_CRITICAL.
    bool is_safe_to_continue();

    // Block until device cools below target temperature.
    // Logs progress every `check_interval_ms` milliseconds.
    // Returns false if temperature doesn't drop within timeout_seconds.
    bool wait_for_cooldown(float target_c, int timeout_seconds = 300,
                           int check_interval_ms = 5000);

private:
    // Read a single integer from a sysfs file.
    // Returns 0 on failure.
    int read_sysfs_int(const char* path);

    // Read a float-like integer from thermal zone (millidegrees).
    float read_thermal_zone(int zone_id);
};

} // namespace halfhex
