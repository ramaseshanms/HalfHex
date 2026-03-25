// ============================================================================
// ThermalMonitor.cpp — Thermal & Throttle Detection Implementation
// ============================================================================

#include "ThermalMonitor.h"
#include <chrono>
#include <thread>

namespace halfhex {

// ── get_cpu_temp ────────────────────────────────────────────────────────────
float ThermalMonitor::get_cpu_temp() {
    return read_thermal_zone(0);  // shell_front
}

// ── get_back_temp ───────────────────────────────────────────────────────────
float ThermalMonitor::get_back_temp() {
    return read_thermal_zone(2);  // shell_back
}

// ── is_throttling ───────────────────────────────────────────────────────────
bool ThermalMonitor::is_throttling() {
    return get_throttle_pct() < (THROTTLE_RATIO_THRESHOLD * 100.0f);
}

// ── get_throttle_pct ────────────────────────────────────────────────────────
float ThermalMonitor::get_throttle_pct() {
    int cur = read_sysfs_int("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq");
    int max = read_sysfs_int("/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq");

    if (max <= 0) return 100.0f;  // Can't determine — assume no throttle
    return 100.0f * (float)cur / (float)max;
}

// ── snapshot ────────────────────────────────────────────────────────────────
ThermalSnapshot ThermalMonitor::snapshot() {
    ThermalSnapshot s = {};

    s.cpu_temp_c       = read_thermal_zone(0);
    s.shell_back_c     = read_thermal_zone(2);
    s.battery_temp_c   = -1.0f;  // Requires dumpsys, too slow for hot path
    s.big_core_freq_khz = read_sysfs_int("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq");
    s.big_core_max_khz  = read_sysfs_int("/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq");

    if (s.big_core_max_khz > 0) {
        s.throttle_pct = 100.0f * (float)s.big_core_freq_khz / (float)s.big_core_max_khz;
    } else {
        s.throttle_pct = 100.0f;
    }

    s.is_throttling = s.throttle_pct < (THROTTLE_RATIO_THRESHOLD * 100.0f);
    s.timestamp_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    return s;
}

// ── log_thermal_snapshot ────────────────────────────────────────────────────
void ThermalMonitor::log_thermal_snapshot(const char* label) {
    ThermalSnapshot s = snapshot();

    const char* status = "NORMAL";
    if (s.cpu_temp_c >= THERMAL_EMERGENCY) status = "***EMERGENCY***";
    else if (s.cpu_temp_c >= THERMAL_CRITICAL) status = "**CRITICAL**";
    else if (s.cpu_temp_c >= THERMAL_WARNING) status = "*WARNING*";

    LOGI("[THERMAL][%s] CPU=%.1f°C | Back=%.1f°C | BigCore=%d/%d kHz (%.0f%%) | %s%s",
         label,
         s.cpu_temp_c, s.shell_back_c,
         s.big_core_freq_khz, s.big_core_max_khz, s.throttle_pct,
         status,
         s.is_throttling ? " | THROTTLED" : "");
}

// ── is_safe_to_continue ─────────────────────────────────────────────────────
bool ThermalMonitor::is_safe_to_continue() {
    float temp = get_cpu_temp();
    if (temp < 0) return true;  // Sensor unreadable — continue cautiously

    if (temp >= THERMAL_CRITICAL) {
        LOGE("[THERMAL] CRITICAL: %.1f°C exceeds safe limit of %.1f°C — MUST STOP",
             temp, THERMAL_CRITICAL);
        return false;
    }
    return true;
}

// ── wait_for_cooldown ───────────────────────────────────────────────────────
bool ThermalMonitor::wait_for_cooldown(float target_c, int timeout_seconds,
                                        int check_interval_ms) {
    LOGI("[THERMAL] Waiting for cooldown to %.1f°C (timeout: %ds)...", target_c, timeout_seconds);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);

    while (std::chrono::steady_clock::now() < deadline) {
        float temp = get_cpu_temp();
        if (temp >= 0 && temp <= target_c) {
            LOGI("[THERMAL] Cooled to %.1f°C — ready to proceed", temp);
            return true;
        }
        LOGI("[THERMAL] Current: %.1f°C (target: %.1f°C) — waiting...", temp, target_c);
        std::this_thread::sleep_for(std::chrono::milliseconds(check_interval_ms));
    }

    LOGW("[THERMAL] Timeout: did not cool to %.1f°C within %d seconds", target_c, timeout_seconds);
    return false;
}

// ── read_sysfs_int ──────────────────────────────────────────────────────────
int ThermalMonitor::read_sysfs_int(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    int val = 0;
    if (fscanf(f, "%d", &val) != 1) val = 0;
    fclose(f);
    return val;
}

// ── read_thermal_zone ───────────────────────────────────────────────────────
float ThermalMonitor::read_thermal_zone(int zone_id) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", zone_id);
    int millideg = read_sysfs_int(path);
    if (millideg == 0) return -1.0f;
    return (float)millideg / 1000.0f;
}

} // namespace halfhex
