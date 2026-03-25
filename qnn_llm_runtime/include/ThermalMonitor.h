// ThermalMonitor.h
// Snapdragon 7s Gen 3 WILL throttle — detect it before it destroys your benchmark
#pragma once

#include "Profiler.h"
#include <cstdio>
#include <string>

class ThermalMonitor {
public:
    float get_cpu_temp() {
        // Read from thermal zone — zone 0 is typically CPU on SD 7s Gen 3
        FILE* f = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
        if (!f) {
            LOGW("[THERMAL] Cannot read thermal zone 0");
            return -1.0f;
        }
        float temp_milli = 0;
        if (fscanf(f, "%f", &temp_milli) != 1) {
            fclose(f);
            return -1.0f;
        }
        fclose(f);
        return temp_milli / 1000.0f;
    }

    bool is_throttling() {
        // Compare current CPU freq vs max freq
        FILE* f = fopen("/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq", "r");
        if (!f) return false;
        int cur = 0;
        if (fscanf(f, "%d", &cur) != 1) { fclose(f); return false; }
        fclose(f);

        f = fopen("/sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq", "r");
        if (!f) return false;
        int max = 0;
        if (fscanf(f, "%d", &max) != 1) { fclose(f); return false; }
        fclose(f);

        if (max == 0) return false;

        float ratio = (float)cur / max;
        if (ratio < 0.85f) {
            LOGW("[PROFILE][THERMAL] THROTTLING DETECTED: CPU at %.0f%% max freq (%.0f°C)",
                 ratio * 100, get_cpu_temp());
            return true;
        }
        return false;
    }

    void log_thermal_snapshot(const std::string& label) {
        LOGI("[PROFILE][THERMAL][%s] CPU: %.1f°C | Throttling: %s",
             label.c_str(), get_cpu_temp(), is_throttling() ? "YES" : "NO");
    }
};
