#!/system/bin/sh
# ============================================================================
# HalfHex Device Sandbox Initializer
# ============================================================================
#
# PURPOSE:
#   Creates an isolated, self-contained working environment on the Nothing
#   Phone 3a Pro under /data/local/tmp/halfhex/. This script guarantees:
#
#   1. ISOLATION   — All HalfHex files live under one directory tree.
#                    Nothing outside /data/local/tmp/halfhex/ is modified.
#   2. OOM SAFETY  — Memory limits prevent the inference process from
#                    triggering Android's low-memory killer on user apps.
#   3. CLEANUP     — A companion cleanup script removes everything
#                    we created, leaving the device exactly as found.
#   4. THERMAL     — Baseline thermal readings are captured so benchmarks
#                    can detect if the device was already hot.
#
# USAGE:
#   adb push device/sandbox_init.sh /data/local/tmp/
#   adb shell "chmod +x /data/local/tmp/sandbox_init.sh"
#   adb shell "/data/local/tmp/sandbox_init.sh"
#
# WHAT THIS SCRIPT DOES NOT DO:
#   - Does NOT modify /system, /vendor, or /product partitions
#   - Does NOT install APKs or modify app data
#   - Does NOT change SELinux policy persistently
#   - Does NOT modify CPU governor (that's a separate benchmark-only step)
#   - Does NOT touch any file outside /data/local/tmp/halfhex/
#
# ============================================================================

set -e

SANDBOX_ROOT="/data/local/tmp/halfhex"
LOG_FILE="${SANDBOX_ROOT}/sandbox_init.log"

# ── Safety: refuse to run if sandbox already exists with active process ──────
if [ -f "${SANDBOX_ROOT}/.pid" ]; then
    OLD_PID=$(cat "${SANDBOX_ROOT}/.pid")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[ERROR] HalfHex process already running (PID $OLD_PID). Stop it first."
        echo "[ERROR] Run: adb shell 'kill $OLD_PID'"
        exit 1
    fi
fi

# ── Create directory structure ───────────────────────────────────────────────
echo "[SANDBOX] Creating directory tree at ${SANDBOX_ROOT}..."
mkdir -p "${SANDBOX_ROOT}/bin"           # runtime binaries
mkdir -p "${SANDBOX_ROOT}/lib"           # QNN shared libraries
mkdir -p "${SANDBOX_ROOT}/models"        # compiled model .so files
mkdir -p "${SANDBOX_ROOT}/tokenizer"     # tokenizer model files
mkdir -p "${SANDBOX_ROOT}/logs"          # profiling output
mkdir -p "${SANDBOX_ROOT}/profiles"      # CSV profiler dumps
mkdir -p "${SANDBOX_ROOT}/tmp"           # scratch space (cleaned each run)

# ── Write sandbox metadata ───────────────────────────────────────────────────
cat > "${SANDBOX_ROOT}/.sandbox_meta" << 'METAEOF'
{
  "project": "HalfHex",
  "purpose": "QNN-native LLM inference runtime for Qwen3-1.7B",
  "device": "Nothing Phone 3a Pro (A059P)",
  "soc": "Snapdragon 7s Gen 3 (volcano)",
  "sandbox_root": "/data/local/tmp/halfhex",
  "safety": {
    "max_memory_mb": 3072,
    "oom_score_adj": 800,
    "max_temp_celsius": 55,
    "isolation": "all files confined to sandbox_root"
  }
}
METAEOF

# ── Capture baseline device state ────────────────────────────────────────────
echo "[SANDBOX] Capturing baseline device state..."
{
    echo "========================================="
    echo "HalfHex Sandbox Init Log"
    echo "Date: $(date 2>/dev/null || echo 'unknown')"
    echo "========================================="
    echo ""
    echo "--- Memory Baseline ---"
    cat /proc/meminfo | grep -E 'MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree'
    echo ""
    echo "--- Storage Baseline ---"
    df -h /data/local/tmp
    echo ""
    echo "--- Thermal Baseline ---"
    for i in 0 1 2 3 4; do
        temp=$(cat /sys/class/thermal/thermal_zone${i}/temp 2>/dev/null)
        type=$(cat /sys/class/thermal/thermal_zone${i}/type 2>/dev/null)
        if [ -n "$temp" ]; then
            echo "  Zone $i ($type): ${temp}"
        fi
    done
    echo ""
    echo "--- Battery Baseline ---"
    dumpsys battery 2>/dev/null | grep -E 'level|temperature|status' || echo "  (unavailable)"
    echo ""
    echo "--- Existing /data/local/tmp contents (DO NOT TOUCH) ---"
    ls -la /data/local/tmp/ | grep -v halfhex | grep -v sandbox_init
} > "${LOG_FILE}" 2>&1

echo "[SANDBOX] Baseline written to ${LOG_FILE}"

# ── Create the OOM guard wrapper script ──────────────────────────────────────
# This wrapper script launches any binary with memory limits and OOM protection.
# It ensures that if our process uses too much memory, IT gets killed — not
# the user's apps.
cat > "${SANDBOX_ROOT}/bin/oom_guard.sh" << 'OOMEOF'
#!/system/bin/sh
# ============================================================================
# OOM Guard Wrapper
# ============================================================================
# Launches a binary with:
#   1. OOM score adjustment (800 = "kill me before user apps")
#   2. Memory usage monitoring (kills process if RSS exceeds limit)
#   3. Thermal monitoring (pauses/kills if device overheats)
#
# USAGE:
#   ./oom_guard.sh <max_rss_mb> <max_temp_c> <binary> [args...]
#
# EXAMPLE:
#   ./oom_guard.sh 3072 55 ./qnn_benchmark --model libqwen3_model.so
# ============================================================================

MAX_RSS_MB=$1
MAX_TEMP_C=$2
shift 2
BINARY="$@"

if [ -z "$MAX_RSS_MB" ] || [ -z "$MAX_TEMP_C" ] || [ -z "$BINARY" ]; then
    echo "[OOM_GUARD] Usage: oom_guard.sh <max_rss_mb> <max_temp_c> <binary> [args...]"
    exit 1
fi

MAX_RSS_KB=$((MAX_RSS_MB * 1024))
SANDBOX_ROOT="/data/local/tmp/halfhex"

echo "[OOM_GUARD] Starting with limits: RSS=${MAX_RSS_MB}MB, Temp=${MAX_TEMP_C}°C"
echo "[OOM_GUARD] Binary: $BINARY"

# Launch the binary in background
$BINARY &
TARGET_PID=$!
echo $TARGET_PID > "${SANDBOX_ROOT}/.pid"

# Set OOM score: 800 means "kill me aggressively before user apps"
# Android's lmkd uses oom_score_adj: -1000 (never kill) to 1000 (kill first)
# 800 = expendable background process, killed before any user app
echo 800 > /proc/${TARGET_PID}/oom_score_adj 2>/dev/null || \
    echo "[OOM_GUARD] WARNING: Could not set oom_score_adj (non-root?)"

echo "[OOM_GUARD] PID=$TARGET_PID, oom_score_adj=800"

# Monitor loop: check memory + thermal every 2 seconds
while kill -0 $TARGET_PID 2>/dev/null; do
    # Check RSS
    RSS_KB=$(cat /proc/${TARGET_PID}/status 2>/dev/null | grep VmRSS | awk '{print $2}')
    if [ -n "$RSS_KB" ] && [ "$RSS_KB" -gt "$MAX_RSS_KB" ]; then
        RSS_MB=$((RSS_KB / 1024))
        echo "[OOM_GUARD] KILL: RSS=${RSS_MB}MB exceeds limit of ${MAX_RSS_MB}MB"
        echo "[OOM_GUARD] Sending SIGTERM to PID $TARGET_PID"
        kill -TERM $TARGET_PID 2>/dev/null
        sleep 2
        kill -KILL $TARGET_PID 2>/dev/null
        rm -f "${SANDBOX_ROOT}/.pid"
        exit 137
    fi

    # Check temperature
    TEMP_MILLI=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
    if [ -n "$TEMP_MILLI" ]; then
        TEMP_C=$((TEMP_MILLI / 1000))
        if [ "$TEMP_C" -gt "$MAX_TEMP_C" ]; then
            echo "[OOM_GUARD] KILL: Temperature=${TEMP_C}°C exceeds limit of ${MAX_TEMP_C}°C"
            echo "[OOM_GUARD] Sending SIGTERM to PID $TARGET_PID to protect device"
            kill -TERM $TARGET_PID 2>/dev/null
            sleep 2
            kill -KILL $TARGET_PID 2>/dev/null
            rm -f "${SANDBOX_ROOT}/.pid"
            exit 138
        fi
    fi

    # Check available memory (system-wide)
    MEM_AVAIL_KB=$(cat /proc/meminfo | grep MemAvailable | awk '{print $2}')
    if [ -n "$MEM_AVAIL_KB" ] && [ "$MEM_AVAIL_KB" -lt 1048576 ]; then
        # Less than 1GB available system-wide — emergency stop
        MEM_AVAIL_MB=$((MEM_AVAIL_KB / 1024))
        echo "[OOM_GUARD] EMERGENCY: System memory critically low (${MEM_AVAIL_MB}MB available)"
        echo "[OOM_GUARD] Killing inference to protect device stability"
        kill -TERM $TARGET_PID 2>/dev/null
        sleep 1
        kill -KILL $TARGET_PID 2>/dev/null
        rm -f "${SANDBOX_ROOT}/.pid"
        exit 139
    fi

    sleep 2
done

# Process finished naturally
wait $TARGET_PID
EXIT_CODE=$?
rm -f "${SANDBOX_ROOT}/.pid"
echo "[OOM_GUARD] Process exited with code $EXIT_CODE"
exit $EXIT_CODE
OOMEOF
chmod +x "${SANDBOX_ROOT}/bin/oom_guard.sh"

# ── Create the device health monitor ────────────────────────────────────────
cat > "${SANDBOX_ROOT}/bin/health_monitor.sh" << 'HEALTHEOF'
#!/system/bin/sh
# ============================================================================
# Device Health Monitor
# ============================================================================
# Continuously logs device health metrics to a file.
# Run alongside benchmarks to capture thermal/memory timeline.
#
# USAGE:
#   ./health_monitor.sh [interval_seconds] [output_file]
# ============================================================================

INTERVAL=${1:-5}
OUTPUT=${2:-/data/local/tmp/halfhex/logs/health_timeline.csv}

echo "timestamp_ms,mem_avail_mb,mem_free_mb,cpu_temp_c,battery_temp_c,battery_pct,cpu4_freq_khz,cpu4_max_khz,throttle_pct" > "$OUTPUT"

while true; do
    TS=$(date +%s%3N 2>/dev/null || echo 0)

    MEM_AVAIL=$(cat /proc/meminfo | grep MemAvailable | awk '{print int($2/1024)}')
    MEM_FREE=$(cat /proc/meminfo | grep MemFree | awk '{print int($2/1024)}')

    CPU_TEMP_MILLI=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
    CPU_TEMP=$(echo "$CPU_TEMP_MILLI" | awk '{printf "%.1f", $1/1000}')

    BAT_TEMP=$(dumpsys battery 2>/dev/null | grep temperature | awk '{printf "%.1f", $2/10}' || echo 0)
    BAT_PCT=$(dumpsys battery 2>/dev/null | grep level | awk '{print $2}' || echo 0)

    CPU4_CUR=$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq 2>/dev/null || echo 0)
    CPU4_MAX=$(cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 1)

    if [ "$CPU4_MAX" -gt 0 ] 2>/dev/null; then
        THROTTLE=$(echo "$CPU4_CUR $CPU4_MAX" | awk '{printf "%.1f", 100*$1/$2}')
    else
        THROTTLE="100.0"
    fi

    echo "${TS},${MEM_AVAIL},${MEM_FREE},${CPU_TEMP},${BAT_TEMP},${BAT_PCT},${CPU4_CUR},${CPU4_MAX},${THROTTLE}" >> "$OUTPUT"

    sleep "$INTERVAL"
done
HEALTHEOF
chmod +x "${SANDBOX_ROOT}/bin/health_monitor.sh"

# ── Create cleanup script ───────────────────────────────────────────────────
cat > "${SANDBOX_ROOT}/cleanup.sh" << 'CLEANEOF'
#!/system/bin/sh
# ============================================================================
# HalfHex Sandbox Cleanup
# ============================================================================
# Removes ONLY the HalfHex sandbox directory and its contents.
# Does NOT touch any other file in /data/local/tmp/.
#
# USAGE:
#   adb shell "/data/local/tmp/halfhex/cleanup.sh"
#
# After running, the device is returned to its pre-HalfHex state.
# ============================================================================

SANDBOX_ROOT="/data/local/tmp/halfhex"

# Kill any running HalfHex processes first
if [ -f "${SANDBOX_ROOT}/.pid" ]; then
    PID=$(cat "${SANDBOX_ROOT}/.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[CLEANUP] Stopping running process (PID $PID)..."
        kill -TERM "$PID" 2>/dev/null
        sleep 2
        kill -KILL "$PID" 2>/dev/null
    fi
fi

# Verify we're only removing our sandbox
if [ -f "${SANDBOX_ROOT}/.sandbox_meta" ]; then
    echo "[CLEANUP] Removing HalfHex sandbox at ${SANDBOX_ROOT}..."
    rm -rf "${SANDBOX_ROOT}"
    echo "[CLEANUP] Done. Device restored to pre-HalfHex state."
else
    echo "[CLEANUP] ERROR: ${SANDBOX_ROOT}/.sandbox_meta not found."
    echo "[CLEANUP] Refusing to remove — this may not be a HalfHex sandbox."
    exit 1
fi
CLEANEOF
chmod +x "${SANDBOX_ROOT}/cleanup.sh"

# ── Final report ─────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  HalfHex Sandbox Initialized"
echo "========================================="
echo "  Root:     ${SANDBOX_ROOT}"
echo "  Binaries: ${SANDBOX_ROOT}/bin/"
echo "  Models:   ${SANDBOX_ROOT}/models/"
echo "  Libs:     ${SANDBOX_ROOT}/lib/"
echo "  Logs:     ${SANDBOX_ROOT}/logs/"
echo "  Cleanup:  ${SANDBOX_ROOT}/cleanup.sh"
echo ""
echo "  OOM Guard: ${SANDBOX_ROOT}/bin/oom_guard.sh"
echo "    Usage: oom_guard.sh 3072 55 ./qnn_benchmark [args]"
echo "    - Max RSS: 3072 MB (protects device apps)"
echo "    - Max Temp: 55°C (protects device hardware)"
echo "    - Emergency stop if system RAM < 1GB"
echo ""
echo "  Health Monitor: ${SANDBOX_ROOT}/bin/health_monitor.sh"
echo "    Usage: health_monitor.sh 5 output.csv"
echo ""
du -sh "${SANDBOX_ROOT}" 2>/dev/null
echo "========================================="
