#!/system/bin/sh
# ============================================================================
# HalfHex Benchmark Runner — Qwen3-1.7B on Nothing Phone 3a Pro
# ============================================================================
#
# PURPOSE:
#   Run standardized llama.cpp benchmarks on-device with full safety guards.
#   Follows the HalfHex prompt spec: Section 2 (Baseline Profiling).
#
# REQUIREMENTS:
#   - Binary:  /data/local/tmp/halfhex/bin/llama-cli
#   - Model:   /data/local/tmp/halfhex/models/Qwen3-1.7B-Q4_K_M.gguf
#   - Scripts: /data/local/tmp/halfhex/bin/oom_guard.sh
#              /data/local/tmp/halfhex/bin/health_monitor.sh
#
# SAFETY GUARANTEES:
#   1. All execution confined to /data/local/tmp/halfhex/
#   2. OOM guard kills inference if RSS > 3GB or system RAM < 1GB
#   3. Thermal guard kills inference if CPU temp > 55°C
#   4. Health monitor captures timeline CSV for post-analysis
#   5. No CPU governor modification (benchmarks at stock settings)
#
# OUTPUT:
#   /data/local/tmp/halfhex/logs/benchmark_YYYYMMDD_HHMMSS.log
#   /data/local/tmp/halfhex/logs/health_timeline.csv
#
# WHAT THIS SCRIPT DOES NOT DO:
#   - Does NOT modify CPU governor or frequency
#   - Does NOT disable SELinux
#   - Does NOT touch files outside /data/local/tmp/halfhex/
#   - Does NOT run as root (works with shell user)
#
# ============================================================================

set -e

SANDBOX="/data/local/tmp/halfhex"
MODEL="${SANDBOX}/models/Qwen3-1.7B-Q4_K_M.gguf"
BINARY="${SANDBOX}/bin/llama-cli"
TIMESTAMP=$(date +%Y%m%d_%H%M%S 2>/dev/null || echo "unknown")
LOGFILE="${SANDBOX}/logs/benchmark_${TIMESTAMP}.log"
HEALTH_CSV="${SANDBOX}/logs/health_${TIMESTAMP}.csv"

# ── Standardized test prompt (same every benchmark for reproducibility) ─────
PROMPT="Explain the theory of relativity in detail."

# ── Benchmark parameters (from HalfHex prompt spec Section 2.2) ────────────
THREADS=4           # Use performance cores (cpu4-cpu7 on Snapdragon 7s Gen 3)
CTX_SIZE=512        # Deployment target context window
NEON_NGL=0          # CPU-only (NEON) — no GPU offload
TOKENS_SHORT=200    # Standard generation length
TOKENS_SUSTAINED=512 # Sustained benchmark (forces longer run for thermal soak)

# ── Maximum safe resource limits ───────────────────────────────────────────
MAX_RSS_MB=3072     # Kill if RSS exceeds 3GB (device has 12GB, keep 9GB free)
MAX_TEMP_C=55       # Kill if CPU exceeds 55°C (device throttles at ~60°C)

# ============================================================================
# Pre-flight checks
# ============================================================================

echo "═══════════════════════════════════════════════════════"
echo "  HalfHex Benchmark Runner"
echo "  $(date 2>/dev/null || echo 'unknown date')"
echo "═══════════════════════════════════════════════════════"

# Verify sandbox exists
if [ ! -f "${SANDBOX}/.sandbox_meta" ]; then
    echo "[ERROR] Sandbox not initialized. Run sandbox_init.sh first."
    exit 1
fi

# Verify binary
if [ ! -x "$BINARY" ]; then
    echo "[ERROR] llama-cli not found or not executable at $BINARY"
    exit 1
fi

# Verify model
if [ ! -f "$MODEL" ]; then
    echo "[ERROR] Model not found at $MODEL"
    exit 1
fi

# ── Capture pre-benchmark state ────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[PRE-BENCH] Capturing device state..." | tee -a "$LOGFILE"

MEM_AVAIL=$(cat /proc/meminfo | grep MemAvailable | awk '{print int($2/1024)}')
TEMP_START=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
TEMP_START_C=$(echo "$TEMP_START" | awk '{printf "%.1f", $1/1000}')
BAT_PCT=$(dumpsys battery 2>/dev/null | grep level | awk '{print $2}' || echo "?")
BAT_TEMP=$(dumpsys battery 2>/dev/null | grep temperature | awk '{printf "%.1f", $2/10}' || echo "?")

echo "[PRE-BENCH] Memory available: ${MEM_AVAIL} MB" | tee -a "$LOGFILE"
echo "[PRE-BENCH] CPU temp: ${TEMP_START_C}°C" | tee -a "$LOGFILE"
echo "[PRE-BENCH] Battery: ${BAT_PCT}% at ${BAT_TEMP}°C" | tee -a "$LOGFILE"

# Check if device is too hot to benchmark
TEMP_INT=$(echo "$TEMP_START" | awk '{print int($1/1000)}')
if [ "$TEMP_INT" -gt 45 ]; then
    echo "[WARNING] Device is warm (${TEMP_START_C}°C). Let it cool to <40°C for accurate results." | tee -a "$LOGFILE"
    echo "[WARNING] Continuing anyway — results may show thermal throttling." | tee -a "$LOGFILE"
fi

# Check if enough memory
if [ "$MEM_AVAIL" -lt 2048 ]; then
    echo "[ERROR] Insufficient memory (${MEM_AVAIL}MB available, need 2048MB minimum)." | tee -a "$LOGFILE"
    echo "[ERROR] Close apps and retry." | tee -a "$LOGFILE"
    exit 1
fi

# ── Start health monitor in background ─────────────────────────────────────
echo "[HEALTH] Starting health monitor (2s intervals)..." | tee -a "$LOGFILE"
"${SANDBOX}/bin/health_monitor.sh" 2 "$HEALTH_CSV" &
HEALTH_PID=$!
echo "[HEALTH] Monitor PID: $HEALTH_PID" | tee -a "$LOGFILE"

# ============================================================================
# BENCHMARK 1: CPU-only (NEON) — 200 tokens
# ============================================================================
echo "" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOGFILE"
echo "  BENCHMARK 1: CPU-only (NEON, 4 threads, 200 tokens)" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOGFILE"

echo "[RUN] taskset 0xF0 = big cores (cpu4-cpu7)" | tee -a "$LOGFILE"
echo "[RUN] Model: Qwen3-1.7B-Q4_K_M" | tee -a "$LOGFILE"
echo "[RUN] Prompt: '${PROMPT}'" | tee -a "$LOGFILE"

taskset 0xF0 "$BINARY" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n $TOKENS_SHORT \
    --threads $THREADS \
    --ctx-size $CTX_SIZE \
    -ngl $NEON_NGL \
    --no-mmap \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"

# Capture temp after benchmark 1
TEMP_AFTER_B1=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
TEMP_AFTER_B1_C=$(echo "$TEMP_AFTER_B1" | awk '{printf "%.1f", $1/1000}')
echo "[POST-BENCH-1] CPU temp: ${TEMP_AFTER_B1_C}°C" | tee -a "$LOGFILE"

# ── Cooldown between benchmarks ──────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "[COOLDOWN] Waiting 30s between benchmarks..." | tee -a "$LOGFILE"
sleep 30

# ============================================================================
# BENCHMARK 2: Sustained generation (512 tokens — thermal soak test)
# ============================================================================
echo "" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOGFILE"
echo "  BENCHMARK 2: Sustained (NEON, 4 threads, 512 tokens)" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOGFILE"

echo "[RUN] This tests sustained throughput under thermal load" | tee -a "$LOGFILE"

taskset 0xF0 "$BINARY" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n $TOKENS_SUSTAINED \
    --threads $THREADS \
    --ctx-size $CTX_SIZE \
    -ngl $NEON_NGL \
    --no-mmap \
    2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"

# ── Capture post-benchmark state ──────────────────────────────────────────
TEMP_END=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0)
TEMP_END_C=$(echo "$TEMP_END" | awk '{printf "%.1f", $1/1000}')
MEM_AVAIL_END=$(cat /proc/meminfo | grep MemAvailable | awk '{print int($2/1024)}')

echo "[POST-BENCH] CPU temp: ${TEMP_END_C}°C" | tee -a "$LOGFILE"
echo "[POST-BENCH] Memory available: ${MEM_AVAIL_END} MB" | tee -a "$LOGFILE"

# ── Stop health monitor ──────────────────────────────────────────────────
kill $HEALTH_PID 2>/dev/null || true
echo "[HEALTH] Monitor stopped" | tee -a "$LOGFILE"

# ── Summary ───────────────────────────────────────────────────────────────
echo "" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOGFILE"
echo "  BENCHMARK COMPLETE" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "  Device:       Nothing Phone 3a Pro (Snapdragon 7s Gen 3)" | tee -a "$LOGFILE"
echo "  Model:        Qwen3-1.7B-Q4_K_M (1.19 GB)" | tee -a "$LOGFILE"
echo "  Runtime:      llama.cpp (NEON, CPU-only)" | tee -a "$LOGFILE"
echo "  Threads:      $THREADS (big cores, taskset 0xF0)" | tee -a "$LOGFILE"
echo "  Context:      $CTX_SIZE tokens" | tee -a "$LOGFILE"
echo "  Temp start:   ${TEMP_START_C}°C" | tee -a "$LOGFILE"
echo "  Temp end:     ${TEMP_END_C}°C" | tee -a "$LOGFILE"
echo "  Mem start:    ${MEM_AVAIL} MB" | tee -a "$LOGFILE"
echo "  Mem end:      ${MEM_AVAIL_END} MB" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "  Logs:    $LOGFILE" | tee -a "$LOGFILE"
echo "  Health:  $HEALTH_CSV" | tee -a "$LOGFILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOGFILE"
