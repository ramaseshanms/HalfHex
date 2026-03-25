#!/bin/bash
# benchmarks/run_benchmarks.sh
# Run this script — it handles everything

set -e
DEVICE_DIR=/data/local/tmp/qnn_runtime
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE=./logs/benchmark_${DATE}.log

echo "═══════════════════════════════════════"
echo "QNN HTP Benchmark Runner"
echo "Date: $(date)"
echo "═══════════════════════════════════════"

# Push latest binary
adb push ./build-android/qnn_benchmark $DEVICE_DIR/
adb push ./models/qwen3-1.7b-qnn-int4/libqwen3_model.so $DEVICE_DIR/

# Push QNN libraries
adb push $QNN_SDK/lib/aarch64-android/libQnnHtp.so $DEVICE_DIR/
adb push $QNN_SDK/lib/aarch64-android/libQnnHtpV73Skel.so $DEVICE_DIR/

# Set performance mode
adb shell "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
adb shell "echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"

# Log device state before benchmark
echo "--- Device State ---" | tee -a $LOG_FILE
adb shell cat /sys/class/thermal/thermal_zone0/temp | tee -a $LOG_FILE
adb shell cat /proc/meminfo | grep -E "MemFree|MemAvailable" | tee -a $LOG_FILE

# Run benchmark
echo "--- Benchmark Output ---" | tee -a $LOG_FILE
adb shell "cd $DEVICE_DIR && \
  LD_LIBRARY_PATH=$DEVICE_DIR \
  taskset 0xF0 ./qnn_benchmark \
  --model libqwen3_model.so \
  --tokenizer tokenizer.model \
  --output-tokens 100 \
  --warmup 3 \
  2>&1" | tee -a $LOG_FILE

# Pull profiler CSV off device
adb pull /sdcard/qnn_profile_*.csv ./logs/ 2>/dev/null || true

echo "═══════════════════════════════════════"
echo "Benchmark complete. Log: $LOG_FILE"
echo "═══════════════════════════════════════"
