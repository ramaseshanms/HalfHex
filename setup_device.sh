#!/bin/bash
# Device setup script for Nothing Phone 3a Pro
set -e

echo "═══════════════════════════════════════"
echo "Device Setup — Nothing Phone 3a Pro"
echo "═══════════════════════════════════════"

# Verify ADB connection
adb devices
adb root

# Temporarily disable SELinux for profiling
adb shell setenforce 0

# Set CPU governor to performance
adb shell "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
adb shell "echo performance > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"

# Create device working directory
adb shell "mkdir -p /data/local/tmp/qnn_runtime"

# Push QNN libraries
adb push $QNN_SDK/lib/aarch64-android/libQnnHtp.so /data/local/tmp/qnn_runtime/
adb push $QNN_SDK/lib/aarch64-android/libQnnHtpV73Skel.so /data/local/tmp/qnn_runtime/

# Verify HTP device is visible
adb push $QNN_SDK/bin/aarch64-android/qnn-net-run /data/local/tmp/qnn_runtime/
adb shell "chmod +x /data/local/tmp/qnn_runtime/qnn-net-run"

echo ""
echo "Device setup complete."
echo "Verify HTP: adb shell '/data/local/tmp/qnn_runtime/qnn-net-run --help'"
echo "═══════════════════════════════════════"
