# ============================================================================
# CMakeLists.txt — HalfHex QNN LLM Runtime Build Configuration
# ============================================================================
#
# OVERVIEW:
#   Cross-compiles the QNN inference runtime for Android aarch64 (ARM64).
#   Produces two targets:
#     1. qnn_benchmark — CLI binary for on-device benchmarking
#     2. qnn_inference_jni — Shared library (.so) for Android app integration
#
# PREREQUISITES:
#   - Android NDK r26b (exact — later versions break QNN integration)
#   - Qualcomm AI Engine Direct SDK (QNN SDK) >= v2.20
#   - CMake >= 3.18
#
# BUILD COMMANDS:
#
#   # Set environment variables:
#   export ANDROID_NDK=/path/to/ndk/r26b
#   export QNN_SDK=/path/to/qnn-sdk
#
#   # Configure:
#   cmake -B build-android \
#     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_PLATFORM=android-33 \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DQNN_SDK_ROOT=$QNN_SDK \
#     -DPROFILING_ENABLED=ON \
#     -S qnn_llm_runtime
#
#   # Build:
#   cmake --build build-android --config Release -j$(nproc)
#
#   # Output: build-android/qnn_benchmark
#
# ============================================================================

cmake_minimum_required(VERSION 3.18)
project(halfhex_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── QNN SDK Path ─────────────────────────────────────────────────────────────
# Must be provided via -DQNN_SDK_ROOT=/path/to/qnn-sdk
if(NOT DEFINED QNN_SDK_ROOT)
    message(WARNING "QNN_SDK_ROOT not defined. QNN headers will not be available. "
                    "Set via -DQNN_SDK_ROOT=/path/to/qnn-sdk")
endif()

# ── Build Options ────────────────────────────────────────────────────────────
option(PROFILING_ENABLED "Enable microsecond profiling macros (adds ~48MB ring buffer)" ON)
option(BUILD_JNI "Build JNI shared library for Android app integration" OFF)

# ── Include Directories ──────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)
if(DEFINED QNN_SDK_ROOT)
    include_directories(${QNN_SDK_ROOT}/include)
    include_directories(${QNN_SDK_ROOT}/include/QNN)
endif()

# ── Compile Definitions ──────────────────────────────────────────────────────
if(PROFILING_ENABLED)
    add_compile_definitions(PROFILING_ENABLED)
    message(STATUS "Profiling: ENABLED (compile with -DPROFILING_ENABLED=OFF for release)")
endif()

# ── Compiler Warnings ────────────────────────────────────────────────────────
# Treat warnings seriously — memory bugs in LLM inference are silent killers.
add_compile_options(
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter      # QNN callback signatures have unused params
    -Werror=return-type        # Missing return is always a bug
    -Werror=uninitialized      # Uninitialized reads cause silent corruption
)

# ── Source Files ─────────────────────────────────────────────────────────────
set(RUNTIME_SOURCES
    src/Profiler.cpp
    src/MemoryGuard.cpp
    src/QnnRuntime.cpp
    src/KVCacheManager.cpp
    src/ThermalMonitor.cpp
    src/TokenizerWrapper.cpp
)

# ── Target 1: CLI Benchmark Binary ───────────────────────────────────────────
add_executable(qnn_benchmark
    src/main.cpp
    ${RUNTIME_SOURCES}
)

target_link_libraries(qnn_benchmark PRIVATE
    dl          # dlopen/dlsym for loading QNN .so at runtime
    log         # Android logging (liblog)
)

# ── Target 2: JNI Shared Library (optional) ──────────────────────────────────
if(BUILD_JNI)
    add_library(qnn_inference_jni SHARED
        jni/InferenceJNI.cpp
        ${RUNTIME_SOURCES}
    )

    target_link_libraries(qnn_inference_jni PRIVATE
        dl
        log
    )
endif()

# ── Android-Specific Optimization ────────────────────────────────────────────
if(ANDROID)
    # ARM NEON is mandatory on arm64-v8a — no flag needed, but be explicit.
    # -O3 for maximum loop unrolling and vectorization.
    # -ffast-math for HTP tensor prep (not for accumulation — HTP handles that).
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -ffast-math -DNDEBUG")

    # Strip debug symbols in release (saves ~2MB on device).
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

# ── Target 3: Test Suite ─────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests, accuracy tests, and performance regression tests" ON)

if(BUILD_TESTS)
    # Test sources — each file contains auto-registering TEST() macros.
    set(TEST_SOURCES
        tests/test_main.cpp
        tests/test_profiler.cpp
        tests/test_memory_guard.cpp
        tests/test_kv_cache.cpp
        tests/test_thermal_monitor.cpp
        tests/test_accuracy.cpp
        tests/test_perf_regression.cpp
    )

    # Runtime sources needed by tests (no QnnRuntime or TokenizerWrapper —
    # those depend on QNN SDK / SentencePiece which may not be available).
    set(TESTABLE_SOURCES
        src/Profiler.cpp
        src/MemoryGuard.cpp
        src/KVCacheManager.cpp
        src/ThermalMonitor.cpp
    )

    add_executable(halfhex_tests ${TEST_SOURCES} ${TESTABLE_SOURCES})

    target_include_directories(halfhex_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/tests
    )

    # On Android, link log library. On host, no extra libs needed.
    if(ANDROID)
        target_link_libraries(halfhex_tests PRIVATE log)
    endif()

    # Always need pthreads for std::thread in tests
    find_package(Threads REQUIRED)
    target_link_libraries(halfhex_tests PRIVATE Threads::Threads)

    # Enable CTest integration
    enable_testing()
    add_test(NAME halfhex_unit_tests COMMAND halfhex_tests)

    message(STATUS "  Tests:           ON (halfhex_tests)")
endif()

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS qnn_benchmark DESTINATION bin)
if(BUILD_JNI)
    install(TARGETS qnn_inference_jni DESTINATION lib)
endif()

# ── Print Configuration Summary ──────────────────────────────────────────────
message(STATUS "============================================================")
message(STATUS "HalfHex Runtime Configuration:")
message(STATUS "  C++ Standard:    C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  Profiling:       ${PROFILING_ENABLED}")
message(STATUS "  JNI Library:     ${BUILD_JNI}")
message(STATUS "  QNN SDK Root:    ${QNN_SDK_ROOT}")
if(ANDROID)
    message(STATUS "  Android ABI:     ${ANDROID_ABI}")
    message(STATUS "  Android Platform:${ANDROID_PLATFORM}")
endif()
message(STATUS "============================================================")
