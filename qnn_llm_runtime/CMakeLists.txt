cmake_minimum_required(VERSION 3.18)
project(qnn_llm_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── QNN SDK path (must be set via -DQNN_SDK_ROOT=...) ────────────────────────
if(NOT DEFINED QNN_SDK_ROOT)
    message(FATAL_ERROR "QNN_SDK_ROOT not defined. Set via -DQNN_SDK_ROOT=/path/to/qnn-sdk")
endif()

# ── Build options ─────────────────────────────────────────────────────────────
option(PROFILING_ENABLED "Enable profiling macros" ON)
option(BUILD_JNI "Build JNI shared library for Android app integration" OFF)

# ── Include directories ──────────────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${QNN_SDK_ROOT}/include
    ${QNN_SDK_ROOT}/include/QNN
)

# ── Compile definitions ──────────────────────────────────────────────────────
if(PROFILING_ENABLED)
    add_compile_definitions(PROFILING_ENABLED)
endif()

# ── Source files ─────────────────────────────────────────────────────────────
set(RUNTIME_SOURCES
    src/Profiler.cpp
    src/QnnRuntime.cpp
    src/KVCacheManager.cpp
    src/ThermalMonitor.cpp
    src/TokenizerWrapper.cpp
)

# ── CLI benchmark binary ─────────────────────────────────────────────────────
add_executable(qnn_benchmark
    src/main.cpp
    ${RUNTIME_SOURCES}
)

target_link_libraries(qnn_benchmark
    dl          # for dlopen/dlsym (loading QNN .so at runtime)
    log         # Android logging (liblog)
)

# ── JNI shared library (for Android app) ─────────────────────────────────────
if(BUILD_JNI)
    add_library(qnn_inference_jni SHARED
        jni/InferenceJNI.cpp
        ${RUNTIME_SOURCES}
    )

    target_link_libraries(qnn_inference_jni
        dl
        log
    )
endif()

# ── Android-specific flags ───────────────────────────────────────────────────
if(ANDROID)
    # Enable NEON SIMD
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -O3")

    # Strip debug symbols for release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
endif()

# ── Install targets ──────────────────────────────────────────────────────────
install(TARGETS qnn_benchmark DESTINATION bin)
if(BUILD_JNI)
    install(TARGETS qnn_inference_jni DESTINATION lib)
endif()
