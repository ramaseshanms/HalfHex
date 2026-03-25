// ============================================================================
// InferenceJNI.cpp — Android JNI Bridge for HalfHex Runtime
// ============================================================================
//
// PURPOSE:
//   Exposes the QNN inference runtime to Android applications via JNI
//   (Java Native Interface). This allows a Kotlin/Java Android app to
//   call into the native C++ runtime for model loading, inference, and
//   device monitoring.
//
// JAVA/KOTLIN INTERFACE:
//   package com.halfhex.inference;
//   public class InferenceEngine {
//       public native boolean nativeInit(String modelPath, String tokenizerPath);
//       public native String  nativeGenerate(String prompt, int maxTokens);
//       public native float   nativeGetTemperature();
//       public native boolean nativeIsThrottling();
//       public native String  nativeGetMemoryState();
//       public native void    nativeShutdown();
//   }
//
// THREAD SAFETY:
//   All native methods acquire resources that are NOT thread-safe.
//   The Java caller MUST ensure single-threaded access (e.g., via a
//   HandlerThread or synchronized block). Concurrent calls to
//   nativeGenerate() will cause undefined behavior.
//
// MEMORY MANAGEMENT:
//   JNI string pointers (GetStringUTFChars) are released in every code path
//   including error returns. The JNI bridge owns the runtime objects and
//   cleans them up in nativeShutdown() or when the JNI is unloaded.
//
// ============================================================================

#include <jni.h>
#include <string>
#include <vector>

#include "QnnRuntime.h"
#include "Profiler.h"
#include "MemoryGuard.h"
#include "KVCacheManager.h"
#include "ThermalMonitor.h"
#include "TokenizerWrapper.h"

using namespace halfhex;

// ── Global Runtime Objects ──────────────────────────────────────────────────
// Initialized once via nativeInit(), cleaned up via nativeShutdown().
// Single-instance by design — Android apps use one model at a time.
static QnnRuntime*       g_runtime   = nullptr;
static KVCacheManager*   g_kv_cache  = nullptr;
static ThermalMonitor*   g_thermal   = nullptr;
static TokenizerWrapper* g_tokenizer = nullptr;

extern "C" {

// ── nativeInit ──────────────────────────────────────────────────────────────
JNIEXPORT jboolean JNICALL
Java_com_halfhex_inference_InferenceEngine_nativeInit(
    JNIEnv* env, jobject /* thiz */,
    jstring model_path_j, jstring tokenizer_path_j)
{
    const char* model_path = env->GetStringUTFChars(model_path_j, nullptr);
    const char* tokenizer_path = env->GetStringUTFChars(tokenizer_path_j, nullptr);

    if (!model_path || !tokenizer_path) {
        if (model_path) env->ReleaseStringUTFChars(model_path_j, model_path);
        if (tokenizer_path) env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);
        return JNI_FALSE;
    }

    LOGI("[JNI] Initializing HalfHex runtime...");
    LOGI("[JNI] Model: %s", model_path);
    LOGI("[JNI] Tokenizer: %s", tokenizer_path);

    // Initialize MemoryGuard.
    if (!g_memory_guard.initialize(QWEN3_1_7B_BUDGET)) {
        LOGE("[JNI] MemoryGuard init failed — insufficient memory");
        env->ReleaseStringUTFChars(model_path_j, model_path);
        env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);
        return JNI_FALSE;
    }

    // Initialize tokenizer.
    g_tokenizer = new (std::nothrow) TokenizerWrapper();
    if (!g_tokenizer || !g_tokenizer->load(tokenizer_path)) {
        LOGE("[JNI] Tokenizer init failed");
        delete g_tokenizer; g_tokenizer = nullptr;
        env->ReleaseStringUTFChars(model_path_j, model_path);
        env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);
        return JNI_FALSE;
    }

    // Initialize QNN runtime.
    g_runtime = new (std::nothrow) QnnRuntime();
    if (!g_runtime || !g_runtime->initialize(model_path)) {
        LOGE("[JNI] QNN runtime init failed");
        delete g_runtime;   g_runtime = nullptr;
        delete g_tokenizer; g_tokenizer = nullptr;
        env->ReleaseStringUTFChars(model_path_j, model_path);
        env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);
        return JNI_FALSE;
    }

    // Allocate KV cache.
    g_kv_cache = new (std::nothrow) KVCacheManager();
    ModelConfig config;
    if (!g_kv_cache || !g_kv_cache->allocate(config)) {
        LOGE("[JNI] KV cache allocation failed");
        delete g_kv_cache;  g_kv_cache = nullptr;
        delete g_runtime;   g_runtime = nullptr;
        delete g_tokenizer; g_tokenizer = nullptr;
        env->ReleaseStringUTFChars(model_path_j, model_path);
        env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);
        return JNI_FALSE;
    }

    g_thermal = new (std::nothrow) ThermalMonitor();

    env->ReleaseStringUTFChars(model_path_j, model_path);
    env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);

    LOGI("[JNI] Initialization complete");
    return JNI_TRUE;
}

// ── nativeGenerate ──────────────────────────────────────────────────────────
JNIEXPORT jstring JNICALL
Java_com_halfhex_inference_InferenceEngine_nativeGenerate(
    JNIEnv* env, jobject /* thiz */,
    jstring prompt_j, jint max_tokens)
{
    if (!g_runtime || !g_tokenizer || !g_kv_cache) {
        return env->NewStringUTF("[ERROR] Runtime not initialized. Call nativeInit() first.");
    }

    const char* prompt = env->GetStringUTFChars(prompt_j, nullptr);
    if (!prompt) return env->NewStringUTF("[ERROR] Invalid prompt");

    LOGI("[JNI] Generate: prompt='%.50s...' max_tokens=%d", prompt, max_tokens);

    // Tokenize.
    std::vector<int32_t> prompt_ids = g_tokenizer->encode(prompt);
    env->ReleaseStringUTFChars(prompt_j, prompt);

    if (prompt_ids.empty()) {
        return env->NewStringUTF("[ERROR] Tokenization failed");
    }

    // Check memory before inference.
    if (g_memory_guard.should_emergency_stop()) {
        return env->NewStringUTF("[ERROR] System memory critically low — refusing inference");
    }

    // Prefill.
    g_kv_cache->reset_seq_len();
    auto logits = g_runtime->prefill(prompt_ids);

    // Decode loop.
    std::string output;
    output.reserve(max_tokens * 4);  // Pre-allocate estimate
    int position = (int)prompt_ids.size();

    for (int i = 0; i < max_tokens; i++) {
        // Argmax.
        int32_t next_token = 0;
        if (!logits.empty()) {
            float max_val = logits[0];
            for (size_t j = 1; j < logits.size(); j++) {
                if (logits[j] > max_val) {
                    max_val = logits[j];
                    next_token = (int32_t)j;
                }
            }
        }

        if (next_token == g_tokenizer->eos_id()) break;
        if (g_kv_cache->is_full()) break;

        output += g_tokenizer->decode_token(next_token);

        logits = g_runtime->decode_step({next_token}, *g_kv_cache, position);
        g_kv_cache->advance_seq_len();
        position++;

        // Memory check every 50 tokens.
        if (i % 50 == 0 && g_memory_guard.should_emergency_stop()) {
            output += "\n[STOPPED: System memory low]";
            break;
        }
    }

    return env->NewStringUTF(output.c_str());
}

// ── nativeGetTemperature ────────────────────────────────────────────────────
JNIEXPORT jfloat JNICALL
Java_com_halfhex_inference_InferenceEngine_nativeGetTemperature(
    JNIEnv* /* env */, jobject /* thiz */)
{
    if (!g_thermal) return -1.0f;
    return g_thermal->get_cpu_temp();
}

// ── nativeIsThrottling ──────────────────────────────────────────────────────
JNIEXPORT jboolean JNICALL
Java_com_halfhex_inference_InferenceEngine_nativeIsThrottling(
    JNIEnv* /* env */, jobject /* thiz */)
{
    if (!g_thermal) return JNI_FALSE;
    return g_thermal->is_throttling() ? JNI_TRUE : JNI_FALSE;
}

// ── nativeGetMemoryState ────────────────────────────────────────────────────
JNIEXPORT jstring JNICALL
Java_com_halfhex_inference_InferenceEngine_nativeGetMemoryState(
    JNIEnv* env, jobject /* thiz */)
{
    char buf[512];
    size_t rss   = g_memory_guard.get_current_rss_bytes() / (1024 * 1024);
    size_t total = g_memory_guard.total_allocated() / (1024 * 1024);
    size_t avail = g_memory_guard.get_system_available_bytes() / (1024 * 1024);

    snprintf(buf, sizeof(buf),
             "RSS=%zuMB | Tracked=%zuMB | SysAvail=%zuMB | Pressure=%s",
             rss, total, avail,
             g_memory_guard.is_memory_pressure() ? "YES" : "no");

    return env->NewStringUTF(buf);
}

// ── nativeShutdown ──────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_halfhex_inference_InferenceEngine_nativeShutdown(
    JNIEnv* /* env */, jobject /* thiz */)
{
    LOGI("[JNI] Shutting down...");

    if (g_kv_cache)  { g_kv_cache->release();  delete g_kv_cache;  g_kv_cache = nullptr; }
    if (g_runtime)   { g_runtime->shutdown();   delete g_runtime;   g_runtime = nullptr;  }
    if (g_thermal)   { delete g_thermal;   g_thermal = nullptr;   }
    if (g_tokenizer) { delete g_tokenizer; g_tokenizer = nullptr; }

    LOGI("[JNI] Shutdown complete");
}

} // extern "C"
