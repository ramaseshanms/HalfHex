// InferenceJNI.cpp — Android JNI bridge for the QNN LLM runtime
#include <jni.h>
#include <string>
#include <vector>

#include "QnnRuntime.h"
#include "Profiler.h"
#include "KVCacheManager.h"
#include "ThermalMonitor.h"
#include "TokenizerWrapper.h"

// Qwen3-1.7B configuration
static const int NUM_LAYERS   = 28;
static const int NUM_KV_HEADS = 4;
static const int HEAD_DIM     = 128;
static const int MAX_SEQ_LEN  = 512;

// Global runtime objects — initialized once via JNI
static QnnRuntime*       g_runtime   = nullptr;
static KVCacheManager*   g_kv_cache  = nullptr;
static ThermalMonitor*   g_thermal   = nullptr;
static TokenizerWrapper* g_tokenizer = nullptr;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_qnn_llm_InferenceEngine_nativeInit(
    JNIEnv* env, jobject thiz,
    jstring model_path_j, jstring tokenizer_path_j)
{
    const char* model_path = env->GetStringUTFChars(model_path_j, nullptr);
    const char* tokenizer_path = env->GetStringUTFChars(tokenizer_path_j, nullptr);

    LOGI("[JNI] Initializing QNN runtime...");
    LOGI("[JNI] Model: %s", model_path);
    LOGI("[JNI] Tokenizer: %s", tokenizer_path);

    // Initialize tokenizer
    g_tokenizer = new TokenizerWrapper();
    if (!g_tokenizer->load(tokenizer_path)) {
        LOGE("[JNI] Failed to load tokenizer");
        env->ReleaseStringUTFChars(model_path_j, model_path);
        env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);
        return JNI_FALSE;
    }

    // Initialize QNN runtime
    g_runtime = new QnnRuntime();
    if (!g_runtime->initialize(model_path)) {
        LOGE("[JNI] Failed to initialize QNN runtime");
        env->ReleaseStringUTFChars(model_path_j, model_path);
        env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);
        return JNI_FALSE;
    }

    // Allocate KV cache
    g_kv_cache = new KVCacheManager();
    if (!g_kv_cache->allocate(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN)) {
        LOGE("[JNI] Failed to allocate KV cache");
        env->ReleaseStringUTFChars(model_path_j, model_path);
        env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);
        return JNI_FALSE;
    }

    g_thermal = new ThermalMonitor();

    env->ReleaseStringUTFChars(model_path_j, model_path);
    env->ReleaseStringUTFChars(tokenizer_path_j, tokenizer_path);

    LOGI("[JNI] Initialization complete");
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_qnn_llm_InferenceEngine_nativeGenerate(
    JNIEnv* env, jobject thiz,
    jstring prompt_j, jint max_tokens)
{
    if (!g_runtime || !g_tokenizer || !g_kv_cache) {
        return env->NewStringUTF("[ERROR] Runtime not initialized");
    }

    const char* prompt = env->GetStringUTFChars(prompt_j, nullptr);
    LOGI("[JNI] Generate: prompt='%.50s...' max_tokens=%d", prompt, max_tokens);

    // Tokenize
    std::vector<int32_t> prompt_ids = g_tokenizer->encode(prompt);
    env->ReleaseStringUTFChars(prompt_j, prompt);

    if (prompt_ids.empty()) {
        return env->NewStringUTF("[ERROR] Tokenization failed");
    }

    // Prefill
    g_kv_cache->reset_seq_len();
    auto logits = g_runtime->prefill(prompt_ids);

    // Decode loop
    std::string output;
    int position = (int)prompt_ids.size();

    for (int i = 0; i < max_tokens; i++) {
        // Argmax sampling
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

        output += g_tokenizer->decode_token(next_token);

        KVCacheState kv_state = {
            .k_ptr = g_kv_cache->get_k_ptr(0, 0),
            .v_ptr = g_kv_cache->get_v_ptr(0, 0),
            .current_seq_len = g_kv_cache->get_current_seq_len(),
            .num_layers = NUM_LAYERS
        };

        logits = g_runtime->decode_step({next_token}, kv_state, position);
        g_kv_cache->advance_seq_len();
        position++;
    }

    return env->NewStringUTF(output.c_str());
}

JNIEXPORT jfloat JNICALL
Java_com_qnn_llm_InferenceEngine_nativeGetTemperature(
    JNIEnv* env, jobject thiz)
{
    if (!g_thermal) return -1.0f;
    return g_thermal->get_cpu_temp();
}

JNIEXPORT jboolean JNICALL
Java_com_qnn_llm_InferenceEngine_nativeIsThrottling(
    JNIEnv* env, jobject thiz)
{
    if (!g_thermal) return JNI_FALSE;
    return g_thermal->is_throttling() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_qnn_llm_InferenceEngine_nativeShutdown(
    JNIEnv* env, jobject thiz)
{
    LOGI("[JNI] Shutting down...");

    if (g_runtime) { g_runtime->shutdown(); delete g_runtime; g_runtime = nullptr; }
    if (g_kv_cache) { g_kv_cache->release(); delete g_kv_cache; g_kv_cache = nullptr; }
    if (g_thermal) { delete g_thermal; g_thermal = nullptr; }
    if (g_tokenizer) { delete g_tokenizer; g_tokenizer = nullptr; }

    LOGI("[JNI] Shutdown complete");
}

}  // extern "C"
