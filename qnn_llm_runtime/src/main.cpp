// main.cpp — CLI benchmark harness
#include "QnnRuntime.h"
#include "Profiler.h"
#include "KVCacheManager.h"
#include "ThermalMonitor.h"
#include "TokenizerWrapper.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ── Test prompts ─────────────────────────────────────────────────────────────
static const char* TEST_PROMPT_SHORT = "Explain the theory of relativity in detail.";
static const char* TEST_PROMPT_LONG_400_TOKENS =
    "Write a comprehensive essay about the history of artificial intelligence, "
    "covering its origins in the 1950s with Alan Turing and the Dartmouth conference, "
    "the AI winters, the rise of expert systems, the neural network revolution, "
    "deep learning breakthroughs with AlexNet and transformers, and the current state "
    "of large language models. Include discussion of key figures like Marvin Minsky, "
    "John McCarthy, Geoffrey Hinton, Yann LeCun, Yoshua Bengio, and Ilya Sutskever. "
    "Discuss ethical considerations, alignment problems, and the future trajectory "
    "of AI research and deployment.";

// ── Qwen3-1.7B model configuration ──────────────────────────────────────────
static const int NUM_LAYERS    = 28;
static const int NUM_KV_HEADS  = 4;    // GQA: Qwen3-1.7B uses 4 KV heads
static const int HEAD_DIM      = 128;
static const int MAX_SEQ_LEN   = 512;

struct BenchmarkResult {
    double tokens_per_sec;
    double time_to_first_token_ms;
    int total_tokens;
};

BenchmarkResult run_decode_loop(
    QnnRuntime& runtime,
    TokenizerWrapper& tokenizer,
    KVCacheManager& kv_cache,
    const char* prompt_text,
    int num_output_tokens)
{
    BenchmarkResult result = {};

    // Tokenize prompt
    std::vector<int32_t> prompt_ids = tokenizer.encode(prompt_text);
    if (prompt_ids.empty()) {
        LOGE("[BENCHMARK] Tokenization failed for prompt");
        return result;
    }

    // Prefill
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    auto logits = runtime.prefill(prompt_ids);
    auto t_prefill_end = std::chrono::high_resolution_clock::now();

    result.time_to_first_token_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();

    kv_cache.reset_seq_len();

    // Decode loop
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    int position = (int)prompt_ids.size();

    for (int i = 0; i < num_output_tokens; i++) {
        // Simple argmax sampling (production would use temperature/top-p)
        int32_t next_token = 0;  // TODO: argmax from logits
        if (!logits.empty()) {
            float max_val = logits[0];
            for (size_t j = 1; j < logits.size(); j++) {
                if (logits[j] > max_val) {
                    max_val = logits[j];
                    next_token = (int32_t)j;
                }
            }
        }

        // Check for EOS
        if (next_token == tokenizer.eos_id()) {
            result.total_tokens = i;
            break;
        }

        KVCacheState kv_state = {
            .k_ptr = kv_cache.get_k_ptr(0, 0),
            .v_ptr = kv_cache.get_v_ptr(0, 0),
            .current_seq_len = kv_cache.get_current_seq_len(),
            .num_layers = NUM_LAYERS
        };

        logits = runtime.decode_step({next_token}, kv_state, position);
        kv_cache.advance_seq_len();
        position++;
        result.total_tokens = i + 1;
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_sec = std::chrono::duration<double>(
        t_decode_end - t_decode_start).count();

    result.tokens_per_sec = (decode_sec > 0) ? result.total_tokens / decode_sec : 0;
    return result;
}

int main(int argc, char* argv[]) {
    // Parse args
    std::string model_path = "/data/local/tmp/qnn_runtime/libqwen3_model.so";
    std::string tokenizer_path = "/data/local/tmp/qnn_runtime/tokenizer.model";
    int output_tokens = 100;
    int warmup_runs = 3;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc)
            tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--output-tokens") == 0 && i + 1 < argc)
            output_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup_runs = atoi(argv[++i]);
    }

    LOGI("═══════════════════════════════════════");
    LOGI("QNN HTP Runtime Benchmark");
    LOGI("Target: Qwen3-1.7B INT4 on Hexagon V73");
    LOGI("═══════════════════════════════════════");

    QnnRuntime runtime;
    ThermalMonitor thermal;
    TokenizerWrapper tokenizer;
    KVCacheManager kv_cache;

    // ── Load tokenizer ──────────────────────
    if (!tokenizer.load(tokenizer_path)) {
        LOGE("Failed to load tokenizer from %s", tokenizer_path.c_str());
        return 1;
    }

    // ── Phase 1: Load model ──────────────────
    thermal.log_thermal_snapshot("pre_load");
    if (!runtime.initialize(model_path)) {
        LOGE("Failed to initialize QNN runtime");
        return 1;
    }
    thermal.log_thermal_snapshot("post_load");

    // ── Allocate KV cache ────────────────────
    if (!kv_cache.allocate(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN)) {
        LOGE("Failed to allocate KV cache");
        return 1;
    }

    // ── Phase 2: Warmup (not counted) ────────
    LOGI("[BENCHMARK] Warming up (%d runs)...", warmup_runs);
    for (int i = 0; i < warmup_runs; i++) {
        run_decode_loop(runtime, tokenizer, kv_cache, TEST_PROMPT_SHORT, 20);
    }

    // ── Phase 3: Thermal soak (wait for steady state)
    LOGI("[BENCHMARK] Thermal soak: running for 60 seconds...");
    auto soak_end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (std::chrono::steady_clock::now() < soak_end) {
        run_decode_loop(runtime, tokenizer, kv_cache, TEST_PROMPT_SHORT, 10);
    }
    thermal.log_thermal_snapshot("post_soak");

    // ── Phase 4: Actual benchmark ────────────
    g_profiler.reset();
    LOGI("[BENCHMARK] Starting timed benchmark...");

    // Test 1: Short prompt decode speed
    {
        auto result = run_decode_loop(runtime, tokenizer, kv_cache,
                                      TEST_PROMPT_SHORT, output_tokens);
        LOGI("[RESULT] Short prompt | %d tokens | %.2f tok/s | ttft=%.1fms",
             result.total_tokens, result.tokens_per_sec, result.time_to_first_token_ms);
    }

    // Test 2: Long context (near your max)
    {
        auto result = run_decode_loop(runtime, tokenizer, kv_cache,
                                      TEST_PROMPT_LONG_400_TOKENS, output_tokens);
        LOGI("[RESULT] Long context | %d tokens | %.2f tok/s | ttft=%.1fms",
             result.total_tokens, result.tokens_per_sec, result.time_to_first_token_ms);
    }

    // Test 3: Sustained 120-second throughput
    {
        int total_tokens = 0;
        auto start = std::chrono::steady_clock::now();
        auto end   = start + std::chrono::seconds(120);
        while (std::chrono::steady_clock::now() < end) {
            auto result = run_decode_loop(runtime, tokenizer, kv_cache,
                                          TEST_PROMPT_SHORT, 50);
            total_tokens += result.total_tokens;
            if (thermal.is_throttling()) {
                LOGW("[RESULT] Throttle event at token %d", total_tokens);
            }
        }
        double sustained_tps = total_tokens / 120.0;
        LOGI("[RESULT] SUSTAINED 120s: %.2f tok/s (total: %d tokens)",
             sustained_tps, total_tokens);
    }

    // ── Phase 5: Dump full profiler report ───
    g_profiler.dump_stats("final");
    g_profiler.print_layer_breakdown();
    g_profiler.write_to_file();

    thermal.log_thermal_snapshot("benchmark_complete");

    // Cleanup
    kv_cache.release();
    runtime.shutdown();

    return 0;
}
