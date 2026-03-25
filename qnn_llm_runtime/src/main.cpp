// ============================================================================
// main.cpp — HalfHex Benchmark Harness
// ============================================================================
//
// PURPOSE:
//   CLI benchmark binary that measures QNN HTP inference performance on the
//   Nothing Phone 3a Pro. Follows a strict measurement protocol:
//
//   Phase 1: Initialize runtime + allocate KV cache
//   Phase 2: Warmup (3 runs, not counted — JIT/cache warming)
//   Phase 3: Thermal soak (60 seconds — reach steady-state temperature)
//   Phase 4: Timed benchmarks (short prompt, long prompt, sustained 120s)
//   Phase 5: Profiler dump + CSV export
//
// MEASUREMENT PROTOCOL:
//   - All benchmarks run AFTER thermal soak (cold numbers are lies)
//   - Thermal throttle events are logged with timestamp for correlation
//   - Memory pressure is checked before each benchmark phase
//   - Emergency stop if system RAM drops below 1GB or temp exceeds 55°C
//   - Results include both burst and sustained metrics
//
// USAGE:
//   ./qnn_benchmark \
//     --model /data/local/tmp/halfhex/models/libqwen3_model.so \
//     --tokenizer /data/local/tmp/halfhex/tokenizer/tokenizer.model \
//     --output-tokens 100 \
//     --warmup 3 \
//     --soak-seconds 60 \
//     --sustained-seconds 120
//
// ============================================================================

#include "QnnRuntime.h"
#include "Profiler.h"
#include "MemoryGuard.h"
#include "KVCacheManager.h"
#include "ThermalMonitor.h"
#include "TokenizerWrapper.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

using namespace halfhex;

// ── Test Prompts ────────────────────────────────────────────────────────────
static const char* TEST_PROMPT_SHORT =
    "Explain the theory of relativity in detail.";

static const char* TEST_PROMPT_LONG =
    "Write a comprehensive essay about the history of artificial intelligence, "
    "covering its origins in the 1950s with Alan Turing and the Dartmouth conference, "
    "the AI winters, the rise of expert systems, the neural network revolution, "
    "deep learning breakthroughs with AlexNet and transformers, and the current state "
    "of large language models. Include discussion of key figures like Marvin Minsky, "
    "John McCarthy, Geoffrey Hinton, Yann LeCun, Yoshua Bengio, and Ilya Sutskever. "
    "Discuss ethical considerations, alignment problems, and the future trajectory "
    "of AI research and deployment across industries.";

// ── Benchmark Result ────────────────────────────────────────────────────────
struct BenchmarkResult {
    double tokens_per_sec        = 0.0;
    double time_to_first_token_ms = 0.0;
    int    total_tokens          = 0;
    int    throttle_events       = 0;
};

// ── CLI Arguments ───────────────────────────────────────────────────────────
struct Args {
    std::string model_path     = "/data/local/tmp/halfhex/models/libqwen3_model.so";
    std::string tokenizer_path = "/data/local/tmp/halfhex/tokenizer/tokenizer.model";
    int output_tokens          = 100;
    int warmup_runs            = 3;
    int soak_seconds           = 60;
    int sustained_seconds      = 120;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            args.model_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc)
            args.tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--output-tokens") == 0 && i + 1 < argc)
            args.output_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            args.warmup_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--soak-seconds") == 0 && i + 1 < argc)
            args.soak_seconds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sustained-seconds") == 0 && i + 1 < argc)
            args.sustained_seconds = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: qnn_benchmark [options]\n");
            printf("  --model PATH          Model .so path\n");
            printf("  --tokenizer PATH      Tokenizer model path\n");
            printf("  --output-tokens N     Tokens to generate per run\n");
            printf("  --warmup N            Warmup runs\n");
            printf("  --soak-seconds N      Thermal soak duration\n");
            printf("  --sustained-seconds N Sustained benchmark duration\n");
            exit(0);
        }
    }
    return args;
}

// ── Decode Loop ─────────────────────────────────────────────────────────────
BenchmarkResult run_decode_loop(
    QnnRuntime& runtime,
    TokenizerWrapper& tokenizer,
    KVCacheManager& kv_cache,
    ThermalMonitor& thermal,
    const char* prompt_text,
    int num_output_tokens)
{
    BenchmarkResult result = {};

    // Tokenize.
    std::vector<int32_t> prompt_ids = tokenizer.encode(prompt_text);
    if (prompt_ids.empty()) {
        LOGE("[BENCH] Tokenization failed");
        return result;
    }

    // Prefill.
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    auto logits = runtime.prefill(prompt_ids);
    auto t_prefill_end = std::chrono::high_resolution_clock::now();

    result.time_to_first_token_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();

    kv_cache.reset_seq_len();

    // Decode loop.
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    int position = (int)prompt_ids.size();

    for (int i = 0; i < num_output_tokens; i++) {
        // Argmax sampling (production would use temperature/top-p/top-k).
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

        if (next_token == tokenizer.eos_id()) {
            result.total_tokens = i;
            break;
        }

        if (kv_cache.is_full()) {
            LOGW("[BENCH] Context window full at token %d", i);
            result.total_tokens = i;
            break;
        }

        logits = runtime.decode_step({next_token}, kv_cache, position);
        kv_cache.advance_seq_len();
        position++;
        result.total_tokens = i + 1;

        // Check for throttle events.
        if (thermal.is_throttling()) {
            result.throttle_events++;
        }

        // Emergency memory check every 50 tokens.
        if (i % 50 == 0 && g_memory_guard.should_emergency_stop()) {
            LOGE("[BENCH] EMERGENCY: System memory critically low — stopping decode");
            break;
        }
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_sec = std::chrono::duration<double>(t_decode_end - t_decode_start).count();
    result.tokens_per_sec = (decode_sec > 0) ? result.total_tokens / decode_sec : 0;

    return result;
}

// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    LOGI("============================================================");
    LOGI("  HalfHex QNN HTP Benchmark");
    LOGI("  Target: Qwen3-1.7B on Snapdragon 7s Gen 3 (Hexagon V73)");
    LOGI("============================================================");

    // ── Initialize MemoryGuard ───────────────────────────────────────────
    if (!g_memory_guard.initialize(QWEN3_1_7B_BUDGET)) {
        LOGE("[MAIN] MemoryGuard initialization failed — insufficient system memory");
        return 1;
    }

    QnnRuntime runtime;
    ThermalMonitor thermal;
    TokenizerWrapper tokenizer;
    KVCacheManager kv_cache;
    ModelConfig model_config;  // Qwen3-1.7B defaults

    // ── Load tokenizer ───────────────────────────────────────────────────
    if (!tokenizer.load(args.tokenizer_path)) {
        LOGE("[MAIN] Failed to load tokenizer from %s", args.tokenizer_path.c_str());
        return 1;
    }

    // ── Phase 1: Initialize runtime ──────────────────────────────────────
    LOGI("[PHASE 1] Initializing QNN runtime...");
    thermal.log_thermal_snapshot("pre_load");

    if (!runtime.initialize(args.model_path)) {
        LOGE("[MAIN] QNN runtime initialization failed");
        return 1;
    }

    // ── Allocate KV cache ────────────────────────────────────────────────
    if (!kv_cache.allocate(model_config)) {
        LOGE("[MAIN] KV cache allocation failed");
        return 1;
    }

    thermal.log_thermal_snapshot("post_load");
    g_memory_guard.log_memory_state("post_init");

    // ── Phase 2: Warmup ──────────────────────────────────────────────────
    LOGI("[PHASE 2] Warmup (%d runs, not counted)...", args.warmup_runs);
    for (int i = 0; i < args.warmup_runs; i++) {
        run_decode_loop(runtime, tokenizer, kv_cache, thermal, TEST_PROMPT_SHORT, 20);
    }

    // ── Phase 3: Thermal soak ────────────────────────────────────────────
    LOGI("[PHASE 3] Thermal soak (%d seconds)...", args.soak_seconds);
    auto soak_end = std::chrono::steady_clock::now() +
                    std::chrono::seconds(args.soak_seconds);
    int soak_iterations = 0;
    while (std::chrono::steady_clock::now() < soak_end) {
        run_decode_loop(runtime, tokenizer, kv_cache, thermal, TEST_PROMPT_SHORT, 10);
        soak_iterations++;

        if (!thermal.is_safe_to_continue()) {
            LOGE("[PHASE 3] Thermal safety limit reached during soak — aborting");
            return 1;
        }
    }
    thermal.log_thermal_snapshot("post_soak");
    LOGI("[PHASE 3] Soak complete after %d iterations", soak_iterations);

    // ── Phase 4: Timed benchmarks ────────────────────────────────────────
    g_profiler.reset();
    LOGI("[PHASE 4] Starting timed benchmarks...");
    LOGI("------------------------------------------------------------");

    // Test 1: Short prompt decode speed.
    {
        auto result = run_decode_loop(runtime, tokenizer, kv_cache, thermal,
                                      TEST_PROMPT_SHORT, args.output_tokens);
        LOGI("[RESULT] Short prompt  | %3d tok | %6.2f tok/s | ttft=%6.1fms | throttle=%d",
             result.total_tokens, result.tokens_per_sec,
             result.time_to_first_token_ms, result.throttle_events);
    }

    // Test 2: Long context decode speed.
    {
        auto result = run_decode_loop(runtime, tokenizer, kv_cache, thermal,
                                      TEST_PROMPT_LONG, args.output_tokens);
        LOGI("[RESULT] Long context  | %3d tok | %6.2f tok/s | ttft=%6.1fms | throttle=%d",
             result.total_tokens, result.tokens_per_sec,
             result.time_to_first_token_ms, result.throttle_events);
    }

    // Test 3: Sustained throughput.
    {
        LOGI("[PHASE 4] Sustained %d-second throughput test...", args.sustained_seconds);
        int total_tokens = 0;
        int total_throttle = 0;
        auto start = std::chrono::steady_clock::now();
        auto end   = start + std::chrono::seconds(args.sustained_seconds);

        while (std::chrono::steady_clock::now() < end) {
            auto result = run_decode_loop(runtime, tokenizer, kv_cache, thermal,
                                          TEST_PROMPT_SHORT, 50);
            total_tokens   += result.total_tokens;
            total_throttle += result.throttle_events;

            if (!thermal.is_safe_to_continue()) {
                LOGE("[PHASE 4] Thermal safety limit — stopping sustained test");
                break;
            }

            if (g_memory_guard.should_emergency_stop()) {
                LOGE("[PHASE 4] Memory emergency — stopping sustained test");
                break;
            }
        }

        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        double sustained_tps = (elapsed > 0) ? total_tokens / elapsed : 0;

        LOGI("[RESULT] SUSTAINED %ds | %d tok | %6.2f tok/s | throttle_events=%d",
             args.sustained_seconds, total_tokens, sustained_tps, total_throttle);
    }

    LOGI("------------------------------------------------------------");

    // ── Phase 5: Profiler dump ───────────────────────────────────────────
    LOGI("[PHASE 5] Dumping profiler data...");
    g_profiler.dump_stats("final");
    g_profiler.print_layer_breakdown();
    std::string csv_path = g_profiler.write_to_file();
    if (!csv_path.empty()) {
        LOGI("[PHASE 5] Profile CSV: %s", csv_path.c_str());
    }

    thermal.log_thermal_snapshot("benchmark_complete");
    g_memory_guard.log_memory_state("final");
    kv_cache.log_cache_stats();

    // ── Cleanup ──────────────────────────────────────────────────────────
    kv_cache.release();
    runtime.shutdown();

    LOGI("============================================================");
    LOGI("  HalfHex benchmark complete.");
    LOGI("============================================================");

    return 0;
}
