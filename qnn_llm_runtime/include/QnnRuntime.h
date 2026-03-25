// ============================================================================
// QnnRuntime.h — Qualcomm QNN HTP Backend Interface
// ============================================================================
//
// PURPOSE:
//   Wraps the Qualcomm AI Engine Direct SDK (QNN SDK) to execute compiled
//   model graphs on the Hexagon Tensor Processor (HTP). This is the core
//   compute interface — all tensor operations flow through graphExecute().
//
// QNN SDK ARCHITECTURE:
//   Application → QnnRuntime → libQnnHtp.so → Hexagon DSP firmware
//
//   The QNN SDK uses a provider pattern:
//   1. dlopen("libQnnHtp.so") to load the HTP backend
//   2. QnnInterface_getProviders() to get the function table
//   3. Use function pointers for all operations (backend, context, graph)
//   4. Load pre-compiled model .so (output of qnn-model-lib-generator)
//   5. Create graph from model, set I/O tensors, execute
//
// HEXAGON HTP V73 CAPABILITIES (Snapdragon 7s Gen 3):
//   - INT4 dot-product engines (primary acceleration target)
//   - INT8 matrix multiply engines
//   - FP16 compute for precision-sensitive ops
//   - 2MB L2 cache shared across all HVX units
//   - Up to 4 HVX (Hexagon Vector eXtension) threads
//
// INITIALIZATION SEQUENCE:
//   1. Load libQnnHtp.so (backend)
//   2. Get QNN interface providers
//   3. Create backend handle with HTP config
//   4. Create device handle
//   5. Create context
//   6. Load compiled model .so
//   7. Compose graph with model's compose function
//   8. Finalize graph
//   9. Set up I/O tensors
//
// ERROR HANDLING:
//   All QNN API calls return Qnn_ErrorHandle_t. We check every return value
//   and log failures with both the error code and the operation name. Errors
//   during initialization are fatal (return false). Errors during graph
//   execute return empty logits and increment an error counter.
//
// ============================================================================

#pragma once

#include "Profiler.h"
#include "KVCacheManager.h"
#include <string>
#include <vector>
#include <cstdint>

namespace halfhex {

// ── QnnRuntime ──────────────────────────────────────────────────────────────
class QnnRuntime {
public:
    QnnRuntime() = default;
    ~QnnRuntime();

    // Non-copyable (owns dlopen handles).
    QnnRuntime(const QnnRuntime&) = delete;
    QnnRuntime& operator=(const QnnRuntime&) = delete;

    // ── Initialization ───────────────────────────────────────────────────

    // Initialize the QNN backend and load the compiled model.
    //
    // model_path: path to the compiled model .so file on device
    //             (e.g., "/data/local/tmp/halfhex/models/libqwen3_model.so")
    //
    // Returns false if any QNN API call fails. Check logcat for details.
    bool initialize(const std::string& model_path);

    // ── Inference ────────────────────────────────────────────────────────

    // Prefill: process the entire prompt in one forward pass.
    // This produces the first token and populates the KV cache.
    //
    // prompt_ids: tokenized prompt (e.g., [1, 2345, 6789, ...])
    //
    // Returns: logits vector of size vocab_size (151936 for Qwen3-1.7B)
    //          Empty vector on failure.
    std::vector<float> prefill(const std::vector<int32_t>& prompt_ids);

    // Decode: generate one token given the previous token and KV cache.
    // This is the HOT PATH — every microsecond here matters.
    //
    // input_ids:   single-element vector with the previous token ID
    // kv_cache:    pointer to KVCacheManager (for tensor binding)
    // position_id: current position in the sequence
    //
    // Returns: logits vector of size vocab_size
    //          Empty vector on failure.
    std::vector<float> decode_step(
        const std::vector<int32_t>& input_ids,
        KVCacheManager& kv_cache,
        int position_id);

    // ── Lifecycle ────────────────────────────────────────────────────────

    // Release all QNN resources. Called by destructor.
    void shutdown();

    // ── Status ───────────────────────────────────────────────────────────

    // Check if the runtime is initialized and ready for inference.
    bool is_ready() const { return initialized_; }

    // Get the number of failed graph executions since initialization.
    int error_count() const { return error_count_; }

private:
    // Log /proc/self/status memory fields.
    void log_memory_usage(const char* label);

    // Get current time in microseconds (for manual profiling).
    int64_t get_time_us();

    // QNN SDK handles (opaque pointers, managed by QNN backend).
    void* backend_lib_handle_ = nullptr;  // dlopen handle for libQnnHtp.so
    void* model_lib_handle_   = nullptr;  // dlopen handle for compiled model .so
    void* backend_handle_     = nullptr;  // Qnn_BackendHandle_t
    void* device_handle_      = nullptr;  // Qnn_DeviceHandle_t
    void* context_handle_     = nullptr;  // Qnn_ContextHandle_t
    void* graph_handle_       = nullptr;  // Qnn_GraphHandle_t
    void* log_handle_         = nullptr;  // Qnn_LogHandle_t

    // State tracking.
    bool initialized_ = false;
    int  error_count_  = 0;
    std::string model_path_;
};

} // namespace halfhex
