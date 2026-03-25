// QnnRuntime.h — Core HTP execution interface
#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct KVCacheState {
    void* k_ptr;
    void* v_ptr;
    int current_seq_len;
    int num_layers;
};

struct DecodeResult {
    double tokens_per_sec;
    double time_to_first_token_ms;
    int total_tokens;
};

class QnnRuntime {
public:
    QnnRuntime() = default;
    ~QnnRuntime();

    // Initialize QNN backend and load model
    bool initialize(const std::string& model_path);

    // Single token decode — this is the hot path
    std::vector<float> decode_step(
        const std::vector<int32_t>& input_ids,
        const KVCacheState& kv_state,
        int position_id);

    // Prefill — process full prompt in one graph execute
    std::vector<float> prefill(
        const std::vector<int32_t>& prompt_ids);

    // Cleanup
    void shutdown();

private:
    void log_memory_usage(const std::string& label);
    int64_t get_time_us();

    void set_tensor(const std::string& name, const std::vector<int32_t>& data);
    void set_kv_cache_tensors(const KVCacheState& kv_state);
    std::vector<float> extract_logits();

    // QNN handles
    void* backend_lib_handle_ = nullptr;
    void* model_lib_handle_   = nullptr;
    void* backend_handle_     = nullptr;
    void* context_handle_     = nullptr;
    void* graph_handle_       = nullptr;
    void* log_handle_         = nullptr;

    // QNN interface function pointers
    struct QnnInterface {
        void* backendCreate;
        void* graphCreate;
        void* graphExecute;
        void* contextCreate;
    } qnn_interface_ = {};

    // Tensor state
    struct TensorWrapper {
        std::string name;
        void* data;
        size_t size;
    };
    std::vector<TensorWrapper> input_tensors_;
    std::vector<TensorWrapper> output_tensors_;
};
