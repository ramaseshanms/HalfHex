// QnnRuntime.cpp — every QNN API call is profiled
#include "QnnRuntime.h"
#include "Profiler.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>

// QNN SDK headers — paths relative to QNN_SDK/include
// These are provided by the Qualcomm AI Engine Direct SDK
// #include "QNN/QnnInterface.h"
// #include "QNN/QnnTypes.h"
// #include "QNN/HTP/QnnHtpDevice.h"
// #include "QNN/HTP/QnnHtpGraph.h"

// Status codes (from QNN SDK)
#define QNN_SUCCESS 0

// Helper macro for logging QNN status
#define LOG_QNN_STATUS(op, status) \
    if (status != QNN_SUCCESS) { \
        LOGE("[QNN] %s failed with status %d", op, (int)status); \
    } else { \
        LOGI("[QNN] %s succeeded", op); \
    }

QnnRuntime::~QnnRuntime() {
    shutdown();
}

bool QnnRuntime::initialize(const std::string& model_path) {
    PROFILE_SCOPE("QnnRuntime::initialize");

    // ── 1. Load QNN HTP backend ──────────────────────────────────────
    {
        PROFILE_SCOPE("load_htp_backend");
        backend_lib_handle_ = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
        if (!backend_lib_handle_) {
            LOGE("Failed to load libQnnHtp.so: %s", dlerror());
            return false;
        }
        LOGI("[PROFILE] libQnnHtp.so loaded successfully");
    }

    // ── 2. Resolve QNN interface function pointers ──────────────────
    {
        PROFILE_SCOPE("resolve_qnn_interface");
        // In production: use QnnInterface_getProviders() to get the full interface table
        // The QNN SDK provides a function table via the shared library
        typedef int (*QnnInterface_getProvidersFn_t)(
            const void** providerList, uint32_t* numProviders);

        auto getProviders = (QnnInterface_getProvidersFn_t)dlsym(
            backend_lib_handle_, "QnnInterface_getProviders");
        if (!getProviders) {
            LOGE("Failed to resolve QnnInterface_getProviders: %s", dlerror());
            return false;
        }
        LOGI("[PROFILE] QNN interface resolved");

        // TODO: Extract full interface table from providers
        // const void* providerList = nullptr;
        // uint32_t numProviders = 0;
        // getProviders(&providerList, &numProviders);
    }

    // ── 3. Create QNN Backend ────────────────────────────────────────
    {
        PROFILE_SCOPE("create_backend");
        // QnnHtpDevice_CustomConfig_t device_config = {};
        // device_config.option = QNN_HTP_DEVICE_CONFIG_OPTION_PERFORMANCE_INFRASTRUCTURE;
        // Set HTP to sustained high performance mode
        // device_config.performanceInfrastructure.powerConfigId = 1;

        // Qnn_BackendConfig_t backend_cfg = QNN_BACKEND_CONFIG_INIT;
        // auto status = qnn_interface_.backendCreate(
        //     log_handle_, nullptr, 0, &backend_handle_);
        // LOG_QNN_STATUS("backendCreate", status);
        LOGI("[PROFILE] Backend created (placeholder — wire up QNN SDK headers)");
    }

    // ── 4. Load Model ────────────────────────────────────────────────
    {
        PROFILE_SCOPE("load_model_from_file");
        // Load pre-compiled .so (the compiled QNN model binary)
        model_lib_handle_ = dlopen(model_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!model_lib_handle_) {
            LOGE("Failed to load model: %s", dlerror());
            return false;
        }
        LOGI("[PROFILE] Model binary loaded successfully from %s", model_path.c_str());
    }

    // ── 5. Create Graph ──────────────────────────────────────────────
    {
        PROFILE_SCOPE("create_graph");
        // QnnHtpGraph_CustomConfig_t graph_config = {};
        // graph_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        // Enable all HTP V73 optimizations
        // graph_config.optimizationOption.type =
        //     QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        // graph_config.optimizationOption.floatValue = 3.0f;  // max opt level

        // auto status = qnn_interface_.graphCreate(
        //     context_handle_, "qwen3_decode", nullptr, 0, &graph_handle_);
        // LOG_QNN_STATUS("graphCreate", status);
        LOGI("[PROFILE] Graph created (placeholder — wire up QNN SDK headers)");
    }

    log_memory_usage("post_initialize");
    return true;
}

// Single token decode — this is the hot path
// Profile every microsecond here
std::vector<float> QnnRuntime::decode_step(
    const std::vector<int32_t>& input_ids,
    const KVCacheState& kv_state,
    int position_id)
{
    PROFILE_SCOPE("decode_step_total");

    // ── Input preparation ────────────────────────────────────────────
    {
        PROFILE_SCOPE("decode_input_prep");
        set_tensor("input_ids", input_ids);
        set_tensor("position_ids", {position_id});
        set_kv_cache_tensors(kv_state);  // pre-pinned memory — no copy
    }

    // ── HTP Graph Execute ────────────────────────────────────────────
    {
        PROFILE_SCOPE("htp_graph_execute");  // ← this is your core metric
        // auto status = qnn_interface_.graphExecute(
        //     graph_handle_,
        //     input_tensors_.data(), input_tensors_.size(),
        //     output_tensors_.data(), output_tensors_.size(),
        //     nullptr, nullptr
        // );
        // if (status != QNN_SUCCESS) {
        //     LOGE("[PROFILE] graphExecute failed: %d", status);
        //     return {};
        // }
    }

    // ── Output extraction ────────────────────────────────────────────
    {
        PROFILE_SCOPE("decode_output_extract");
        return extract_logits();
    }
}

// Prefill — process full prompt in one graph execute
// Different profile target: time-to-first-token
std::vector<float> QnnRuntime::prefill(
    const std::vector<int32_t>& prompt_ids)
{
    PROFILE_SCOPE("prefill_total");
    LOGI("[PROFILE] Prefill: %zu tokens", prompt_ids.size());

    int64_t t0 = get_time_us();

    {
        PROFILE_SCOPE("prefill_input_prep");
        set_tensor("input_ids", prompt_ids);
        // attention_mask: all 1s for prompt
        std::vector<int32_t> mask(prompt_ids.size(), 1);
        set_tensor("attention_mask", mask);
    }

    {
        PROFILE_SCOPE("prefill_htp_execute");
        // auto status = qnn_interface_.graphExecute(
        //     graph_handle_,
        //     input_tensors_.data(), input_tensors_.size(),
        //     output_tensors_.data(), output_tensors_.size(),
        //     nullptr, nullptr
        // );
        // LOG_QNN_STATUS("prefill_execute", status);
    }

    int64_t t1 = get_time_us();
    double prefill_ms = (t1 - t0) / 1000.0;
    double prefill_tps = prompt_ids.size() / (prefill_ms / 1000.0);
    LOGI("[PROFILE] Prefill: %.1fms | %.1f tokens/sec", prefill_ms, prefill_tps);

    return extract_logits();
}

void QnnRuntime::shutdown() {
    if (model_lib_handle_) {
        dlclose(model_lib_handle_);
        model_lib_handle_ = nullptr;
    }
    if (backend_lib_handle_) {
        dlclose(backend_lib_handle_);
        backend_lib_handle_ = nullptr;
    }
    // TODO: Properly free QNN handles via qnn_interface_
    LOGI("[PROFILE] QnnRuntime shutdown complete");
}

void QnnRuntime::log_memory_usage(const std::string& label) {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0 ||
            strncmp(line, "VmPeak:", 7) == 0) {
            LOGI("[PROFILE][MEM][%s] %s", label.c_str(), line);
        }
    }
    fclose(f);
}

int64_t QnnRuntime::get_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

void QnnRuntime::set_tensor(const std::string& name, const std::vector<int32_t>& data) {
    // TODO: Wire up to QNN tensor API
    // In production: use Qnn_Tensor_t and qnn_interface_ to set input tensor data
    (void)name;
    (void)data;
}

void QnnRuntime::set_kv_cache_tensors(const KVCacheState& kv_state) {
    // TODO: Wire up KV cache pointers to QNN graph inputs
    // These should be zero-copy — just pass the pointer from KVCacheManager
    (void)kv_state;
}

std::vector<float> QnnRuntime::extract_logits() {
    // TODO: Extract logits from QNN output tensors
    // In production: read from output_tensors_ and return vocabulary logits
    return {};
}
