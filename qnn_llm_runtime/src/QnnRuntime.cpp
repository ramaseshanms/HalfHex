// ============================================================================
// QnnRuntime.cpp — Qualcomm QNN HTP Backend Implementation
// ============================================================================
//
// CURRENT STATUS: Scaffold with profiling hooks.
// QNN API calls are commented out and marked with TODO because compilation
// requires the QNN SDK headers (not redistributable). The structure and
// profiling are production-ready — once QNN_SDK_ROOT is set and headers
// are available, uncomment the marked sections.
//
// HOW TO WIRE UP QNN SDK:
//   1. Download QNN SDK >= v2.20 from https://qpm.qualcomm.com
//   2. Set QNN_SDK_ROOT in CMakeLists.txt
//   3. Uncomment the #include directives below
//   4. Uncomment the API calls in each function
//   5. Build with: cmake -DQNN_SDK_ROOT=/path/to/sdk ...
//
// ============================================================================

#include "QnnRuntime.h"
#include "MemoryGuard.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <chrono>

// ── QNN SDK Headers (uncomment when SDK is available) ───────────────────────
// #include "QNN/QnnInterface.h"
// #include "QNN/QnnTypes.h"
// #include "QNN/QnnBackend.h"
// #include "QNN/QnnContext.h"
// #include "QNN/QnnGraph.h"
// #include "QNN/QnnDevice.h"
// #include "QNN/QnnLog.h"
// #include "QNN/HTP/QnnHtpDevice.h"
// #include "QNN/HTP/QnnHtpGraph.h"

#define QNN_SUCCESS 0

#define LOG_QNN_STATUS(op, status) \
    do { \
        if ((status) != QNN_SUCCESS) { \
            LOGE("[QNN] %s FAILED with status %d", (op), (int)(status)); \
            error_count_++; \
        } else { \
            LOGI("[QNN] %s succeeded", (op)); \
        } \
    } while (0)

namespace halfhex {

// ── Destructor ──────────────────────────────────────────────────────────────
QnnRuntime::~QnnRuntime() {
    shutdown();
}

// ── initialize ──────────────────────────────────────────────────────────────
bool QnnRuntime::initialize(const std::string& model_path) {
    PROFILE_SCOPE("QnnRuntime::initialize");
    model_path_ = model_path;

    LOGI("[QNN] Initializing HTP backend...");
    LOGI("[QNN] Model: %s", model_path.c_str());

    // ── Step 1: Load HTP backend library ─────────────────────────────────
    {
        PROFILE_SCOPE("qnn_load_htp_backend");
        backend_lib_handle_ = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
        if (!backend_lib_handle_) {
            LOGE("[QNN] Failed to load libQnnHtp.so: %s", dlerror());
            LOGE("[QNN] Ensure libQnnHtp.so is in LD_LIBRARY_PATH or the sandbox lib/ dir");
            return false;
        }
        LOGI("[QNN] libQnnHtp.so loaded successfully");
    }

    // ── Step 2: Resolve QNN interface function table ─────────────────────
    {
        PROFILE_SCOPE("qnn_resolve_interface");

        // The QNN SDK exports QnnInterface_getProviders() which returns
        // a list of QnnInterface_t structs containing all API function pointers.
        //
        // TODO: Uncomment when QNN SDK headers are available:
        //
        // typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn)(
        //     const QnnInterface_t*** providerList,
        //     uint32_t* numProviders);
        //
        // auto getProviders = (QnnInterfaceGetProvidersFn)dlsym(
        //     backend_lib_handle_, "QnnInterface_getProviders");
        // if (!getProviders) {
        //     LOGE("[QNN] Cannot resolve QnnInterface_getProviders: %s", dlerror());
        //     return false;
        // }
        //
        // const QnnInterface_t** providers = nullptr;
        // uint32_t numProviders = 0;
        // auto status = getProviders(&providers, &numProviders);
        // LOG_QNN_STATUS("QnnInterface_getProviders", status);
        // if (status != QNN_SUCCESS || numProviders == 0) return false;
        //
        // qnn_interface_ = providers[0]->QNN_INTERFACE_VER_NAME;

        LOGI("[QNN] Interface resolved (placeholder — wire up QNN SDK)");
    }

    // ── Step 3: Create backend with HTP configuration ────────────────────
    {
        PROFILE_SCOPE("qnn_create_backend");

        // TODO: Uncomment when QNN SDK headers are available:
        //
        // // Enable sustained high performance mode on HTP.
        // QnnHtpDevice_CustomConfig_t deviceConfig = {};
        // deviceConfig.option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
        // deviceConfig.socModel = QNN_SOC_MODEL_SM7550;  // Snapdragon 7s Gen 3
        //
        // const QnnDevice_Config_t* deviceConfigs[] = {&deviceConfig, nullptr};
        //
        // auto status = qnn_interface_.deviceCreate(
        //     log_handle_, deviceConfigs, &device_handle_);
        // LOG_QNN_STATUS("deviceCreate", status);

        LOGI("[QNN] Backend created (placeholder)");
    }

    // ── Step 4: Load compiled model .so ──────────────────────────────────
    {
        PROFILE_SCOPE("qnn_load_model");
        model_lib_handle_ = dlopen(model_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!model_lib_handle_) {
            LOGE("[QNN] Failed to load model: %s", dlerror());
            LOGE("[QNN] Ensure the model .so was compiled with qnn-model-lib-generator");
            return false;
        }
        LOGI("[QNN] Model binary loaded from %s", model_path.c_str());
    }

    // ── Step 5: Create context and compose graph ─────────────────────────
    {
        PROFILE_SCOPE("qnn_create_graph");

        // TODO: Uncomment when QNN SDK headers are available:
        //
        // // Create context.
        // auto status = qnn_interface_.contextCreate(
        //     backend_handle_, device_handle_, nullptr, 0, &context_handle_);
        // LOG_QNN_STATUS("contextCreate", status);
        //
        // // Get the model's compose function from the .so.
        // typedef Qnn_ErrorHandle_t (*ComposeGraphsFn)(
        //     Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE,
        //     Qnn_ContextHandle_t, const QnnContext_Config_t**,
        //     const char*, uint32_t*, const QnnGraph_Config_t**,
        //     Qnn_ProfileHandle_t*, uint32_t);
        //
        // auto composeGraphs = (ComposeGraphsFn)dlsym(
        //     model_lib_handle_, "QnnModel_composeGraphs");
        // if (!composeGraphs) {
        //     LOGE("[QNN] Model .so missing QnnModel_composeGraphs symbol");
        //     return false;
        // }
        //
        // // Enable max HTP optimization level.
        // QnnHtpGraph_CustomConfig_t graphConfig = {};
        // graphConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
        // graphConfig.optimizationOption.type =
        //     QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
        // graphConfig.optimizationOption.floatValue = 3.0f;
        //
        // // Compose and finalize.
        // status = composeGraphs(backend_handle_, qnn_interface_,
        //     context_handle_, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
        // LOG_QNN_STATUS("composeGraphs", status);
        //
        // status = qnn_interface_.graphFinalize(graph_handle_, nullptr, nullptr);
        // LOG_QNN_STATUS("graphFinalize", status);

        LOGI("[QNN] Graph created (placeholder)");
    }

    initialized_ = true;
    log_memory_usage("post_init");

    LOGI("[QNN] Initialization complete. Ready for inference.");
    return true;
}

// ── prefill ─────────────────────────────────────────────────────────────────
std::vector<float> QnnRuntime::prefill(const std::vector<int32_t>& prompt_ids) {
    PROFILE_SCOPE("prefill_total");

    if (!initialized_) {
        LOGE("[QNN] prefill() called before initialization");
        return {};
    }

    LOGI("[QNN] Prefill: %zu tokens", prompt_ids.size());
    int64_t t0 = get_time_us();

    // ── Input preparation ────────────────────────────────────────────────
    {
        PROFILE_SCOPE("prefill_input_prep");
        // TODO: Bind input_ids tensor with prompt_ids data.
        // TODO: Bind attention_mask tensor (all 1s for prompt).
        // TODO: Bind position_ids tensor ([0, 1, 2, ..., N-1]).
    }

    // ── HTP Graph Execute ────────────────────────────────────────────────
    {
        PROFILE_SCOPE("prefill_htp_execute");

        // Check thermal safety before expensive compute.
        // (ThermalMonitor check happens in the benchmark loop, not here,
        //  to keep the prefill path minimal.)

        // TODO: Uncomment when QNN SDK headers are available:
        //
        // auto status = qnn_interface_.graphExecute(
        //     graph_handle_,
        //     input_tensors_.data(), input_tensors_.size(),
        //     output_tensors_.data(), output_tensors_.size(),
        //     nullptr, nullptr);
        // if (status != QNN_SUCCESS) {
        //     LOGE("[QNN] prefill graphExecute failed: %d", status);
        //     error_count_++;
        //     return {};
        // }
    }

    int64_t t1 = get_time_us();
    double prefill_ms  = (t1 - t0) / 1000.0;
    double prefill_tps = (prefill_ms > 0)
        ? prompt_ids.size() / (prefill_ms / 1000.0)
        : 0;

    LOGI("[QNN] Prefill: %.1fms | %.1f tokens/sec", prefill_ms, prefill_tps);

    // TODO: Extract logits from output tensor and return.
    return {};
}

// ── decode_step ─────────────────────────────────────────────────────────────
std::vector<float> QnnRuntime::decode_step(
    const std::vector<int32_t>& input_ids,
    KVCacheManager& kv_cache,
    int position_id)
{
    PROFILE_SCOPE("decode_step_total");

    if (!initialized_) {
        LOGE("[QNN] decode_step() called before initialization");
        return {};
    }

    // ── Input preparation ────────────────────────────────────────────────
    {
        PROFILE_SCOPE("decode_input_prep");
        // TODO: Bind single-token input_ids tensor.
        // TODO: Bind position_ids tensor ([position_id]).
        // TODO: Bind KV cache tensors from kv_cache (zero-copy pointers).
        //
        // CRITICAL: KV cache binding MUST be zero-copy.
        // Use kv_cache.get_k_layer_ptr(layer) and get_v_layer_ptr(layer)
        // directly as QNN tensor data pointers. No memcpy allowed here.
    }

    // ── HTP Graph Execute (HOT PATH) ────────────────────────────────────
    {
        PROFILE_SCOPE("htp_graph_execute");

        // TODO: Uncomment when QNN SDK headers are available:
        //
        // auto status = qnn_interface_.graphExecute(
        //     graph_handle_,
        //     input_tensors_.data(), input_tensors_.size(),
        //     output_tensors_.data(), output_tensors_.size(),
        //     nullptr, nullptr);
        // if (status != QNN_SUCCESS) {
        //     LOGE("[QNN] decode graphExecute failed: %d", status);
        //     error_count_++;
        //     return {};
        // }
    }

    // ── Output extraction ────────────────────────────────────────────────
    {
        PROFILE_SCOPE("decode_output_extract");
        // TODO: Extract logits from output tensor.
        // The output is a float32 tensor of shape [1, vocab_size].
    }

    return {};
}

// ── shutdown ────────────────────────────────────────────────────────────────
void QnnRuntime::shutdown() {
    PROFILE_SCOPE("QnnRuntime::shutdown");

    // TODO: Free QNN handles in reverse order:
    // 1. graphFree(graph_handle_)
    // 2. contextFree(context_handle_)
    // 3. deviceFree(device_handle_)
    // 4. backendFree(backend_handle_)
    // 5. logFree(log_handle_)

    if (model_lib_handle_) {
        dlclose(model_lib_handle_);
        model_lib_handle_ = nullptr;
    }
    if (backend_lib_handle_) {
        dlclose(backend_lib_handle_);
        backend_lib_handle_ = nullptr;
    }

    initialized_ = false;
    LOGI("[QNN] Shutdown complete (errors during session: %d)", error_count_);
}

// ── log_memory_usage ────────────────────────────────────────────────────────
void QnnRuntime::log_memory_usage(const char* label) {
    g_memory_guard.log_memory_state(label);
}

// ── get_time_us ─────────────────────────────────────────────────────────────
int64_t QnnRuntime::get_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

} // namespace halfhex
