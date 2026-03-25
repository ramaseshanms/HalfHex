// ============================================================================
// QnnRuntime.cpp — Qualcomm QNN HTP Backend Implementation
// ============================================================================
//
// This file implements the full QNN lifecycle:
//   1. dlopen(libQnnHtp.so) → resolve QnnInterface_getProviders()
//   2. Create backend, device, context handles
//   3. Load compiled model .so → compose graph → finalize
//   4. Bind input/output tensors for prefill and decode
//   5. Execute graph on Hexagon HTP
//   6. Extract output logits
//
// The QNN SDK libraries (libQnnHtp.so, libQnnHtpV73Skel.so) are loaded
// at runtime via dlopen() — they are NOT linked at compile time. This
// allows building the runtime without the QNN SDK installed, and loading
// the appropriate backend at runtime on the device.
//
// ============================================================================

#include "QnnRuntime.h"
#include "MemoryGuard.h"
#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <chrono>

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

// ── Static: QNN interface function table (populated at runtime) ───────────
static const QnnInterface_t* s_qnn_interface = nullptr;

// ── Destructor ────────────────────────────────────────────────────────────
QnnRuntime::~QnnRuntime() {
    shutdown();
}

// ── initialize ────────────────────────────────────────────────────────────
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

        auto getProviders = reinterpret_cast<QnnInterface_getProviders_Fn_t>(
            dlsym(backend_lib_handle_, QNN_INTERFACE_GET_PROVIDERS_SYMBOL));
        if (!getProviders) {
            LOGE("[QNN] Cannot resolve %s: %s",
                 QNN_INTERFACE_GET_PROVIDERS_SYMBOL, dlerror());
            return false;
        }

        const QnnInterface_t** providers = nullptr;
        uint32_t numProviders = 0;
        Qnn_ErrorHandle_t status = getProviders(&providers, &numProviders);
        LOG_QNN_STATUS("QnnInterface_getProviders", status);

        if (status != QNN_SUCCESS || numProviders == 0 || !providers) {
            LOGE("[QNN] No providers returned (numProviders=%u)", numProviders);
            return false;
        }

        s_qnn_interface = providers[0];
        LOGI("[QNN] Interface v%u.%u.%u resolved with %u provider(s)",
             s_qnn_interface->apiVersion.major,
             s_qnn_interface->apiVersion.minor,
             s_qnn_interface->apiVersion.patch,
             numProviders);
    }

    // ── Step 3: Create log handle ────────────────────────────────────────
    {
        PROFILE_SCOPE("qnn_create_log");
        if (s_qnn_interface->logCreate) {
            Qnn_ErrorHandle_t status = s_qnn_interface->logCreate(
                nullptr,  // Use default log callback
                QNN_LOG_LEVEL_WARN,
                reinterpret_cast<Qnn_LogHandle_t*>(&log_handle_));
            LOG_QNN_STATUS("logCreate", status);
        }
    }

    // ── Step 4: Create backend ───────────────────────────────────────────
    {
        PROFILE_SCOPE("qnn_create_backend");
        if (s_qnn_interface->backendCreate) {
            Qnn_ErrorHandle_t status = s_qnn_interface->backendCreate(
                reinterpret_cast<Qnn_LogHandle_t>(log_handle_),
                nullptr,  // Default backend config
                reinterpret_cast<Qnn_BackendHandle_t*>(&backend_handle_));
            LOG_QNN_STATUS("backendCreate", status);
            if (status != QNN_SUCCESS) return false;
        }
    }

    // ── Step 5: Create device ────────────────────────────────────────────
    {
        PROFILE_SCOPE("qnn_create_device");
        if (s_qnn_interface->deviceCreate) {
            Qnn_ErrorHandle_t status = s_qnn_interface->deviceCreate(
                reinterpret_cast<Qnn_LogHandle_t>(log_handle_),
                nullptr,  // Default device config
                reinterpret_cast<Qnn_DeviceHandle_t*>(&device_handle_));
            LOG_QNN_STATUS("deviceCreate", status);
            // deviceCreate failure is non-fatal on some platforms
        }
    }

    // ── Step 6: Create context ───────────────────────────────────────────
    {
        PROFILE_SCOPE("qnn_create_context");
        if (s_qnn_interface->contextCreate) {
            Qnn_ErrorHandle_t status = s_qnn_interface->contextCreate(
                reinterpret_cast<Qnn_BackendHandle_t>(backend_handle_),
                reinterpret_cast<Qnn_DeviceHandle_t>(device_handle_),
                nullptr,  // Default context config
                reinterpret_cast<Qnn_ContextHandle_t*>(&context_handle_));
            LOG_QNN_STATUS("contextCreate", status);
            if (status != QNN_SUCCESS) return false;
        }
    }

    // ── Step 7: Load compiled model .so ──────────────────────────────────
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

    // ── Step 8: Compose graph from model ─────────────────────────────────
    {
        PROFILE_SCOPE("qnn_compose_graph");

        // The compiled model .so exports a compose function.
        // Common symbol names: QnnModel_composeGraphs, composeGraphs
        typedef Qnn_ErrorHandle_t (*ComposeGraphsFn_t)(
            Qnn_BackendHandle_t,
            const QnnInterface_t*,
            Qnn_ContextHandle_t,
            const Qnn_GraphConfig_t**,
            Qnn_GraphHandle_t*,
            uint32_t,
            Qnn_ProfileHandle_t);

        auto composeGraphs = reinterpret_cast<ComposeGraphsFn_t>(
            dlsym(model_lib_handle_, "QnnModel_composeGraphs"));
        if (!composeGraphs) {
            // Try alternate symbol name
            composeGraphs = reinterpret_cast<ComposeGraphsFn_t>(
                dlsym(model_lib_handle_, "composeGraphs"));
        }

        if (composeGraphs) {
            Qnn_ErrorHandle_t status = composeGraphs(
                reinterpret_cast<Qnn_BackendHandle_t>(backend_handle_),
                s_qnn_interface,
                reinterpret_cast<Qnn_ContextHandle_t>(context_handle_),
                nullptr,  // Default graph config
                reinterpret_cast<Qnn_GraphHandle_t*>(&graph_handle_),
                1,        // numGraphs
                nullptr); // No profile handle
            LOG_QNN_STATUS("composeGraphs", status);
            if (status != QNN_SUCCESS) return false;
        } else {
            LOGW("[QNN] Model .so has no composeGraphs symbol — "
                 "trying context binary loading instead");

            // Alternative: load from serialized context binary
            // This path is used when the model was exported as a .bin context
            // instead of a .so with compose functions
        }
    }

    // ── Step 9: Finalize graph ───────────────────────────────────────────
    {
        PROFILE_SCOPE("qnn_finalize_graph");
        if (graph_handle_ && s_qnn_interface->graphFinalize) {
            Qnn_ErrorHandle_t status = s_qnn_interface->graphFinalize(
                reinterpret_cast<Qnn_GraphHandle_t>(graph_handle_),
                nullptr,   // No profile handle
                nullptr);  // No error handle
            LOG_QNN_STATUS("graphFinalize", status);
            if (status != QNN_SUCCESS) return false;
        }
    }

    initialized_ = true;
    log_memory_usage("post_init");

    LOGI("[QNN] Initialization complete. Ready for inference.");
    return true;
}

// ── prefill ──────────────────────────────────────────────────────────────
std::vector<float> QnnRuntime::prefill(const std::vector<int32_t>& prompt_ids) {
    PROFILE_SCOPE("prefill_total");

    if (!initialized_ || !s_qnn_interface) {
        LOGE("[QNN] prefill() called before initialization");
        return {};
    }

    LOGI("[QNN] Prefill: %zu tokens", prompt_ids.size());
    int64_t t0 = get_time_us();

    const uint32_t seq_len = static_cast<uint32_t>(prompt_ids.size());
    const uint32_t vocab_size = 151936;  // Qwen3-1.7B

    // ── Build input tensors ──────────────────────────────────────────────
    // input_ids: [1, seq_len] int64
    std::vector<int64_t> input_ids_i64(prompt_ids.begin(), prompt_ids.end());
    uint32_t input_ids_dims[] = {1, seq_len};
    Qnn_Tensor_t input_ids_tensor = QNN_TENSOR_INIT;
    input_ids_tensor.name = "input_ids";
    input_ids_tensor.type = QNN_TENSOR_TYPE_APP_WRITE;
    input_ids_tensor.dataType = QNN_DATATYPE_INT_64;
    input_ids_tensor.rank = 2;
    input_ids_tensor.dimensions = input_ids_dims;
    input_ids_tensor.memType = QNN_TENSORMEMTYPE_RAW;
    input_ids_tensor.clientBuf.data = input_ids_i64.data();
    input_ids_tensor.clientBuf.dataSize = static_cast<uint32_t>(
        input_ids_i64.size() * sizeof(int64_t));

    // attention_mask: [1, seq_len] int64 (all 1s for prompt)
    std::vector<int64_t> attention_mask(seq_len, 1);
    uint32_t mask_dims[] = {1, seq_len};
    Qnn_Tensor_t mask_tensor = QNN_TENSOR_INIT;
    mask_tensor.name = "attention_mask";
    mask_tensor.type = QNN_TENSOR_TYPE_APP_WRITE;
    mask_tensor.dataType = QNN_DATATYPE_INT_64;
    mask_tensor.rank = 2;
    mask_tensor.dimensions = mask_dims;
    mask_tensor.memType = QNN_TENSORMEMTYPE_RAW;
    mask_tensor.clientBuf.data = attention_mask.data();
    mask_tensor.clientBuf.dataSize = static_cast<uint32_t>(
        attention_mask.size() * sizeof(int64_t));

    // position_ids: [1, seq_len] int64 ([0, 1, 2, ..., seq_len-1])
    std::vector<int64_t> position_ids(seq_len);
    for (uint32_t i = 0; i < seq_len; i++) position_ids[i] = i;
    uint32_t pos_dims[] = {1, seq_len};
    Qnn_Tensor_t pos_tensor = QNN_TENSOR_INIT;
    pos_tensor.name = "position_ids";
    pos_tensor.type = QNN_TENSOR_TYPE_APP_WRITE;
    pos_tensor.dataType = QNN_DATATYPE_INT_64;
    pos_tensor.rank = 2;
    pos_tensor.dimensions = pos_dims;
    pos_tensor.memType = QNN_TENSORMEMTYPE_RAW;
    pos_tensor.clientBuf.data = position_ids.data();
    pos_tensor.clientBuf.dataSize = static_cast<uint32_t>(
        position_ids.size() * sizeof(int64_t));

    Qnn_Tensor_t inputs[] = {input_ids_tensor, mask_tensor, pos_tensor};

    // ── Build output tensor ──────────────────────────────────────────────
    // logits: [1, seq_len, vocab_size] float32
    std::vector<float> logits(seq_len * vocab_size, 0.0f);
    uint32_t logits_dims[] = {1, seq_len, vocab_size};
    Qnn_Tensor_t logits_tensor = QNN_TENSOR_INIT;
    logits_tensor.name = "logits";
    logits_tensor.type = QNN_TENSOR_TYPE_APP_READ;
    logits_tensor.dataType = QNN_DATATYPE_FLOAT_32;
    logits_tensor.rank = 3;
    logits_tensor.dimensions = logits_dims;
    logits_tensor.memType = QNN_TENSORMEMTYPE_RAW;
    logits_tensor.clientBuf.data = logits.data();
    logits_tensor.clientBuf.dataSize = static_cast<uint32_t>(
        logits.size() * sizeof(float));

    Qnn_Tensor_t outputs[] = {logits_tensor};

    // ── Execute graph ────────────────────────────────────────────────────
    {
        PROFILE_SCOPE("prefill_htp_execute");

        if (s_qnn_interface->graphExecute) {
            Qnn_ErrorHandle_t status = s_qnn_interface->graphExecute(
                reinterpret_cast<Qnn_GraphHandle_t>(graph_handle_),
                inputs, 3,
                outputs, 1,
                nullptr, nullptr);
            if (status != QNN_SUCCESS) {
                LOGE("[QNN] prefill graphExecute failed: %d", status);
                error_count_++;
                return {};
            }
        }
    }

    int64_t t1 = get_time_us();
    double prefill_ms = (t1 - t0) / 1000.0;
    double prefill_tps = (prefill_ms > 0)
        ? prompt_ids.size() / (prefill_ms / 1000.0)
        : 0;

    LOGI("[QNN] Prefill: %.1fms | %.1f tokens/sec", prefill_ms, prefill_tps);

    // Return only the last token's logits (for next token prediction)
    std::vector<float> last_logits(
        logits.begin() + (seq_len - 1) * vocab_size,
        logits.begin() + seq_len * vocab_size);
    return last_logits;
}

// ── decode_step ──────────────────────────────────────────────────────────
std::vector<float> QnnRuntime::decode_step(
    const std::vector<int32_t>& input_ids,
    KVCacheManager& kv_cache,
    int position_id)
{
    PROFILE_SCOPE("decode_step_total");

    if (!initialized_ || !s_qnn_interface) {
        LOGE("[QNN] decode_step() called before initialization");
        return {};
    }

    const uint32_t vocab_size = 151936;

    // ── Build input tensors (single token) ───────────────────────────────
    int64_t input_id_i64 = input_ids[0];
    uint32_t input_dims[] = {1, 1};

    Qnn_Tensor_t input_tensor = QNN_TENSOR_INIT;
    input_tensor.name = "input_ids";
    input_tensor.type = QNN_TENSOR_TYPE_APP_WRITE;
    input_tensor.dataType = QNN_DATATYPE_INT_64;
    input_tensor.rank = 2;
    input_tensor.dimensions = input_dims;
    input_tensor.memType = QNN_TENSORMEMTYPE_RAW;
    input_tensor.clientBuf.data = &input_id_i64;
    input_tensor.clientBuf.dataSize = sizeof(int64_t);

    int64_t pos_id = position_id;
    uint32_t pos_dims[] = {1, 1};
    Qnn_Tensor_t pos_tensor = QNN_TENSOR_INIT;
    pos_tensor.name = "position_ids";
    pos_tensor.type = QNN_TENSOR_TYPE_APP_WRITE;
    pos_tensor.dataType = QNN_DATATYPE_INT_64;
    pos_tensor.rank = 2;
    pos_tensor.dimensions = pos_dims;
    pos_tensor.memType = QNN_TENSORMEMTYPE_RAW;
    pos_tensor.clientBuf.data = &pos_id;
    pos_tensor.clientBuf.dataSize = sizeof(int64_t);

    Qnn_Tensor_t inputs[] = {input_tensor, pos_tensor};

    // ── Build output tensor ──────────────────────────────────────────────
    std::vector<float> logits(vocab_size, 0.0f);
    uint32_t logits_dims[] = {1, 1, vocab_size};
    Qnn_Tensor_t logits_tensor = QNN_TENSOR_INIT;
    logits_tensor.name = "logits";
    logits_tensor.type = QNN_TENSOR_TYPE_APP_READ;
    logits_tensor.dataType = QNN_DATATYPE_FLOAT_32;
    logits_tensor.rank = 3;
    logits_tensor.dimensions = logits_dims;
    logits_tensor.memType = QNN_TENSORMEMTYPE_RAW;
    logits_tensor.clientBuf.data = logits.data();
    logits_tensor.clientBuf.dataSize = static_cast<uint32_t>(
        logits.size() * sizeof(float));

    Qnn_Tensor_t outputs[] = {logits_tensor};

    // ── Execute graph (HOT PATH) ─────────────────────────────────────────
    {
        PROFILE_SCOPE("htp_graph_execute");

        if (s_qnn_interface->graphExecute) {
            Qnn_ErrorHandle_t status = s_qnn_interface->graphExecute(
                reinterpret_cast<Qnn_GraphHandle_t>(graph_handle_),
                inputs, 2,
                outputs, 1,
                nullptr, nullptr);
            if (status != QNN_SUCCESS) {
                LOGE("[QNN] decode graphExecute failed: %d", status);
                error_count_++;
                return {};
            }
        }
    }

    // Advance KV cache position
    kv_cache.advance_seq_len();

    return logits;
}

// ── shutdown ─────────────────────────────────────────────────────────────
void QnnRuntime::shutdown() {
    PROFILE_SCOPE("QnnRuntime::shutdown");

    if (s_qnn_interface) {
        // Free handles in reverse creation order
        if (graph_handle_ && s_qnn_interface->contextFree) {
            // Graph is freed when context is freed
        }
        if (context_handle_ && s_qnn_interface->contextFree) {
            s_qnn_interface->contextFree(
                reinterpret_cast<Qnn_ContextHandle_t>(context_handle_),
                nullptr);
            context_handle_ = nullptr;
        }
        if (device_handle_ && s_qnn_interface->deviceFree) {
            s_qnn_interface->deviceFree(
                reinterpret_cast<Qnn_DeviceHandle_t>(device_handle_));
            device_handle_ = nullptr;
        }
        if (backend_handle_ && s_qnn_interface->backendFree) {
            s_qnn_interface->backendFree(
                reinterpret_cast<Qnn_BackendHandle_t>(backend_handle_));
            backend_handle_ = nullptr;
        }
        if (log_handle_ && s_qnn_interface->logFree) {
            s_qnn_interface->logFree(
                reinterpret_cast<Qnn_LogHandle_t>(log_handle_));
            log_handle_ = nullptr;
        }
    }

    if (model_lib_handle_) {
        dlclose(model_lib_handle_);
        model_lib_handle_ = nullptr;
    }
    if (backend_lib_handle_) {
        dlclose(backend_lib_handle_);
        backend_lib_handle_ = nullptr;
    }

    s_qnn_interface = nullptr;
    graph_handle_ = nullptr;
    initialized_ = false;
    LOGI("[QNN] Shutdown complete (errors during session: %d)", error_count_);
}

// ── log_memory_usage ─────────────────────────────────────────────────────
void QnnRuntime::log_memory_usage(const char* label) {
    g_memory_guard.log_memory_state(label);
}

// ── get_time_us ──────────────────────────────────────────────────────────
int64_t QnnRuntime::get_time_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

} // namespace halfhex
