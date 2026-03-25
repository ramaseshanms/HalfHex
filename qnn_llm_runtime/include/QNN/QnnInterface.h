// ============================================================================
// QNN/QnnInterface.h — QNN Interface Function Pointer Table
// ============================================================================
//
// PURPOSE:
//   Defines the QNN interface structure obtained via QnnInterface_getProviders().
//   This is the central dispatch table for all QNN operations — backend creation,
//   graph composition, tensor management, and execution.
//
//   The actual function pointers are populated at runtime by loading libQnnHtp.so
//   via dlopen() and calling QnnInterface_getProviders().
//
// API REFERENCE:
//   https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/
//
// ============================================================================

#pragma once

#include "QnnTypes.h"

// ── API Version ───────────────────────────────────────────────────────────
typedef struct {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} Qnn_Version_t;

typedef struct {
    Qnn_Version_t coreApiVersion;
    Qnn_Version_t backendApiVersion;
} Qnn_ApiVersion_t;

// ── Function pointer types ────────────────────────────────────────────────

// Backend lifecycle
typedef Qnn_ErrorHandle_t (*Qnn_BackendCreate_Fn_t)(
    Qnn_LogHandle_t logHandle,
    const Qnn_BackendConfig_t** config,
    Qnn_BackendHandle_t* backendHandle);

typedef Qnn_ErrorHandle_t (*Qnn_BackendFree_Fn_t)(
    Qnn_BackendHandle_t backendHandle);

typedef Qnn_ErrorHandle_t (*Qnn_BackendGetApiVersion_Fn_t)(
    Qnn_BackendHandle_t backendHandle,
    const Qnn_ApiVersion_t** apiVersion);

// Device lifecycle
typedef Qnn_ErrorHandle_t (*Qnn_DeviceCreate_Fn_t)(
    Qnn_LogHandle_t logHandle,
    const void* config,
    Qnn_DeviceHandle_t* deviceHandle);

typedef Qnn_ErrorHandle_t (*Qnn_DeviceFree_Fn_t)(
    Qnn_DeviceHandle_t deviceHandle);

// Context lifecycle
typedef Qnn_ErrorHandle_t (*Qnn_ContextCreate_Fn_t)(
    Qnn_BackendHandle_t backendHandle,
    Qnn_DeviceHandle_t deviceHandle,
    const Qnn_ContextConfig_t** config,
    Qnn_ContextHandle_t* contextHandle);

typedef Qnn_ErrorHandle_t (*Qnn_ContextFree_Fn_t)(
    Qnn_ContextHandle_t contextHandle,
    Qnn_ProfileHandle_t profileHandle);

typedef Qnn_ErrorHandle_t (*Qnn_ContextGetBinarySize_Fn_t)(
    Qnn_ContextHandle_t contextHandle,
    uint64_t* binarySize);

typedef Qnn_ErrorHandle_t (*Qnn_ContextGetBinary_Fn_t)(
    Qnn_ContextHandle_t contextHandle,
    void* binaryBuffer,
    uint64_t binaryBufferSize,
    uint64_t* writtenSize);

typedef Qnn_ErrorHandle_t (*Qnn_ContextCreateFromBinary_Fn_t)(
    Qnn_BackendHandle_t backendHandle,
    Qnn_DeviceHandle_t deviceHandle,
    const Qnn_ContextConfig_t** config,
    const void* binaryBuffer,
    uint64_t binaryBufferSize,
    Qnn_ContextHandle_t* contextHandle,
    Qnn_ProfileHandle_t profileHandle);

// Graph lifecycle
typedef Qnn_ErrorHandle_t (*Qnn_GraphCreate_Fn_t)(
    Qnn_ContextHandle_t contextHandle,
    const char* graphName,
    const Qnn_GraphConfig_t** config,
    Qnn_GraphHandle_t* graphHandle);

typedef Qnn_ErrorHandle_t (*Qnn_GraphAddNode_Fn_t)(
    Qnn_GraphHandle_t graphHandle,
    Qnn_OpConfig_t opConfig);

typedef Qnn_ErrorHandle_t (*Qnn_GraphFinalize_Fn_t)(
    Qnn_GraphHandle_t graphHandle,
    Qnn_ProfileHandle_t profileHandle,
    Qnn_ErrorHandle_t* errorHandle);

typedef Qnn_ErrorHandle_t (*Qnn_GraphExecute_Fn_t)(
    Qnn_GraphHandle_t graphHandle,
    const Qnn_Tensor_t* inputs,
    uint32_t numInputs,
    Qnn_Tensor_t* outputs,
    uint32_t numOutputs,
    Qnn_ProfileHandle_t profileHandle,
    Qnn_ErrorHandle_t* errorHandle);

typedef Qnn_ErrorHandle_t (*Qnn_GraphRetrieve_Fn_t)(
    Qnn_ContextHandle_t contextHandle,
    const char* graphName,
    Qnn_GraphHandle_t* graphHandle);

// Tensor
typedef Qnn_ErrorHandle_t (*Qnn_TensorCreateGraphTensor_Fn_t)(
    Qnn_GraphHandle_t graphHandle,
    Qnn_Tensor_t* tensor);

// Log
typedef Qnn_ErrorHandle_t (*Qnn_LogCreate_Fn_t)(
    Qnn_LogCallback_t callback,
    Qnn_LogLevel_t maxLogLevel,
    Qnn_LogHandle_t* logHandle);

typedef Qnn_ErrorHandle_t (*Qnn_LogFree_Fn_t)(
    Qnn_LogHandle_t logHandle);

// Profile
typedef Qnn_ErrorHandle_t (*Qnn_ProfileCreate_Fn_t)(
    Qnn_BackendHandle_t backendHandle,
    Qnn_LogLevel_t level,
    Qnn_ProfileHandle_t* profileHandle);

typedef Qnn_ErrorHandle_t (*Qnn_ProfileFree_Fn_t)(
    Qnn_ProfileHandle_t profileHandle);

// ── QNN Interface Structure ───────────────────────────────────────────────
// This is the master dispatch table. All QNN operations go through this.
// Obtained via QnnInterface_getProviders() after dlopen(libQnnHtp.so).

typedef struct {
    Qnn_Version_t  apiVersion;
    // Backend
    Qnn_BackendCreate_Fn_t          backendCreate;
    Qnn_BackendFree_Fn_t            backendFree;
    Qnn_BackendGetApiVersion_Fn_t   backendGetApiVersion;
    // Device
    Qnn_DeviceCreate_Fn_t           deviceCreate;
    Qnn_DeviceFree_Fn_t             deviceFree;
    // Context
    Qnn_ContextCreate_Fn_t          contextCreate;
    Qnn_ContextFree_Fn_t            contextFree;
    Qnn_ContextGetBinarySize_Fn_t   contextGetBinarySize;
    Qnn_ContextGetBinary_Fn_t       contextGetBinary;
    Qnn_ContextCreateFromBinary_Fn_t contextCreateFromBinary;
    // Graph
    Qnn_GraphCreate_Fn_t            graphCreate;
    Qnn_GraphAddNode_Fn_t           graphAddNode;
    Qnn_GraphFinalize_Fn_t          graphFinalize;
    Qnn_GraphExecute_Fn_t           graphExecute;
    Qnn_GraphRetrieve_Fn_t          graphRetrieve;
    // Tensor
    Qnn_TensorCreateGraphTensor_Fn_t tensorCreateGraphTensor;
    // Log
    Qnn_LogCreate_Fn_t              logCreate;
    Qnn_LogFree_Fn_t                logFree;
    // Profile
    Qnn_ProfileCreate_Fn_t          profileCreate;
    Qnn_ProfileFree_Fn_t            profileFree;
} QnnInterface_t;

// ── Provider Function ─────────────────────────────────────────────────────
// This is the entry point obtained via dlsym() after dlopen(libQnnHtp.so).
// It returns an array of interface providers (usually just one).

typedef Qnn_ErrorHandle_t (*QnnInterface_getProviders_Fn_t)(
    const QnnInterface_t*** providerList,
    uint32_t* numProviders);

// Symbol name to look up via dlsym():
#define QNN_INTERFACE_GET_PROVIDERS_SYMBOL "QnnInterface_getProviders"
