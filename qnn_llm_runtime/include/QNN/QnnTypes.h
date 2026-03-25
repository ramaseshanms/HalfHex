// ============================================================================
// QNN/QnnTypes.h — Minimal QNN Type Definitions
// ============================================================================
//
// PURPOSE:
//   Provides the fundamental type definitions needed to interface with the
//   Qualcomm AI Engine Direct (QNN) SDK. These definitions match the public
//   API specification documented at:
//     https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/
//
// USAGE:
//   When the actual QNN SDK is available, replace this directory with:
//     -DQNN_SDK_ROOT=/path/to/qnn-sdk
//   and use ${QNN_SDK_ROOT}/include/QNN/ instead.
//
//   This file exists so the HalfHex runtime can compile and be tested
//   without the proprietary QNN SDK installed. At runtime, the actual
//   QNN shared libraries (libQnnHtp.so) are loaded via dlopen().
//
// LICENSE:
//   Type definitions derived from Qualcomm's public API documentation.
//   No proprietary code is included. This is an API-compatible header
//   for compilation purposes only.
//
// ============================================================================

#pragma once

#include <cstdint>
#include <cstddef>

// ── Error Handle ──────────────────────────────────────────────────────────
typedef int32_t Qnn_ErrorHandle_t;

#define QNN_SUCCESS         0
#define QNN_ERROR_GENERAL   1
#define QNN_MIN_ERROR_GRAPH 1000

// ── Opaque Handles ────────────────────────────────────────────────────────
typedef void* Qnn_BackendHandle_t;
typedef void* Qnn_DeviceHandle_t;
typedef void* Qnn_ContextHandle_t;
typedef void* Qnn_GraphHandle_t;
typedef void* Qnn_LogHandle_t;
typedef void* Qnn_ProfileHandle_t;

#define QNN_BACKEND_HANDLE_NULL  nullptr
#define QNN_DEVICE_HANDLE_NULL   nullptr
#define QNN_CONTEXT_HANDLE_NULL  nullptr
#define QNN_GRAPH_HANDLE_NULL    nullptr

// ── Data Types ────────────────────────────────────────────────────────────
typedef enum {
    QNN_DATATYPE_FLOAT_32    = 0x0232,
    QNN_DATATYPE_FLOAT_16    = 0x0216,
    QNN_DATATYPE_INT_8       = 0x0108,
    QNN_DATATYPE_INT_16      = 0x0116,
    QNN_DATATYPE_INT_32      = 0x0132,
    QNN_DATATYPE_INT_64      = 0x0164,
    QNN_DATATYPE_UINT_8      = 0x0408,
    QNN_DATATYPE_UINT_16     = 0x0416,
    QNN_DATATYPE_UINT_32     = 0x0432,
    QNN_DATATYPE_BOOL_8      = 0x0608,
    QNN_DATATYPE_UFIXED_POINT_8  = 0x0508,
    QNN_DATATYPE_UFIXED_POINT_16 = 0x0516,
    QNN_DATATYPE_UNDEFINED   = 0x7FFFFFFF,
} Qnn_DataType_t;

// ── Tensor Types ──────────────────────────────────────────────────────────
typedef enum {
    QNN_TENSOR_TYPE_APP_WRITE     = 0,
    QNN_TENSOR_TYPE_APP_READ      = 1,
    QNN_TENSOR_TYPE_APP_READWRITE = 2,
    QNN_TENSOR_TYPE_NATIVE        = 3,
    QNN_TENSOR_TYPE_STATIC        = 4,
    QNN_TENSOR_TYPE_NULL          = 5,
    QNN_TENSOR_TYPE_UNDEFINED     = 0x7FFFFFFF,
} Qnn_TensorType_t;

// ── Tensor Memory Type ────────────────────────────────────────────────────
typedef enum {
    QNN_TENSORMEMTYPE_RAW         = 0,
    QNN_TENSORMEMTYPE_MEMHANDLE   = 1,
    QNN_TENSORMEMTYPE_UNDEFINED   = 0x7FFFFFFF,
} Qnn_TensorMemType_t;

// ── Client Buffer ─────────────────────────────────────────────────────────
typedef struct {
    void*   data;
    uint32_t dataSize;
} Qnn_ClientBuffer_t;

// ── Quantize Params ───────────────────────────────────────────────────────
typedef enum {
    QNN_QUANTIZATION_ENCODING_UNDEFINED          = 0,
    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET       = 1,
    QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET  = 2,
    QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET    = 3,
    QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET = 4,
} Qnn_QuantizationEncoding_t;

typedef struct {
    float    scale;
    int32_t  offset;
} Qnn_ScaleOffset_t;

typedef struct {
    Qnn_QuantizationEncoding_t encodingDefinition;
    union {
        Qnn_ScaleOffset_t scaleOffsetEncoding;
    };
} Qnn_QuantizeParams_t;

#define QNN_QUANTIZE_PARAMS_INIT  \
    { QNN_QUANTIZATION_ENCODING_UNDEFINED, { {0.0f, 0} } }

// ── Tensor V2 ─────────────────────────────────────────────────────────────
typedef struct {
    uint32_t           id;
    const char*        name;
    Qnn_TensorType_t   type;
    Qnn_DataType_t     dataType;
    Qnn_QuantizeParams_t quantizeParams;
    uint32_t           rank;
    uint32_t*          dimensions;
    Qnn_TensorMemType_t memType;
    union {
        Qnn_ClientBuffer_t clientBuf;
    };
} Qnn_Tensor_t;

#define QNN_TENSOR_INIT {0, nullptr, QNN_TENSOR_TYPE_UNDEFINED, \
    QNN_DATATYPE_UNDEFINED, QNN_QUANTIZE_PARAMS_INIT, 0, nullptr, \
    QNN_TENSORMEMTYPE_RAW, {{nullptr, 0}}}

// ── Operator Config ───────────────────────────────────────────────────────
typedef struct {
    const char* name;
    const char* packageName;
    const char* typeName;
    uint32_t    numOfParams;
    void*       params;
    uint32_t    numOfInputs;
    Qnn_Tensor_t* inputTensors;
    uint32_t    numOfOutputs;
    Qnn_Tensor_t* outputTensors;
} Qnn_OpConfig_t;

// ── Graph Config ──────────────────────────────────────────────────────────
typedef enum {
    QNN_GRAPH_CONFIG_OPTION_PRIORITY  = 1,
    QNN_GRAPH_CONFIG_OPTION_UNDEFINED = 0x7FFFFFFF,
} Qnn_GraphConfigOption_t;

typedef struct {
    Qnn_GraphConfigOption_t option;
    // Simplified — real SDK has a union here
} Qnn_GraphConfig_t;

// ── Backend Config ────────────────────────────────────────────────────────
typedef struct {
    uint32_t option;
    // Simplified
} Qnn_BackendConfig_t;

// ── Context Config ────────────────────────────────────────────────────────
typedef struct {
    uint32_t option;
} Qnn_ContextConfig_t;

// ── Log Config ────────────────────────────────────────────────────────────
typedef enum {
    QNN_LOG_LEVEL_ERROR   = 1,
    QNN_LOG_LEVEL_WARN    = 2,
    QNN_LOG_LEVEL_INFO    = 3,
    QNN_LOG_LEVEL_VERBOSE = 4,
    QNN_LOG_LEVEL_DEBUG   = 5,
    QNN_LOG_LEVEL_MAX     = 0x7FFFFFFF,
} Qnn_LogLevel_t;

typedef void (*Qnn_LogCallback_t)(const char* fmt, Qnn_LogLevel_t level,
                                   uint64_t timestamp, ...);
