# QNN HTP Runtime — Performance Comparison

## Device: Nothing Phone 3a Pro (Snapdragon 7s Gen 3, 12GB RAM)
## Model: Qwen3-1.7B-Q4_K_M (1.19 GiB, 2.03B params)

Update after every milestone.

| Config                        | Prefill (tok/s) | Decode (tok/s) | Sustained 512tok | TTFT (512 tok) |
|-------------------------------|-----------------|----------------|------------------|----------------|
| llama.cpp Q4_K_M CPU 2t       | 12.40           | **9.05**       | —                | ~41.3s         |
| llama.cpp Q4_K_M CPU 4t (big) | 5.70            | 1.49           | 1.45             | ~89.8s         |
| llama.cpp Q4_K_M CPU 6t       | 23.54           | 3.74           | —                | ~21.8s         |
| llama.cpp Q4_K_M CPU 8t       | **25.29**       | 5.26           | **5.91**         | ~20.2s         |
| llama.cpp Q4_K_M Vulkan       | —               | —              | —                | —              |
| QNN INT8 (ONNX → HTP)        | —               | —              | —                | —              |
| QNN INT4 (ONNX → HTP)        | —               | —              | —                | —              |
| QNN INT4 + KV cache pinned   | —               | —              | —                | —              |
| QNN INT4 + async pipeline    | —               | —              | —                | —              |
| QNN INT4 + speculative decode| —               | —              | —                | —              |

### Key Insight: Optimal thread strategy
- **Prefill (batched, compute-bound):** 8 threads → 25.29 t/s
- **Decode (sequential, memory-bound):** 2 threads → 9.05 t/s
- **Hybrid strategy:** Switch thread count between phases for best end-to-end latency

### Thermal profile during benchmarks
- Cold: 32-35°C → Peak: 42.4°C (stock governor, no performance mode)
- No throttling observed at any point
- Well within 55°C safety limit

### Targets for QNN HTP runtime
- Decode: >15 t/s (1.7x improvement over best CPU baseline)
- Prefill: >40 t/s (1.6x improvement over best CPU baseline)
- Sustained: >12 t/s over 120 seconds
