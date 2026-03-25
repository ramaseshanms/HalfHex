# QNN HTP Runtime — Performance Comparison

Update after every milestone.

| Config                        | Prefill (tok/s) | Decode (tok/s) | Sustained 120s | TTFT (512 tok) |
|-------------------------------|-----------------|----------------|----------------|----------------|
| llama.cpp Q4_0 CPU baseline   |                 |                |                |                |
| llama.cpp Q4_0 Vulkan         |                 |                |                |                |
| QNN INT8 (ONNX → HTP)         |                 |                |                |                |
| QNN INT4 (ONNX → HTP)         |                 |                |                |                |
| QNN INT4 + KV cache pinned    |                 |                |                |                |
| QNN INT4 + async pipeline     |                 |                |                |                |
| QNN INT4 + speculative decode |                 |                |                |                |
