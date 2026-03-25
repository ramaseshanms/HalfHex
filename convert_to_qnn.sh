#!/bin/bash
# Convert ONNX model to QNN HTP-optimized binary
set -e

echo "═══════════════════════════════════════"
echo "ONNX → QNN Conversion Pipeline"
echo "═══════════════════════════════════════"

mkdir -p ./models/qwen3-1.7b-qnn ./models/qwen3-1.7b-qnn-int4 ./logs

# ── Step 1: Simplify ONNX graph ──────────────────────────────────────────
echo "[1/5] Simplifying ONNX graph..."
python -m onnxsim \
  ./models/qwen3-1.7b-onnx/model.onnx \
  ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --overwrite-input-shape "input_ids:1,1" "attention_mask:1,512" \
  --check-n 3

# ── Step 2: Validate with ONNX Runtime on host ──────────────────────────
echo "[2/5] Validating ONNX model on host..."
python -c "
import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession('./models/qwen3-1.7b-onnx/model_simplified.onnx')
inputs = {
    'input_ids': np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
    'attention_mask': np.ones((1, 5), dtype=np.int64),
}
# Warmup
for _ in range(3):
    outputs = sess.run(None, inputs)

# Profile
times = []
for _ in range(20):
    t0 = time.perf_counter()
    outputs = sess.run(None, inputs)
    times.append(time.perf_counter() - t0)

print(f'[PROFILE] ONNX host inference:')
print(f'  mean: {np.mean(times)*1000:.1f}ms')
print(f'  p50:  {np.percentile(times, 50)*1000:.1f}ms')
print(f'  p95:  {np.percentile(times, 95)*1000:.1f}ms')
print(f'  p99:  {np.percentile(times, 99)*1000:.1f}ms')
"

# ── Step 3: INT8 QNN conversion ──────────────────────────────────────────
echo "[3/5] Converting to QNN INT8..."
qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --output_path ./models/qwen3-1.7b-qnn/model.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --quantization_overrides ./quantization_config.json \
  --act_bw 8 \
  --weight_bw 8 \
  --bias_bw 32 \
  --float_fallback \
  --keep_int64_inputs \
  --verbose \
  2>&1 | tee ./logs/qnn_conversion.log

# ── Step 4: Compile INT8 to HTP binary ───────────────────────────────────
echo "[4/5] Compiling INT8 model to HTP binary..."
qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn/model.cpp \
  -b ./models/qwen3-1.7b-qnn/model.bin \
  -o ./models/qwen3-1.7b-qnn/ \
  -t android_aarch64 \
  2>&1 | tee ./logs/qnn_compilation.log

# ── Step 5: INT4 QNN conversion (after INT8 is validated) ────────────────
echo "[5/5] Converting to QNN INT4 (W4A8)..."
qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_simplified.onnx \
  --output_path ./models/qwen3-1.7b-qnn-int4/model.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --act_bw 8 \
  --weight_bw 4 \
  --bias_bw 32 \
  --float_fallback \
  --quantization_overrides ./quantization_config_int4.json \
  --verbose \
  2>&1 | tee ./logs/qnn_int4_conversion.log

qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn-int4/model.cpp \
  -b ./models/qwen3-1.7b-qnn-int4/model.bin \
  -o ./models/qwen3-1.7b-qnn-int4/ \
  -t android_aarch64 \
  2>&1 | tee -a ./logs/qnn_int4_conversion.log

echo ""
echo "═══════════════════════════════════════"
echo "QNN conversion complete!"
echo "Check ./logs/ for conversion logs"
echo "═══════════════════════════════════════"
