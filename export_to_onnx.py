#!/usr/bin/env python3
"""Export Qwen3-1.7B from HuggingFace to ONNX format with profiling hooks."""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.exporters.onnx import main_export
import os
from collections import Counter

MODEL_PATH = "./models/qwen3-1.7b-hf"
ONNX_PATH  = "./models/qwen3-1.7b-onnx"

print("[PROFILE] Starting model load...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
t1 = time.perf_counter()
print(f"[PROFILE] Model load: {t1-t0:.2f}s")
print(f"[PROFILE] Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"[PROFILE] Model size fp16: {sum(p.numel()*2 for p in model.parameters())/1e9:.2f}GB")

# Export — sequence length 512 is your target context
print("[PROFILE] Starting ONNX export...")
t0 = time.perf_counter()
main_export(
    model_name_or_path=MODEL_PATH,
    output=ONNX_PATH,
    task="text-generation-with-past",  # CRITICAL: with-past = KV cache support
    opset=17,
    device="cpu",
    fp16=False,     # keep fp32 for ONNX — QNN handles quantization separately
    optimize="O2",
)
t1 = time.perf_counter()
print(f"[PROFILE] ONNX export: {t1-t0:.2f}s")

# Profile the exported graph
import onnx
model_onnx = onnx.load(f"{ONNX_PATH}/model.onnx")
print(f"[PROFILE] ONNX graph nodes: {len(model_onnx.graph.node)}")
print(f"[PROFILE] ONNX inputs: {[i.name for i in model_onnx.graph.input]}")
print(f"[PROFILE] ONNX outputs: {[o.name for o in model_onnx.graph.output]}")

# Count op types — know what you're converting
op_counts = Counter(node.op_type for node in model_onnx.graph.node)
print("[PROFILE] Op type distribution:")
for op, count in op_counts.most_common(15):
    print(f"  {op:30s}: {count}")
