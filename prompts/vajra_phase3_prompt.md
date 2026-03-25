# VAJRA — Phase 3: Per-Layer Mixed Precision on HTP
## Prerequisite: Phase 2 complete. HQQ INT4 benchmarked. Comparison table rows 3-5 filled.

> Exit criteria for Phase 3:
> Mixed precision configuration achieves BOTH:
> - Perplexity <= HQQ INT4 full model (quality preserved or improved)
> - Speed >= HQQ INT4 full model (no regression from mixed precision overhead)
> The goal is quality recovery in sensitive layers with zero speed cost.

---

## 0. Phase 3 Mindset

INT4 uniform quantization treats every layer identically.
But transformer layers are NOT equally sensitive to precision loss.

The FFN down-projection accumulates errors from gate and up projections.
The attention output projection is the last thing written to the residual stream.
The first and last transformer layers are disproportionately quality-sensitive.

Mixed precision assigns higher bit-width to sensitive layers and lower to
insensitive ones — targeting the same average memory footprint but better
perplexity. This is the concept behind K-quants in llama.cpp, reimplemented
here for HTP using HQQ's per-layer error signal.

---

## 1. Identify Sensitive Layers via HQQ Error Signal

```python
# find_sensitive_layers.py
# HQQ computes a per-layer quantization error during optimization.
# This error is your guide for which layers need INT8 vs INT4.

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig

MODEL_PATH = "./models/qwen3-1.7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")

# Quantize with error tracking enabled
cfg = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True)
SKIP = ["lm_head", "embed_tokens", "model.norm"]

layer_errors = {}

for name, module in model.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue
    if any(s in name for s in SKIP):
        continue

    # Compute quantization error for this layer
    W_orig  = module.weight.data.float()
    hqq_mod = HQQLinear(module, cfg, compute_dtype=torch.float16)

    # Dequantize and measure reconstruction error
    W_dequant = hqq_mod.dequantize().float()
    rel_error = (W_orig - W_dequant).norm() / W_orig.norm()

    layer_errors[name] = float(rel_error.item())
    print(f"  {name:65s}: error={rel_error:.6f}")

# Rank layers by error (highest error = most sensitive = needs INT8)
sorted_layers = sorted(layer_errors.items(), key=lambda x: x[1], reverse=True)

print("\n" + "="*70)
print("LAYERS RANKED BY QUANTIZATION ERROR (highest = most sensitive)")
print("="*70)
for i, (name, err) in enumerate(sorted_layers[:20]):
    flag = "⚠ HIGH" if err > 0.05 else ("→ MED" if err > 0.02 else "  LOW")
    print(f"  [{i+1:2d}] {flag} {name:60s}: {err:.6f}")

# Save for use in mixed precision config
with open("./results/layer_errors.json", "w") as f:
    json.dump(dict(sorted_layers), f, indent=2)

# Identify threshold: layers above P75 error get INT8, rest get INT4
errors  = np.array(list(layer_errors.values()))
p75     = np.percentile(errors, 75)
int8_layers = [n for n, e in layer_errors.items() if e > p75]
int4_layers = [n for n, e in layer_errors.items() if e <= p75]

print(f"\nThreshold (P75 error): {p75:.6f}")
print(f"INT8 layers: {len(int8_layers)} ({100*len(int8_layers)/len(layer_errors):.0f}%)")
print(f"INT4 layers: {len(int4_layers)} ({100*len(int4_layers)/len(layer_errors):.0f}%)")

with open("./configs/mixed_precision_layers.json", "w") as f:
    json.dump({"int8": int8_layers, "int4": int4_layers, "p75_threshold": float(p75)}, f, indent=2)
```

### 1.2 Also Check by Layer Position
```python
# check_layer_sensitivity_by_position.py
# First/last transformer layers are typically more sensitive
# Check if this holds for Qwen3-1.7B specifically

import json

with open("./results/layer_errors.json") as f:
    layer_errors = json.load(f)

# Group by transformer layer index
import re
from collections import defaultdict
layer_group = defaultdict(list)

for name, err in layer_errors.items():
    match = re.search(r'layers\.(\d+)\.', name)
    if match:
        layer_idx = int(match.group(1))
        layer_group[layer_idx].append(err)

print("Average quantization error by transformer layer:")
for idx in sorted(layer_group.keys()):
    avg_err = sum(layer_group[idx]) / len(layer_group[idx])
    bar = "█" * int(avg_err * 1000)
    print(f"  Layer {idx:2d}: {avg_err:.6f}  {bar}")

# Check first 3 and last 3 layers
n_layers = max(layer_group.keys()) + 1
print(f"\nFirst 3 layers avg error: {sum(sum(layer_group[i]) for i in range(3)) / (3*len(list(layer_group.values())[0])):.6f}")
print(f"Last  3 layers avg error: {sum(sum(layer_group[n_layers-3+i]) for i in range(3)) / (3*len(list(layer_group.values())[0])):.6f}")
```

---

## 2. Build Mixed Precision HQQ Config

```python
# build_mixed_precision_model.py
# Strategy: INT8 for top P75 error layers, INT4 for the rest
# This recovers quality in sensitive layers while keeping most of the INT4 speedup

import torch, json, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig

MODEL_PATH  = "./models/qwen3-1.7b-hf"
OUTPUT_PATH = "./models/qwen3-1.7b-hqq-mixed"

with open("./configs/mixed_precision_layers.json") as f:
    mp_config = json.load(f)

int8_set = set(mp_config["int8"])
int4_set = set(mp_config["int4"])

print(f"Mixed precision plan:")
print(f"  INT8: {len(int8_set)} layers")
print(f"  INT4: {len(int4_set)} layers")
print(f"  fp16: embed, lm_head, norm (always)")

model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

cfg_int4 = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True)
cfg_int8 = BaseQuantizeConfig(nbits=8, group_size=128, quant_zero=True)
SKIP     = ["lm_head", "embed_tokens", "model.norm"]

stats = {"int4": 0, "int8": 0, "fp16": 0}

for name, module in model.named_modules():
    if not isinstance(module, torch.nn.Linear):
        continue
    if any(s in name for s in SKIP):
        stats["fp16"] += 1
        continue

    parent = model.get_submodule(".".join(name.split(".")[:-1]))
    child  = name.split(".")[-1]

    if name in int8_set:
        setattr(parent, child, HQQLinear(module, cfg_int8, compute_dtype=torch.float16))
        stats["int8"] += 1
        print(f"  INT8: {name}")
    else:
        setattr(parent, child, HQQLinear(module, cfg_int4, compute_dtype=torch.float16))
        stats["int4"] += 1

print(f"\nQuantization complete: {stats}")

# Estimate memory footprint
# (rough estimate: INT8 layers = 1 byte/weight, INT4 = 0.5 byte/weight)
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"✓ Mixed precision model saved to {OUTPUT_PATH}")
```

---

## 3. Convert Mixed Precision Model to QNN

```python
# generate_mixed_precision_overrides.py
# Build the QNN override JSON that encodes INT8 for sensitive layers,
# INT4 for everything else — using HQQ's optimised params for both

import torch, json
from transformers import AutoModelForCausalLM
from hqq.core.quantize import HQQLinear

model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen3-1.7b-hqq-mixed",
    torch_dtype=torch.float16,
)

overrides = {
    "activation_encodings": {},
    "param_encodings": {
        "model.embed_tokens.weight": [{"bitwidth": 16, "dtype": "float"}],
        "lm_head.weight":            [{"bitwidth": 16, "dtype": "float"}],
        "model.norm.weight":         [{"bitwidth": 16, "dtype": "float"}],
    }
}

for name, module in model.named_modules():
    if not isinstance(module, HQQLinear):
        continue

    meta       = module.meta
    scale      = meta["scale"].flatten().float().numpy()
    zero       = meta["zero"].flatten().float().numpy()
    nbits      = meta["nbits"]   # 4 or 8 depending on which layer
    max_val    = 2**nbits - 1

    encodings = []
    for i in range(len(scale)):
        s = float(scale[i])
        z = float(zero[i])
        encodings.append({
            "bitwidth": nbits,
            "dtype": "int",
            "is_symmetric": False,
            "min": s * (0 - z),
            "max": s * (max_val - z),
        })

    overrides["param_encodings"][name + ".weight"] = encodings

with open("./quantization_config_hqq_mixed.json", "w") as f:
    json.dump(overrides, f)

# Report the breakdown
int4_count = sum(1 for v in overrides["param_encodings"].values()
                 if v[0].get("bitwidth") == 4)
int8_count = sum(1 for v in overrides["param_encodings"].values()
                 if v[0].get("bitwidth") == 8)
fp16_count = sum(1 for v in overrides["param_encodings"].values()
                 if v[0].get("dtype") == "float")
print(f"Override breakdown: INT4={int4_count}, INT8={int8_count}, fp16={fp16_count}")
```

```bash
# Convert mixed precision model to QNN
mkdir -p ./models/qwen3-1.7b-qnn-hqq-mixed

qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_sim.onnx \
  --output_path   ./models/qwen3-1.7b-qnn-hqq-mixed/qwen3_hqq_mixed.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --act_bw    8 \
  --weight_bw 4 \    # default INT4, overridden per layer in JSON
  --bias_bw   32 \
  --float_fallback \
  --quantization_overrides ./quantization_config_hqq_mixed.json \
  --verbose \
  2>&1 | tee ./logs/qnn_hqq_mixed_conversion.log

qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn-hqq-mixed/qwen3_hqq_mixed.cpp \
  -b ./models/qwen3-1.7b-qnn-hqq-mixed/qwen3_hqq_mixed.bin \
  -o ./models/qwen3-1.7b-qnn-hqq-mixed/ \
  -t aarch64-android \
  2>&1 | tee ./logs/qnn_hqq_mixed_compile.log
```

---

## 4. Perplexity Sweep: Find Optimal INT8 Threshold

```python
# threshold_sweep.py
# P75 was our initial guess. Sweep to find the optimal threshold.
# Trade-off: more INT8 layers = better quality but slower.

import numpy as np, json, math
from datasets import load_dataset

THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]  # percentile thresholds
results    = {}

with open("./results/layer_errors.json") as f:
    layer_errors = json.load(f)

errors = np.array(list(layer_errors.values()))

for pct in THRESHOLDS:
    threshold  = np.percentile(errors, pct * 100)
    n_int8     = sum(1 for e in errors if e > threshold)
    n_int4     = sum(1 for e in errors if e <= threshold)
    pct_int8   = 100 * n_int8 / len(errors)

    # Rough memory estimate: INT8 layers = 2x INT4 memory
    # This estimates the effective bit-width of the whole model
    eff_bits = (n_int8 * 8 + n_int4 * 4) / len(errors)

    print(f"P{pct*100:.0f}: threshold={threshold:.6f} → "
          f"{n_int8} INT8 ({pct_int8:.0f}%) + {n_int4} INT4 | "
          f"effective bits={eff_bits:.2f}")

    results[f"p{int(pct*100)}"] = {
        "threshold": float(threshold),
        "n_int8": n_int8, "n_int4": n_int4,
        "pct_int8": pct_int8, "eff_bits": eff_bits,
    }

# Choose the threshold where effective bits is closest to 4.5
# (slightly above INT4 to recover quality, significantly below INT8 for speed)
target_bits = 4.5
best_pct = min(results.keys(),
               key=lambda k: abs(results[k]["eff_bits"] - target_bits))
print(f"\nRecommended threshold: {best_pct} (effective bits = {results[best_pct]['eff_bits']:.2f})")
```

---

## 5. Phase 3 Completion Checklist

```
[ ] Layer sensitivity analysis complete (layer_errors.json)
[ ] Layer position analysis confirms first/last layer sensitivity pattern
[ ] Mixed precision HQQ model built and saved
[ ] Mixed precision QNN conversion successful
[ ] On-device benchmark: mixed precision vs full INT4 speed delta < 5%
    (if > 5% slower: too many INT8 layers, raise the threshold)
[ ] Mixed precision perplexity <= full INT4 HQQ perplexity
    (it should be better — if not, error extraction has a bug)
[ ] Optimal threshold identified via sweep
[ ] results/comparison.md row 6 filled in
[ ] All three variants (INT8, INT4-HQQ, mixed-HQQ) profiled with same methodology
```

---
---

