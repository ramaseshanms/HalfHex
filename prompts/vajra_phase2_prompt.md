# VAJRA — Phase 2: HQQ-Optimised INT4 on Hexagon HTP
## Prerequisite: Phase 1 complete. QNN INT8 beats llama.cpp. results/comparison.md row 3 filled.

> Exit criteria for Phase 2:
> `QNN INT4 (HQQ-initialised) decode rate > QNN INT8 decode rate`
> AND perplexity degradation < 0.5 points vs INT8 on wikitext2.
> Both conditions must hold. Speed without quality is not a win.

---

## 0. Phase 2 Mindset

Phase 1 proved HTP executes your model.
Phase 2 answers: **does HQQ's superior weight optimization translate to better
quality at INT4 than QNN's own PTQ?**

The thesis: HQQ minimises weight quantization error without calibration data.
QNN's PTQ minimises activation error using calibration data.
They're optimising different things. The question is which matters more
at INT4 on a transformer model's specific weight distributions.

You will know the answer only by measuring both.
This phase produces the most important quality/speed data point of the project.

---

## 1. HQQ Quantization of Qwen3-1.7B

### 1.1 Install HQQ
```bash
source $HOME/vajra-env/bin/activate

# HQQ from the official mobiusml repo (more actively maintained than dropbox fork)
pip install hqq==0.2.1
pip install transformers>=4.51.0  # already installed from Phase 1

# For perplexity evaluation
pip install datasets lm_eval
```

### 1.2 Quantize with HQQ (INT4)
```python
# hqq_quantize.py
# This is the calibration-free quantization step
# HQQ uses half-quadratic optimization per weight group, no data needed

import torch
import time
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig

MODEL_PATH  = "./models/qwen3-1.7b-hf"
OUTPUT_PATH = "./models/qwen3-1.7b-hqq-int4"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"[{time.strftime('%H:%M:%S')}] Loading Qwen3-1.7B in fp16...")
t0 = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",          # stays on CPU — HQQ quantizes in-place
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print(f"[{time.strftime('%H:%M:%S')}] Loaded in {time.perf_counter()-t0:.1f}s")

# ── HQQ quantization config ──────────────────────────────────────────────────
# group_size=64: HTP V73 tiling-friendly
# nbits=4: INT4 target
# axis=1: column-wise quantization (more stable than row-wise for transformers)
quant_config = BaseQuantizeConfig(
    nbits=4,
    group_size=64,      # try 64 and 128, compare perplexity
    quant_zero=True,    # quantize zero-points too (saves memory)
    quant_scale=False,  # keep scales in fp16 (precision matters here)
    offload_meta=False,
    view_as_float=False,
)

# Layers to skip — these must stay in fp16
# Same policy as our quantization_config.json from Phase 1
SKIP_MODULES = [
    "lm_head",           # output projection — affects token distribution directly
    "model.embed_tokens", # embedding table — quality-critical
    "model.norm",        # final layer norm — tiny but numerically sensitive
]

print(f"[{time.strftime('%H:%M:%S')}] Starting HQQ quantization...")
print(f"  Config: nbits=4, group_size={quant_config.group_size}, axis=1")
print(f"  Skipping: {SKIP_MODULES}")
t0 = time.perf_counter()

# Quantize all linear layers except skipped ones
quantized_layers = 0
skipped_layers   = 0
layer_stats      = []

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        if any(skip in name for skip in SKIP_MODULES):
            skipped_layers += 1
            print(f"  SKIP: {name}")
            continue

        t_layer = time.perf_counter()
        HQQLinear.set_backend(HQQBackend.PYTORCH)

        # Quantize this layer in-place
        parent_name = ".".join(name.split(".")[:-1])
        child_name  = name.split(".")[-1]
        parent = model.get_submodule(parent_name)
        hqq_layer = HQQLinear(module, quant_config, compute_dtype=torch.float16)
        setattr(parent, child_name, hqq_layer)

        layer_time = time.perf_counter() - t_layer
        w_shape    = module.weight.shape
        quantized_layers += 1

        stats = {
            "name": name,
            "shape": list(w_shape),
            "time_s": round(layer_time, 3),
        }
        layer_stats.append(stats)
        print(f"  [{quantized_layers:3d}] {name:60s} {str(w_shape):20s} {layer_time:.2f}s")

total_time = time.perf_counter() - t0
print(f"\n[{time.strftime('%H:%M:%S')}] HQQ complete!")
print(f"  Quantized: {quantized_layers} layers")
print(f"  Skipped:   {skipped_layers} layers")
print(f"  Total time: {total_time:.1f}s")

# Save quantization stats
with open(f"{OUTPUT_PATH}/hqq_layer_stats.json", "w") as f:
    json.dump({
        "config": {
            "nbits": quant_config.nbits,
            "group_size": quant_config.group_size,
            "axis": 1,
        },
        "layers": layer_stats,
        "total_time_s": round(total_time, 2),
        "quantized_count": quantized_layers,
        "skipped_count": skipped_layers,
    }, f, indent=2)

# Save the quantized model
print(f"[{time.strftime('%H:%M:%S')}] Saving quantized model...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"[{time.strftime('%H:%M:%S')}] ✓ Saved to {OUTPUT_PATH}")
```

### 1.3 Measure HQQ Perplexity (wikitext2)
```python
# measure_perplexity.py
# Wikitext2 is the standard — your numbers must be comparable to published results

import torch
import math
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def measure_perplexity(model_path, model_type, max_samples=128, seq_len=512):
    """
    Standard wikitext2 perplexity measurement.
    Matches the protocol used in GPTQ, AWQ, HQQ papers.
    """
    print(f"\n{'='*60}")
    print(f"Measuring perplexity: {model_type}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()

    # Load wikitext2 test split
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text    = "\n\n".join(dataset["text"])
    tokens  = tokenizer(text, return_tensors="pt")["input_ids"]

    # Compute perplexity over sliding windows
    nlls    = []
    n_done  = 0
    t0      = time.perf_counter()

    for i in range(0, min(len(tokens[0]) - seq_len, max_samples * seq_len), seq_len):
        input_ids = tokens[:, i : i + seq_len]
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nll     = outputs.loss
        nlls.append(nll.item())
        n_done += 1

        if n_done % 10 == 0:
            current_ppl = math.exp(sum(nlls) / len(nlls))
            elapsed     = time.perf_counter() - t0
            print(f"  [{n_done:3d}/{max_samples}] ppl={current_ppl:.3f} ({elapsed:.0f}s)")

    final_ppl = math.exp(sum(nlls) / len(nlls))
    print(f"\n  ✓ Final perplexity: {final_ppl:.4f}  (n={n_done} windows)")
    return final_ppl

# Measure all three variants — fp16, QNN PTQ INT8 (Phase 1), HQQ INT4
results = {}

# 1. fp16 baseline (ground truth)
results["fp16_baseline"] = measure_perplexity(
    "./models/qwen3-1.7b-hf", "fp16 baseline")

# 2. HQQ INT4 (group_size=64)
results["hqq_int4_gs64"] = measure_perplexity(
    "./models/qwen3-1.7b-hqq-int4", "HQQ INT4 group_size=64")

# Print comparison
print("\n" + "="*60)
print("PERPLEXITY COMPARISON")
print("="*60)
for name, ppl in results.items():
    degradation = ppl - results["fp16_baseline"]
    print(f"  {name:30s}: {ppl:.4f}  (Δ={degradation:+.4f} vs fp16)")

print("\nAcceptable INT4 degradation: < 0.5 ppl points")
print("If HQQ INT4 degradation > 1.0: try group_size=128")
print("If HQQ INT4 degradation > 2.0: something is wrong, check skipped layers")

# Save results
import json
with open("./results/perplexity_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 2. The Bridge: HQQ Parameters → QNN Quantization Overrides

This is the novel engineering work. No existing tool does this.

### 2.1 Extract HQQ Parameters
```python
# extract_hqq_params.py
# Extract the optimised scale/zero-point from every HQQ layer
# These will override QNN's own PTQ estimates

import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM
from hqq.core.quantize import HQQLinear

model = AutoModelForCausalLM.from_pretrained(
    "./models/qwen3-1.7b-hqq-int4",
    torch_dtype=torch.float16,
)

qnn_overrides = {
    "activation_encodings": {},
    "param_encodings": {}
}

# FP16 layers (same as Phase 1 config)
for layer_name in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]:
    qnn_overrides["param_encodings"][layer_name] = [
        {"bitwidth": 16, "dtype": "float"}
    ]

layer_count      = 0
problem_layers   = []

for name, module in model.named_modules():
    if not isinstance(module, HQQLinear):
        continue

    # Extract HQQ quantization parameters
    # HQQ stores: W_q (quantized weights), meta (scale, zero, group_size)
    try:
        meta       = module.meta
        scale      = meta["scale"]   # shape: [num_groups, 1] or [num_groups]
        zero       = meta["zero"]    # shape: [num_groups, 1] or [num_groups]
        group_size = meta["group_size"]
        nbits      = meta["nbits"]

        # Flatten to 1D
        scale_flat = scale.flatten().float().numpy()
        zero_flat  = zero.flatten().float().numpy()

        # Convert to QNN encoding format
        # QNN uses (min, max) per group, not (scale, zero)
        # Dequantization: W = scale * (W_q - zero)
        # So: min = scale * (0 - zero), max = scale * (2^nbits - 1 - zero)
        max_quant_val = 2**nbits - 1
        encodings = []
        for i in range(len(scale_flat)):
            s = float(scale_flat[i])
            z = float(zero_flat[i])
            enc_min = s * (0 - z)
            enc_max = s * (max_quant_val - z)
            encodings.append({
                "bitwidth":     nbits,
                "dtype":        "int",
                "is_symmetric": False,
                "min":          enc_min,
                "max":          enc_max,
            })

        # QNN weight names use .weight suffix
        qnn_weight_name = name + ".weight"
        qnn_overrides["param_encodings"][qnn_weight_name] = encodings

        layer_count += 1
        if layer_count % 20 == 0:
            print(f"  Processed {layer_count} layers... (last: {name})")
            print(f"    group_size={group_size}, scale_range=[{scale_flat.min():.4f}, {scale_flat.max():.4f}]")
            print(f"    zero_range=[{zero_flat.min():.4f}, {zero_flat.max():.4f}]")
            print(f"    num_groups={len(encodings)}")

    except Exception as e:
        print(f"  PROBLEM extracting {name}: {e}")
        problem_layers.append({"name": name, "error": str(e)})

print(f"\n✓ Extracted {layer_count} HQQ layers into QNN format")
print(f"  Problem layers: {len(problem_layers)}")
if problem_layers:
    print("  INVESTIGATE THESE:")
    for p in problem_layers:
        print(f"    {p['name']}: {p['error']}")

# Save override file
with open("./quantization_config_hqq_int4.json", "w") as f:
    json.dump(qnn_overrides, f)

print(f"\n✓ Saved: ./quantization_config_hqq_int4.json")
print(f"  Total param_encodings entries: {len(qnn_overrides['param_encodings'])}")
```

### 2.2 QNN INT4 Conversion with HQQ Overrides
```bash
mkdir -p ./models/qwen3-1.7b-qnn-hqq-int4 ./logs

qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_sim.onnx \
  --output_path   ./models/qwen3-1.7b-qnn-hqq-int4/qwen3_hqq_int4.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --act_bw    8 \    # activations stay INT8 (W4A8 is the target format)
  --weight_bw 4 \    # weights use HQQ-computed INT4 encodings
  --bias_bw   32 \
  --float_fallback \
  --keep_int64_inputs \
  --quantization_overrides ./quantization_config_hqq_int4.json \
  --verbose \
  2>&1 | tee ./logs/qnn_hqq_int4_conversion.log

# Fallback analysis — INT4 has more fallbacks than INT8, expect some
echo "=== FALLBACK COUNT ==="
grep -c "FALLBACK\|fallback" ./logs/qnn_hqq_int4_conversion.log || echo "0"

echo "=== FALLBACK DETAILS ==="
grep "FALLBACK\|fallback" ./logs/qnn_hqq_int4_conversion.log | sort | uniq -c | sort -rn | head -20

# Compare fallback count between INT8 (Phase 1) and INT4 (this phase)
echo "=== INT8 vs INT4 fallback delta ==="
INT8_FALLBACKS=$(grep -c "FALLBACK\|fallback" ./logs/qnn_int8_conversion.log 2>/dev/null || echo 0)
INT4_FALLBACKS=$(grep -c "FALLBACK\|fallback" ./logs/qnn_hqq_int4_conversion.log 2>/dev/null || echo 0)
echo "INT8 fallbacks: $INT8_FALLBACKS"
echo "INT4 fallbacks: $INT4_FALLBACKS"
echo "Delta: $((INT4_FALLBACKS - INT8_FALLBACKS)) additional fallbacks with INT4"
```

### 2.3 Also Convert with QNN Native INT4 PTQ (for comparison)
```bash
# This gives you the comparison: HQQ-initialised vs QNN's own INT4 PTQ
# You need BOTH to make the claim "HQQ is better"

mkdir -p ./models/qwen3-1.7b-qnn-ptq-int4

qnn-onnx-converter \
  --input_network ./models/qwen3-1.7b-onnx/model_sim.onnx \
  --output_path   ./models/qwen3-1.7b-qnn-ptq-int4/qwen3_ptq_int4.cpp \
  --input_dim input_ids "1,1" \
  --input_dim attention_mask "1,512" \
  --act_bw    8 \
  --weight_bw 4 \
  --bias_bw   32 \
  --float_fallback \
  --keep_int64_inputs \
  --quantization_overrides ./quantization_config_int8.json \  # same float-fallback policy
  --input_list ./calibration/int8/input_list.txt \           # QNN uses your calibration data
  --verbose \
  2>&1 | tee ./logs/qnn_ptq_int4_conversion.log

# Compile PTQ INT4
qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn-ptq-int4/qwen3_ptq_int4.cpp \
  -b ./models/qwen3-1.7b-qnn-ptq-int4/qwen3_ptq_int4.bin \
  -o ./models/qwen3-1.7b-qnn-ptq-int4/ \
  -t aarch64-android \
  2>&1 | tee ./logs/qnn_ptq_int4_compile.log

# Compile HQQ INT4
qnn-model-lib-generator \
  -c ./models/qwen3-1.7b-qnn-hqq-int4/qwen3_hqq_int4.cpp \
  -b ./models/qwen3-1.7b-qnn-hqq-int4/qwen3_hqq_int4.bin \
  -o ./models/qwen3-1.7b-qnn-hqq-int4/ \
  -t aarch64-android \
  2>&1 | tee ./logs/qnn_hqq_int4_compile.log
```

---

## 3. Group Size Sensitivity Analysis

```python
# group_size_sweep.py
# HTP V73 tiling affects which group sizes execute natively vs with padding
# Run this BEFORE committing to a group size for the final model

from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, math, time, json

MODEL_PATH = "./models/qwen3-1.7b-hf"

def quick_perplexity(model, tokenizer, n_samples=32):
    from datasets import load_dataset
    dataset  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text     = "\n\n".join(dataset["text"][:100])
    tokens   = tokenizer(text, return_tensors="pt")["input_ids"]
    nlls     = []
    seq_len  = 256
    model.eval()
    for i in range(0, min(len(tokens[0]) - seq_len, n_samples * seq_len), seq_len):
        inp = tokens[:, i : i + seq_len]
        with torch.no_grad():
            out = model(inp, labels=inp)
        nlls.append(out.loss.item())
    return math.exp(sum(nlls) / len(nlls))

GROUP_SIZES = [32, 64, 128]
results     = {}

for gs in GROUP_SIZES:
    print(f"\n{'='*50}")
    print(f"Testing group_size={gs}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cpu")

    cfg = BaseQuantizeConfig(nbits=4, group_size=gs, quant_zero=True)
    SKIP = ["lm_head", "embed_tokens", "model.norm"]

    t0 = time.perf_counter()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(s in name for s in SKIP):
                continue
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            child  = name.split(".")[-1]
            setattr(parent, child,
                    HQQLinear(module, cfg, compute_dtype=torch.float16))

    quant_time = time.perf_counter() - t0
    ppl        = quick_perplexity(model, tokenizer)

    results[gs] = {"ppl": ppl, "quant_time_s": quant_time}
    print(f"  group_size={gs}: ppl={ppl:.4f}, quant_time={quant_time:.1f}s")
    del model

print("\n" + "="*50)
print("GROUP SIZE SWEEP RESULTS")
print("="*50)
for gs, r in results.items():
    print(f"  gs={gs:4d}: ppl={r['ppl']:.4f}  quant_time={r['quant_time_s']:.1f}s")

# Pick lowest ppl that is HTP-friendly
# gs=64 and gs=128 are generally HTP-friendly
# gs=32 may require padding on some layer shapes — check conversion logs

with open("./results/group_size_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## 4. On-Device Validation

```bash
# Push both INT4 variants to device
adb push ./models/qwen3-1.7b-qnn-ptq-int4/libqwen3_ptq_int4.so  /data/local/tmp/vajra/
adb push ./models/qwen3-1.7b-qnn-hqq-int4/libqwen3_hqq_int4.so  /data/local/tmp/vajra/

# Run benchmark for PTQ INT4
adb shell "cd /data/local/tmp/vajra && \
  LD_LIBRARY_PATH=/data/local/tmp/vajra \
  ./vajra_benchmark \
  --model libqwen3_ptq_int4.so \
  --label qnn_ptq_int4 \
  --warmup 5 --tokens 100 --sustained 120 \
  2>&1" | tee ./logs/benchmark_ptq_int4.log

# Run benchmark for HQQ INT4
adb shell "cd /data/local/tmp/vajra && \
  LD_LIBRARY_PATH=/data/local/tmp/vajra \
  ./vajra_benchmark \
  --model libqwen3_hqq_int4.so \
  --label qnn_hqq_int4 \
  --warmup 5 --tokens 100 --sustained 120 \
  2>&1" | tee ./logs/benchmark_hqq_int4.log

# Pull profiles
adb pull /sdcard/vajra_qnn_ptq_int4_profile.csv  ./logs/
adb pull /sdcard/vajra_qnn_hqq_int4_profile.csv  ./logs/
```

---

## 5. Phase 2 Analysis

```python
# analyze_phase2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os

os.makedirs("./results/plots", exist_ok=True)

# Load all profile CSVs
configs = {
    "llama_cpu":   None,   # from phase 1 baseline file
    "qnn_int8":    pd.read_csv("./logs/vajra_phase1_profile.csv"),
    "qnn_ptq_int4": pd.read_csv("./logs/benchmark_ptq_int4.log.csv"),
    "qnn_hqq_int4": pd.read_csv("./logs/benchmark_hqq_int4.log.csv"),
}

print("="*70)
print("PHASE 2 RESULTS SUMMARY")
print("="*70)

speeds = {}
for name, df in configs.items():
    if df is None:
        continue
    decode = df[df["name"] == "decode_step_total"]["duration_us"]
    if decode.empty:
        print(f"  WARNING: No decode_step_total data for {name}")
        continue
    tps_mean = 1e6 / decode.mean()
    tps_p50  = 1e6 / decode.median()
    tps_min  = 1e6 / decode.max()  # slowest decode = worst sustained speed
    speeds[name] = tps_mean
    print(f"  {name:25s}: mean={tps_mean:.2f} tok/s  p50={tps_p50:.2f}  floor={tps_min:.2f}")

# Load perplexity results
with open("./results/perplexity_comparison.json") as f:
    ppl_results = json.load(f)

print("\nPerplexity on wikitext2:")
for name, ppl in ppl_results.items():
    print(f"  {name:30s}: {ppl:.4f}")

# The key question
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
if "qnn_hqq_int4" in speeds and "qnn_ptq_int4" in speeds:
    hqq_vs_ptq_speed = (speeds["qnn_hqq_int4"] / speeds["qnn_ptq_int4"] - 1) * 100
    print(f"HQQ INT4 vs PTQ INT4 speed: {hqq_vs_ptq_speed:+.1f}%")

if "hqq_int4_gs64" in ppl_results and "fp16_baseline" in ppl_results:
    ppl_delta = ppl_results["hqq_int4_gs64"] - ppl_results["fp16_baseline"]
    print(f"HQQ INT4 perplexity delta vs fp16: {ppl_delta:+.4f}")
    if ppl_delta < 0.5:
        print("  → ACCEPTABLE: Quality degradation within budget")
    else:
        print("  → INVESTIGATE: Quality degradation exceeds budget")

# Speedup chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Speed bar chart
labels = list(speeds.keys())
values = list(speeds.values())
bars = axes[0].bar(labels, values, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"])
axes[0].set_title("Decode Speed Comparison (tok/s)\nHigher is better")
axes[0].set_ylabel("Tokens per second")
axes[0].tick_params(axis="x", rotation=20)
for bar, val in zip(bars, values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

# Perplexity chart
ppl_labels = [k for k in ppl_results if "baseline" not in k or "fp16" in k]
ppl_values = [ppl_results[k] for k in ppl_labels]
axes[1].bar(ppl_labels, ppl_values, color=["#2ecc71", "#3498db", "#e74c3c"])
axes[1].set_title("Perplexity on wikitext2\nLower is better")
axes[1].set_ylabel("Perplexity")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("./results/plots/phase2_comparison.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved: ./results/plots/phase2_comparison.png")
```

---

## 6. Phase 2 Completion Checklist

```
[ ] HQQ quantization completes without errors on Qwen3-1.7B
[ ] HQQ perplexity measured on wikitext2 (64 samples minimum)
[ ] HQQ INT4 perplexity delta vs fp16 < 0.5 points
    (if > 0.5: try group_size=128 before proceeding)
    (if > 1.0: check SKIP_MODULES list, something critical is being quantized)
[ ] HQQ params successfully extracted to QNN override JSON
    (verify: no NaN or inf in scale/zero values)
[ ] QNN PTQ INT4 compiled and benchmarked on device
[ ] QNN HQQ INT4 compiled and benchmarked on device
[ ] Both INT4 variants faster than INT8 baseline from Phase 1
[ ] HQQ INT4 decode rate >= PTQ INT4 decode rate
    (if HQQ is slower: the encoding injection has overhead — investigate fallbacks)
[ ] Group size sweep completed, optimal group size selected
[ ] results/comparison.md rows 3-5 filled in
[ ] phase2_comparison.png generated and saved
[ ] If HQQ quality superior to PTQ: document the delta — this is your research result
```

**Decision gate for arXiv paper:**
If HQQ INT4 achieves BOTH:
- Better perplexity than QNN PTQ INT4 (even 0.2 points matters)
- Same or faster speed

→ You have a publishable result. Begin writing the paper alongside Phase 3.
Title: *"HQQ-Initialized Quantization for Hexagon HTP: Calibration-Free
INT4 LLM Inference on Snapdragon Edge Devices"*
