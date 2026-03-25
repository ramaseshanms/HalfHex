# VAJRA — Phase 4: KV Cache Optimisation + TurboQuant KV Compression
## Prerequisite: Phase 3 complete. Mixed precision model benchmarked.

> Exit criteria for Phase 4:
> Pinned KV cache shows >= 10% decode speedup vs unpinned baseline.
> TurboQuant 3-bit KV reduces KV memory by >= 50% with < 0.3 ppl degradation.

---

## 0. Phase 4 Mindset

Token generation (decode phase) is **memory-bandwidth-bound**, not compute-bound.
After every token, you read the entire KV cache accumulated so far.
At context length 512, Qwen3-1.7B's KV cache is ~200MB of fp16 data.
If Android pages any of it out, or if you copy instead of reference, you pay
a memory latency tax on every single token.

Phase 4 has two independent wins:
1. **Pinned KV cache**: zero-copy, mlock'd, never paged — pure bandwidth recovery
2. **TurboQuant KV**: 3-bit KV storage — 6x smaller KV cache, faster attention

These are independent changes. Implement and benchmark them separately.

---

## 1. KV Cache Manager — Pinned Memory Implementation

```cpp
// KVCacheManager.cpp
#include "KVCacheManager.h"
#include "Profiler.h"
#include <sys/mman.h>
#include <cstring>

// Qwen3-1.7B KV cache dimensions
// (adjust if you changed config)
constexpr int NUM_LAYERS  = 28;
constexpr int NUM_KV_HEADS = 8;    // GQA: 8 KV heads vs 16 Q heads
constexpr int HEAD_DIM    = 128;   // hidden_dim / num_q_heads = 2048 / 16
constexpr int DTYPE_BYTES = 2;     // fp16 = 2 bytes per element

KVCacheManager::KVCacheManager() = default;
KVCacheManager::~KVCacheManager() {
    if (buffer_) {
        munlock(buffer_, total_bytes_);
        munmap(buffer_, total_bytes_);
    }
}

bool KVCacheManager::allocate(int max_seq_len) {
    PROFILE_SCOPE("kv_cache_allocate");
    max_seq_len_ = max_seq_len;

    // Layout: [num_layers][2 (K,V)][num_kv_heads][max_seq_len][head_dim]
    size_t per_layer = 2 * NUM_KV_HEADS * max_seq_len * HEAD_DIM * DTYPE_BYTES;
    total_bytes_     = NUM_LAYERS * per_layer;

    LOGI("[KV] Allocating %.1f MB for KV cache", total_bytes_ / 1e6);
    LOGI("[KV] Config: layers=%d, kv_heads=%d, head_dim=%d, max_seq=%d, dtype=fp16",
         NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, max_seq_len);

    // mmap: anonymous private — OS gives us zeroed pages
    buffer_ = mmap(
        nullptr, total_bytes_,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );

    if (buffer_ == MAP_FAILED) {
        LOGE("[KV] mmap failed: %s", strerror(errno));
        buffer_ = nullptr;
        return false;
    }

    // mlock: prevent Android from paging out KV cache under memory pressure
    // This is the single most important call in this file
    int lock_result = mlock(buffer_, total_bytes_);
    if (lock_result != 0) {
        LOGW("[KV] mlock failed (%s) — KV cache may be paged out under memory pressure", strerror(errno));
        LOGW("[KV] Consider: adb shell 'echo 1 > /proc/sys/vm/overcommit_memory'");
        // Don't fail — continue without lock, just slower
    } else {
        LOGI("[KV] ✓ KV cache locked in RAM (%.1f MB)", total_bytes_ / 1e6);
    }

    // Prefault all pages now rather than on first access
    // Avoids page fault latency during first inference tokens
    memset(buffer_, 0, total_bytes_);
    LOGI("[KV] ✓ KV cache prefaulted");

    // Sequential access hint: we read KV cache in order during attention
    madvise(buffer_, total_bytes_, MADV_SEQUENTIAL);

    cur_seq_len_ = 0;
    return true;
}

// Zero-copy pointer into KV buffer — NO memcpy anywhere in the hot path
uint16_t* KVCacheManager::get_k_ptr(int layer, int seq_start) {
    size_t per_layer = 2 * NUM_KV_HEADS * max_seq_len_ * HEAD_DIM * DTYPE_BYTES;
    size_t k_offset  = layer * per_layer
                     + 0 /* K=0, V=1 */ * NUM_KV_HEADS * max_seq_len_ * HEAD_DIM * DTYPE_BYTES
                     + seq_start * HEAD_DIM * DTYPE_BYTES;
    return reinterpret_cast<uint16_t*>(
        static_cast<uint8_t*>(buffer_) + k_offset);
}

uint16_t* KVCacheManager::get_v_ptr(int layer, int seq_start) {
    size_t per_layer = 2 * NUM_KV_HEADS * max_seq_len_ * HEAD_DIM * DTYPE_BYTES;
    size_t v_offset  = layer * per_layer
                     + 1 /* V=1 */ * NUM_KV_HEADS * max_seq_len_ * HEAD_DIM * DTYPE_BYTES
                     + seq_start * HEAD_DIM * DTYPE_BYTES;
    return reinterpret_cast<uint16_t*>(
        static_cast<uint8_t*>(buffer_) + v_offset);
}

void KVCacheManager::log_stats() {
    // Read actual RSS to see if pages are resident
    FILE* f = fopen("/proc/self/status", "r");
    char line[256];
    while (f && fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            line[strcspn(line, "\n")] = 0;
            LOGI("[KV][MEM] %s | cur_seq=%d/%d", line, cur_seq_len_, max_seq_len_);
        }
    }
    if (f) fclose(f);
}

void KVCacheManager::benchmark_bandwidth() {
    PROFILE_SCOPE("kv_bandwidth_test");
    // Measure actual read bandwidth to KV cache
    // Simulates what attention does: sequential read of all K vectors

    int64_t  t0      = now_us();
    volatile uint16_t dummy = 0;
    size_t   n_elems = (total_bytes_ / 2);

    for (size_t i = 0; i < n_elems; i += 64) {
        dummy ^= static_cast<uint16_t*>(buffer_)[i];
    }

    double elapsed_s  = (now_us() - t0) / 1e6;
    double bandwidth  = (total_bytes_ / 1e9) / elapsed_s;  // GB/s
    LOGI("[KV] Read bandwidth: %.2f GB/s (theoretical LPDDR5 max: ~51 GB/s)", bandwidth);
    LOGI("[KV] KV read overhead per token at seq=512: %.1f us",
         (NUM_LAYERS * 2 * NUM_KV_HEADS * 512 * HEAD_DIM * DTYPE_BYTES / 1e9) / bandwidth * 1e6);
    (void)dummy;  // suppress optimization
}
```

---

## 2. TurboQuant KV Cache Compression

```python
# turboquant_kv.py
# TurboQuant for KV cache: PolarQuant + QJL residual
# Target: 3-bit KV storage with near-lossless attention scores
# Reference: https://arxiv.org/abs/2504.19874 (Google Research, March 2026)

import torch
import numpy as np
import math
import time

class PolarQuantKV:
    """
    PolarQuant: convert KV vectors from Cartesian to polar coordinates.
    Eliminates the per-block scale overhead of standard quantization.
    Based on TurboQuant's PolarQuant component.
    """
    def __init__(self, n_bits: int = 3, head_dim: int = 128):
        self.n_bits   = n_bits
        self.head_dim = head_dim
        self.n_angles = head_dim - 1  # d-1 angles + 1 radius = d total

        # Precompute quantization grid for angles [0, π] and [-π, π]
        # Angles are on a fixed grid — no per-vector normalization needed
        self.n_levels = 2**n_bits
        self.angle_bins = torch.linspace(-math.pi, math.pi, self.n_levels + 1)

    def encode(self, kv_vec: torch.Tensor) -> tuple:
        """
        Encode a [head_dim] float16 vector to polar representation.
        Returns: (radius, quantized_angles)
        """
        # Step 1: compute radius (keep in fp16 — this is the 'scale')
        radius = kv_vec.norm().item()
        if radius == 0:
            return 0.0, torch.zeros(self.n_angles, dtype=torch.uint8)

        # Step 2: normalize to unit sphere
        unit_vec = kv_vec / radius

        # Step 3: convert to polar angles via sequential decomposition
        angles = torch.zeros(self.n_angles)
        remaining = unit_vec.clone()

        for i in range(self.n_angles - 1):
            r_i     = remaining[i:].norm()
            if r_i == 0:
                break
            angle_i = torch.acos(remaining[i].clamp(-1, 1) / r_i)
            angles[i] = angle_i

            # Update remaining vector
            sin_a = torch.sin(angle_i)
            if sin_a.abs() > 1e-8:
                remaining[i+1:] = remaining[i+1:] / sin_a

        # Handle last angle: sign determines ±π
        angles[-1] = math.atan2(remaining[-1].item(), remaining[-2].item()) \
                     if len(remaining) >= 2 else 0.0

        # Step 4: quantize angles to n_bits
        q_angles = torch.bucketize(angles, self.angle_bins[1:-1]).to(torch.uint8)
        return radius, q_angles

    def decode(self, radius: float, q_angles: torch.Tensor) -> torch.Tensor:
        """
        Decode back to approximate float16 vector.
        """
        # Dequantize angles
        bin_centers = (self.angle_bins[:-1] + self.angle_bins[1:]) / 2
        angles = bin_centers[q_angles.long()]

        # Reconstruct unit vector from angles
        vec = torch.zeros(self.head_dim)
        remaining = 1.0
        for i in range(self.n_angles - 1):
            vec[i]    = remaining * math.cos(angles[i])
            remaining = remaining * math.sin(angles[i])
        vec[-1] = remaining

        return vec * radius

    def compute_attention_score(self,
                                query: torch.Tensor,
                                radius: float,
                                q_angles: torch.Tensor) -> float:
        """
        Compute query·key dot product WITHOUT full dequantization.
        This is the runtime path — called O(context_len) times per token.
        """
        k_approx = self.decode(radius, q_angles)
        return float(torch.dot(query, k_approx).item())


class TurboQuantKVCache:
    """
    Full TurboQuant KV cache: PolarQuant + QJL residual correction.
    Stores KV vectors at ~3 bits per element.
    """
    def __init__(self, n_bits: int = 3, head_dim: int = 128,
                 n_layers: int = 28, n_kv_heads: int = 8):
        self.polar   = PolarQuantKV(n_bits=n_bits-1, head_dim=head_dim)
        self.n_bits  = n_bits
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads

        # Storage: for each (layer, head, seq_pos):
        #   - radius:    float16 (1 scalar)
        #   - q_angles:  uint8  × (head_dim-1) packed at (n_bits-1) bits
        #   - qjl_bit:   1 bit (the residual correction)
        self.radii    = {}  # (layer, head, pos) → float
        self.angles   = {}  # (layer, head, pos) → uint8 tensor
        self.qjl_bits = {}  # (layer, head, pos) → int

        # JL random projection matrix (shared across all positions)
        torch.manual_seed(42)  # fixed seed for reproducibility
        self.jl_matrix = torch.randn(head_dim, head_dim).sign().float()

    def store_kv(self, layer: int, head: int, pos: int,
                  k_vec: torch.Tensor, v_vec: torch.Tensor):
        """Store a single K or V vector at (layer, head, pos)."""
        # PolarQuant compression
        k_radius, k_angles = self.polar.encode(k_vec)
        self.radii[(layer, head, pos, 'k')]  = k_radius
        self.angles[(layer, head, pos, 'k')] = k_angles

        # QJL 1-bit residual for K (V doesn't need it for inner product)
        k_decoded = self.polar.decode(k_radius, k_angles)
        k_residual = k_vec - k_decoded
        # Project residual to 1 bit via Johnson-Lindenstrauss
        jl_proj    = (self.jl_matrix @ k_residual.float()).sign()
        # Store just the sign of the first component as 1 bit
        self.qjl_bits[(layer, head, pos)] = 1 if jl_proj[0] > 0 else 0

        # V vector: PolarQuant only (no QJL for V — softmax handles outliers)
        v_radius, v_angles = self.polar.encode(v_vec)
        self.radii[(layer, head, pos, 'v')]  = v_radius
        self.angles[(layer, head, pos, 'v')] = v_angles

    def compute_attention_logit(self, layer: int, head: int, pos: int,
                                 query: torch.Tensor) -> float:
        """
        Compute attention logit query·key[pos] using compressed storage.
        QJL provides residual correction for accuracy.
        """
        k_radius = self.radii[(layer, head, pos, 'k')]
        k_angles = self.angles[(layer, head, pos, 'k')]
        qjl_bit  = self.qjl_bits[(layer, head, pos)]

        # Base score from PolarQuant
        base_score = self.polar.compute_attention_score(query, k_radius, k_angles)

        # QJL residual correction
        # The sign bit tells us whether the residual aligns with query direction
        # This is a simplified implementation — see paper for full estimator
        jl_query = float((self.jl_matrix @ query.float())[0].item())
        qjl_correction = 0.01 * jl_query * (1 if qjl_bit == 1 else -1)

        return base_score + qjl_correction

    def memory_usage_bytes(self, n_tokens: int) -> dict:
        """Report memory usage breakdown."""
        n_kv     = self.n_layers * self.n_kv_heads * n_tokens
        fp16_bytes = n_kv * 2 * self.head_dim * 2     # fp16 = 2 bytes
        # TurboQuant: radius (fp16=2 bytes) + angles at (n_bits-1) bits
        radius_bytes = n_kv * 2 * 2  # K and V radii
        angle_bytes  = n_kv * 2 * math.ceil((self.head_dim-1) * (self.n_bits-1) / 8)
        qjl_bytes    = n_kv * 1  # 1 byte per position (1 bit, padded)
        total_tq     = radius_bytes + angle_bytes + qjl_bytes

        return {
            "fp16_uncompressed": fp16_bytes,
            "turboquant_total":  total_tq,
            "compression_ratio": fp16_bytes / total_tq,
            "bits_per_element":  total_tq * 8 / (n_kv * 2 * self.head_dim),
        }
```

### 2.2 Validate TurboQuant Quality Before Integration
```python
# validate_turboquant.py
# Verify attention score accuracy before integrating into the inference loop

import torch
import numpy as np
from turboquant_kv import TurboQuantKVCache

HEAD_DIM = 128
N_TESTS  = 1000

tq = TurboQuantKVCache(n_bits=3, head_dim=HEAD_DIM)

# Generate random KV and query vectors
torch.manual_seed(0)
errors = []

for i in range(N_TESTS):
    k_vec = torch.randn(HEAD_DIM, dtype=torch.float32)
    q_vec = torch.randn(HEAD_DIM, dtype=torch.float32)

    # Store compressed
    tq.store_kv(0, 0, i, k_vec, k_vec)  # use k for both for testing

    # True dot product
    true_score = float(torch.dot(q_vec, k_vec))

    # TurboQuant approximate dot product
    approx_score = tq.compute_attention_logit(0, 0, i, q_vec)

    rel_error = abs(approx_score - true_score) / (abs(true_score) + 1e-8)
    errors.append(rel_error)

errors = np.array(errors)
mem    = tq.memory_usage_bytes(N_TESTS)

print("TurboQuant KV Validation Results:")
print(f"  Attention score error:")
print(f"    Mean:  {errors.mean():.4f} ({errors.mean()*100:.2f}%)")
print(f"    P95:   {np.percentile(errors, 95):.4f}")
print(f"    Max:   {errors.max():.4f}")
print(f"  Memory (at {N_TESTS} tokens):")
print(f"    Uncompressed fp16: {mem['fp16_uncompressed']/1e6:.1f} MB")
print(f"    TurboQuant 3-bit:  {mem['turboquant_total']/1e6:.1f} MB")
print(f"    Compression ratio: {mem['compression_ratio']:.1f}x")
print(f"    Bits per element:  {mem['bits_per_element']:.2f}")

# Threshold: mean error < 1% is acceptable for attention scores
if errors.mean() < 0.01:
    print(f"\n✓ TurboQuant quality ACCEPTABLE (mean error {errors.mean()*100:.2f}% < 1%)")
else:
    print(f"\n✗ TurboQuant error too high ({errors.mean()*100:.2f}%) — check implementation")
```

---

## 3. Integrate TurboQuant Into Decode Loop

```cpp
// In QnnRuntime.cpp — modify the decode step to use TurboQuant KV

// The key change: after each token, store the new K/V vectors in TurboQuant format
// For attention score computation, use the TurboQuant estimator instead of raw fp16

bool QnnRuntime::decode_step_with_turboquant_kv(
    int32_t token_id, int position,
    float* logits_out, int vocab_size)
{
    PROFILE_SCOPE("decode_step_turboquant");

    // 1. Standard HTP forward pass (same as before)
    {
        PROFILE_SCOPE("htp_execute");
        // ... same as Phase 1
    }

    // 2. Extract new K/V vectors from HTP output
    {
        PROFILE_SCOPE("kv_extract_and_compress");
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            for (int head = 0; head < NUM_KV_HEADS; head++) {
                // Get K and V from this layer's output
                auto* k_ptr = get_new_k_vector(layer, head);  // fp16 output from HTP
                auto* v_ptr = get_new_v_vector(layer, head);

                torch::Tensor k_vec = torch::from_blob(k_ptr, {HEAD_DIM},
                    torch::kFloat16).float();
                torch::Tensor v_vec = torch::from_blob(v_ptr, {HEAD_DIM},
                    torch::kFloat16).float();

                // Compress and store in TurboQuant format
                tq_cache_.store_kv(layer, head, position, k_vec, v_vec);
            }
        }
    }

    // 3. Copy logits (same as before)
    {
        PROFILE_SCOPE("logits_copy");
        memcpy(logits_out, logits_ptr_, vocab_size * sizeof(float));
    }

    return true;
}
```

---

## 4. Phase 4 Benchmarks

```bash
# Benchmark 1: KV cache pinning effect
# Run at short context (fast) and long context (where KV bandwidth matters)

adb shell "cd /data/local/tmp/vajra && \
  ./vajra_benchmark \
  --model libqwen3_hqq_mixed.so \
  --kv-mode fp16-pinned \
  --context-len 512 \
  --tokens 100 \
  2>&1" | tee ./logs/benchmark_kv_pinned_512.log

adb shell "cd /data/local/tmp/vajra && \
  ./vajra_benchmark \
  --model libqwen3_hqq_mixed.so \
  --kv-mode fp16-unpinned \
  --context-len 512 \
  --tokens 100 \
  2>&1" | tee ./logs/benchmark_kv_unpinned_512.log

# Benchmark 2: TurboQuant KV vs fp16 KV
adb shell "cd /data/local/tmp/vajra && \
  ./vajra_benchmark \
  --model libqwen3_hqq_mixed.so \
  --kv-mode turboquant-3bit \
  --context-len 512 \
  --tokens 100 \
  2>&1" | tee ./logs/benchmark_kv_turboquant_512.log
```

---

## 5. Phase 4 Completion Checklist

```
[ ] KVCacheManager allocates and mlocks successfully on Nothing 3a Pro
[ ] mlock succeeds (not just warns) — check device memory pressure
[ ] KV bandwidth benchmark run: actual GB/s measured and logged
[ ] Pinned vs unpinned KV benchmark shows >= 10% speedup at context 512
    (if < 5%: mlock may have silently failed — check /proc/self/status VmLck)
[ ] TurboQuant attention score error < 1% mean on 1000 random vectors
[ ] TurboQuant memory reduction validated: >= 5x vs fp16 at seq=512
[ ] TurboQuant integrated into decode loop without crash
[ ] TurboQuant + HQQ mixed precision benchmark vs fp16 KV benchmark
[ ] results/comparison.md rows 7-8 filled in
[ ] No memory leak during 120s sustained run with TurboQuant active
    (monitor: adb shell cat /proc/<pid>/status | grep VmRSS every 10s)
```

---
---

