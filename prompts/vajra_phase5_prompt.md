# VAJRA — Phase 5: Speculative Decoding + Full Benchmark + Demo
## Prerequisite: Phases 1-4 complete. All comparison.md rows through row 8 filled.

> Exit criteria for Phase 5:
> Final demo-ready artifact: a video of Qwen3-1.7B running on Nothing Phone 3a Pro
> at > 15 tok/s sustained, with full profiling output visible.
> GitHub repo public. Blog post drafted.

---

## 0. Phase 5 Mindset

This phase makes the project presentable to the world.
Speculative decoding adds throughput on top of everything you've built.
The benchmark harness makes your numbers reproducible by anyone.
The demo artifact is what gets you hired.

Do not rush Phase 5. A polished demo with clean numbers
is worth more than additional optimization that isn't documented.

---

## 1. Speculative Decoding Setup

Speculative decoding uses a small draft model to propose multiple tokens,
then verifies them in parallel with the main model.
For your hardware: Qwen3-0.6B as drafter, Qwen3-1.7B as verifier.

The math: if the draft model's token acceptance rate is α and
you draft K tokens per step, effective throughput = (1 + K*α) × verifier_tps.
For typical chat (α ≈ 0.7, K=3): 3.1x theoretical max.
Real-world on edge hardware: expect 1.5-2x.

### 1.1 Prepare Draft Model (Qwen3-0.6B)
```bash
# Repeat Phase 1 + Phase 2 process for Qwen3-0.6B
# But abbreviated — INT4 HQQ only, no mixed precision needed for draft model

python3 << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    local_dir="./models/qwen3-0.6b-hf",
    ignore_patterns=["*.gguf", "*.bin"]
)
EOF

# Export to ONNX
python export_onnx.py --model ./models/qwen3-0.6b-hf --output ./models/qwen3-0.6b-onnx

# HQQ INT4
python hqq_quantize.py --model ./models/qwen3-0.6b-hf --output ./models/qwen3-0.6b-hqq-int4

# QNN convert
qnn-onnx-converter \
  --input_network ./models/qwen3-0.6b-onnx/model_sim.onnx \
  --output_path   ./models/qwen3-0.6b-qnn-int4/qwen3_draft.cpp \
  --act_bw 8 --weight_bw 4 --bias_bw 32 \
  --quantization_overrides ./quantization_config_hqq_int4_06b.json \
  --verbose 2>&1 | tee ./logs/qnn_draft_conversion.log

qnn-model-lib-generator \
  -c ./models/qwen3-0.6b-qnn-int4/qwen3_draft.cpp \
  -b ./models/qwen3-0.6b-qnn-int4/qwen3_draft.bin \
  -o ./models/qwen3-0.6b-qnn-int4/ \
  -t aarch64-android
```

### 1.2 Speculative Decode Implementation
```cpp
// SpeculativeDecoder.cpp
// Draft K tokens with 0.6B, verify with 1.7B in one forward pass

class SpeculativeDecoder {
public:
    SpeculativeDecoder(QnnRuntime* draft, QnnRuntime* verifier,
                       int k = 3)
        : draft_(draft), verifier_(verifier), K_(k) {}

    // Returns accepted tokens (may be fewer than K if early rejection)
    std::vector<int32_t> step(
        const std::vector<int32_t>& context,
        int position)
    {
        PROFILE_SCOPE("speculative_step_total");
        std::vector<int32_t> candidates;
        std::vector<float>   draft_probs;

        // ── Draft K tokens with small model ──────────────────────────
        {
            PROFILE_SCOPE("draft_K_tokens");
            auto cur_context = context;
            for (int k = 0; k < K_; k++) {
                std::vector<float> logits(VOCAB_SIZE);
                draft_->decode_step(
                    cur_context.back(), position + k,
                    logits.data(), VOCAB_SIZE);

                // Sample from draft distribution
                auto probs    = softmax(logits);
                int  token    = sample(probs);
                float tok_prob = probs[token];

                candidates.push_back(token);
                draft_probs.push_back(tok_prob);
                cur_context.push_back(token);
            }
        }

        // ── Verify all K+1 positions in ONE verifier forward pass ────
        // This is the key insight: verifier processes K tokens in parallel
        std::vector<std::vector<float>> verifier_logits(K_ + 1,
            std::vector<float>(VOCAB_SIZE));
        {
            PROFILE_SCOPE("verifier_forward_pass");
            // Verifier processes: [context[-1], candidate[0], ..., candidate[K-1]]
            std::vector<int32_t> verify_input = {context.back()};
            verify_input.insert(verify_input.end(), candidates.begin(), candidates.end());

            for (int i = 0; i < (int)verify_input.size(); i++) {
                verifier_->decode_step(
                    verify_input[i], position + i,
                    verifier_logits[i].data(), VOCAB_SIZE);
            }
        }

        // ── Accept/reject with speculative sampling ───────────────────
        {
            PROFILE_SCOPE("acceptance_sampling");
            std::vector<int32_t> accepted;

            for (int k = 0; k < K_; k++) {
                auto verifier_probs = softmax(verifier_logits[k]);
                float p_verifier = verifier_probs[candidates[k]];
                float p_draft    = draft_probs[k];

                float accept_prob = std::min(1.0f, p_verifier / p_draft);
                float u = uniform_random();

                if (u <= accept_prob) {
                    accepted.push_back(candidates[k]);
                } else {
                    // Rejection: sample a correction token from adjusted distribution
                    auto adj = adjusted_distribution(verifier_probs,
                        softmax(draft_logits_for(k)), candidates[k]);
                    accepted.push_back(sample(adj));
                    break;  // stop at first rejection
                }
            }

            // Always accept at least one token from verifier's last position
            if ((int)accepted.size() == K_) {
                accepted.push_back(sample(softmax(verifier_logits[K_])));
            }

            int accepted_n = accepted.size();
            PROFILE_LOG("speculative: accepted %d/%d drafts (%.0f%%)",
                        accepted_n, K_+1, 100.0f*accepted_n/(K_+1));
            return accepted;
        }
    }

    float get_acceptance_rate() const {
        return total_accepted_ / (float)total_proposed_;
    }

private:
    QnnRuntime* draft_;
    QnnRuntime* verifier_;
    int K_;
    int total_accepted_ = 0;
    int total_proposed_ = 0;
};
```

---

## 2. Final Benchmark Harness

```bash
# benchmark_final.sh — the definitive benchmark script
# This is what you run before publishing numbers
# Do not change parameters between runs

#!/bin/bash
set -e
DATE=$(date +%Y%m%d_%H%M%S)
LOG="./results/final_benchmark_${DATE}.log"

echo "VAJRA Final Benchmark - ${DATE}" | tee $LOG
echo "Device: Nothing Phone 3a Pro (Snapdragon 7s Gen 3 / Hexagon V73)" | tee -a $LOG
echo "Model: Qwen3-1.7B (HQQ INT4 mixed precision)" | tee -a $LOG
echo "Draft: Qwen3-0.6B (HQQ INT4)" | tee -a $LOG

# Device state
echo "--- Device State ---" | tee -a $LOG
adb shell getprop ro.product.model | tee -a $LOG
adb shell cat /sys/class/thermal/thermal_zone0/temp | tee -a $LOG
adb shell free -h | head -2 | tee -a $LOG

# Set performance mode
adb shell "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; \
  do echo performance > \$f; done"

# Run 5 full trials — report median
for trial in 1 2 3 4 5; do
  echo "--- Trial ${trial}/5 ---" | tee -a $LOG
  adb shell "cd /data/local/tmp/vajra && \
    LD_LIBRARY_PATH=. \
    ./vajra_benchmark_final \
    --verifier libqwen3_hqq_mixed.so \
    --draft    libqwen3_0.6b_int4.so \
    --mode speculative \
    --draft-k 3 \
    --warmup-s 60 \
    --sustained-s 120 \
    --prompt-tokens 128 \
    --output-tokens 200 \
    --log-level verbose \
    2>&1" | tee -a $LOG
done

# Pull profile CSVs
adb pull /sdcard/vajra_final_*.csv ./results/ 2>/dev/null || true

echo "=== COMPLETE ===" | tee -a $LOG
echo "Analyze with: python analyze_final.py ./results/final_benchmark_${DATE}.log" | tee -a $LOG
```

---

## 3. Fill the Comparison Table

```markdown
# results/comparison.md

| Config                                | Prefill (tok/s) | Decode (tok/s) | Sustained 120s | TTFT (128 tok) |
|---------------------------------------|-----------------|----------------|----------------|----------------|
| llama.cpp Q4_0 CPU (4 threads)        |                 |                |                |                |
| llama.cpp Q4_0 Vulkan (Adreno 710)    |                 |                |                |                |
| QNN INT8 (standard PTQ)               |                 |                |                |                |
| QNN PTQ INT4                          |                 |                |                |                |
| QNN HQQ INT4 (gs=64)                  |                 |                |                |                |
| QNN HQQ Mixed (INT4+INT8 per layer)   |                 |                |                |                |
| QNN HQQ Mixed + KV pinned             |                 |                |                |                |
| QNN HQQ Mixed + TurboQuant KV (3-bit) |                 |                |                |                |
| + Speculative decode (K=3, 0.6B draft)|                 |                |                |                |
```

---

## 4. Demo Artifact Requirements

```
Your demo must contain ALL of the following:

1. SCREEN RECORDING (2-3 minutes)
   - Start: show device (nothing special, room temperature)
   - adb logcat filtered to VAJRA tag visible on screen
   - Run: ./vajra_benchmark_final with live token output
   - Show tokens streaming with tok/s rate updating in logcat
   - End: show thermal state and sustained number

2. GITHUB REPO (public, clean)
   Directory structure:
   vajra/
   ├── README.md          ← clear build instructions, results table
   ├── phases/
   │   ├── phase1/        ← QNN INT8 baseline
   │   ├── phase2/        ← HQQ INT4 bridge
   │   ├── phase3/        ← mixed precision
   │   ├── phase4/        ← KV cache + TurboQuant
   │   └── phase5/        ← speculative decode
   ├── scripts/
   │   ├── export_onnx.py
   │   ├── hqq_quantize.py
   │   ├── extract_hqq_params.py
   │   ├── measure_perplexity.py
   │   └── analyze_profile.py
   ├── results/
   │   ├── comparison.md  ← the table, filled in completely
   │   ├── plots/         ← all generated charts
   │   └── baselines/     ← llama.cpp baseline files
   └── tools/
       └── run_benchmarks.sh

3. BLOG POST
   Title: "Running Qwen3-1.7B at Xtok/s on a ₹20,000 Android Phone:
           HQQ + Hexagon HTP Deep Dive"
   Must include:
   - The comparison table
   - The throughput timeline chart (shows thermal behaviour)
   - The layer breakdown chart (shows where time goes)
   - The perplexity comparison table (INT8 vs INT4-HQQ vs mixed)
   - Code snippets: HQQ bridge (the novel part)
   Length: 2000-3000 words. Technical. No hype.

4. POSTING SCHEDULE
   Day 1: GitHub repo goes public
   Day 2: Blog post on personal site / Medium / dev.to
   Day 3: HN post (Show HN: ...) + r/LocalLLaMA + X/Twitter
```

---

## 5. Phase 5 Completion Checklist

```
[ ] Qwen3-0.6B draft model converted to QNN INT4 and benchmarked
[ ] Acceptance rate measured: if < 0.5 with K=3, reduce K to 2
[ ] Speculative decode: effective tok/s > solo verifier tok/s
[ ] comparison.md: ALL rows filled in with real numbers
[ ] Screen recording: clearly shows tok/s number > llama.cpp baseline
[ ] GitHub repo: README has build instructions, at least 3 people can follow them
[ ] Blog post: comparison table matches comparison.md exactly
[ ] arXiv paper (if Phase 2 HQQ result was notable): draft written
    Title: "HQQ-Initialized Quantization for Hexagon HTP:
            Calibration-Free INT4 LLM Inference on Snapdragon Edge Devices"
    Sections: Abstract, Intro, Background (HQQ + QNN), Method (bridge layer),
              Experiments, Results, Conclusion

FINAL GATE: Can a stranger clone your repo and reproduce your numbers
on the same hardware? If yes: ship it. If no: keep working.
```
