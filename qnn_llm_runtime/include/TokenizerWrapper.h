// ============================================================================
// TokenizerWrapper.h — SentencePiece Tokenizer for On-Device Inference
// ============================================================================
//
// PURPOSE:
//   Wraps the SentencePiece library to tokenize prompts and decode generated
//   token IDs back to text on the device. This runs on the CPU (not HTP)
//   and is NOT on the critical inference path — it runs once during prefill
//   and once per token during output decoding.
//
// QWEN3-1.7B TOKENIZER:
//   - Type: BPE (Byte Pair Encoding) via SentencePiece
//   - Vocab size: 151,936 tokens
//   - Special tokens: BOS=151643, EOS=151645, PAD=151643
//   - Model file: tokenizer.model (SentencePiece binary format)
//
// BUILD DEPENDENCY:
//   Link against libsentencepiece.a (statically compiled for aarch64-android).
//   Build instructions:
//     git clone https://github.com/google/sentencepiece
//     cd sentencepiece && mkdir build-android && cd build-android
//     cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/.../android.toolchain.cmake \
//              -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-33 \
//              -DSPM_ENABLE_SHARED=OFF
//     make -j$(nproc)
//
// ============================================================================

#pragma once

#include "Profiler.h"
#include <string>
#include <vector>
#include <cstdint>

namespace halfhex {

class TokenizerWrapper {
public:
    TokenizerWrapper() = default;
    ~TokenizerWrapper();

    // Non-copyable (owns SentencePiece processor).
    TokenizerWrapper(const TokenizerWrapper&) = delete;
    TokenizerWrapper& operator=(const TokenizerWrapper&) = delete;

    // Load tokenizer model from file path.
    // Returns false if the file doesn't exist or is invalid.
    bool load(const std::string& model_path);

    // Encode text to token IDs.
    std::vector<int32_t> encode(const std::string& text);

    // Decode token IDs to text.
    std::string decode(const std::vector<int32_t>& ids);

    // Decode a single token ID to its string representation.
    std::string decode_token(int32_t id);

    // Vocabulary size.
    int vocab_size() const { return vocab_size_; }

    // Special token IDs.
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }

    // Check if loaded.
    bool is_loaded() const { return loaded_; }

private:
    void* sp_processor_ = nullptr;  // sentencepiece::SentencePieceProcessor*
    int   vocab_size_   = 0;
    int32_t bos_id_     = 151643;   // Qwen3 default
    int32_t eos_id_     = 151645;   // Qwen3 default
    int32_t pad_id_     = 151643;   // Qwen3 default
    bool    loaded_     = false;
};

} // namespace halfhex
