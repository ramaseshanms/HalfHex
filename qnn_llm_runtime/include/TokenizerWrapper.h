// TokenizerWrapper.h — SentencePiece tokenizer wrapper for on-device inference
#pragma once

#include <string>
#include <vector>
#include <cstdint>

class TokenizerWrapper {
public:
    TokenizerWrapper() = default;
    ~TokenizerWrapper();

    // Load tokenizer model file (sentencepiece .model or tokenizer.json)
    bool load(const std::string& model_path);

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text);

    // Decode token IDs back to text
    std::string decode(const std::vector<int32_t>& ids);

    // Decode a single token ID
    std::string decode_token(int32_t id);

    // Get vocabulary size
    int vocab_size() const { return vocab_size_; }

    // Special token IDs
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }

private:
    void* sp_processor_ = nullptr;  // sentencepiece::SentencePieceProcessor*
    int vocab_size_ = 0;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t pad_id_ = 0;
};
