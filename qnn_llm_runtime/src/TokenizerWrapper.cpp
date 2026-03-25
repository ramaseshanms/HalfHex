// ============================================================================
// TokenizerWrapper.cpp — SentencePiece Tokenizer Implementation
// ============================================================================
//
// STATUS: Scaffold with profiling hooks.
// SentencePiece API calls are commented out because the library must be
// cross-compiled for aarch64-android separately. See TokenizerWrapper.h
// for build instructions.
//
// HOW TO WIRE UP:
//   1. Build sentencepiece for Android (see header comments)
//   2. Add to CMakeLists.txt: target_link_libraries(... sentencepiece)
//   3. Uncomment the #include and API calls below
//
// ============================================================================

#include "TokenizerWrapper.h"

// Uncomment when sentencepiece is available:
// #include "sentencepiece_processor.h"

namespace halfhex {

TokenizerWrapper::~TokenizerWrapper() {
    // if (sp_processor_) {
    //     delete static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
    //     sp_processor_ = nullptr;
    // }
    loaded_ = false;
}

bool TokenizerWrapper::load(const std::string& model_path) {
    PROFILE_SCOPE("tokenizer_load");

    // TODO: Uncomment when sentencepiece is available:
    //
    // auto* sp = new sentencepiece::SentencePieceProcessor();
    // auto status = sp->Load(model_path);
    // if (!status.ok()) {
    //     LOGE("[TOKENIZER] Failed to load: %s", status.ToString().c_str());
    //     delete sp;
    //     return false;
    // }
    //
    // sp_processor_ = sp;
    // vocab_size_ = sp->GetPieceSize();
    //
    // // Qwen3 uses custom special token IDs.
    // // Override SentencePiece defaults with Qwen3 config values.
    // bos_id_ = 151643;
    // eos_id_ = 151645;
    // pad_id_ = 151643;

    loaded_ = true;
    LOGI("[TOKENIZER] Loaded from %s (vocab_size=%d, eos=%d)",
         model_path.c_str(), vocab_size_, eos_id_);
    return true;
}

std::vector<int32_t> TokenizerWrapper::encode(const std::string& text) {
    PROFILE_SCOPE("tokenizer_encode");

    std::vector<int32_t> ids;

    // TODO: Uncomment when sentencepiece is available:
    //
    // if (sp_processor_) {
    //     auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
    //     std::vector<int> pieces;
    //     auto status = sp->Encode(text, &pieces);
    //     if (!status.ok()) {
    //         LOGE("[TOKENIZER] Encode failed: %s", status.ToString().c_str());
    //         return {};
    //     }
    //     ids.reserve(pieces.size());
    //     for (int p : pieces) ids.push_back(static_cast<int32_t>(p));
    // }

    LOGD("[TOKENIZER] Encoded %zu chars -> %zu tokens", text.size(), ids.size());
    return ids;
}

std::string TokenizerWrapper::decode(const std::vector<int32_t>& ids) {
    PROFILE_SCOPE("tokenizer_decode");

    std::string text;

    // TODO: Uncomment when sentencepiece is available:
    //
    // if (sp_processor_) {
    //     auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
    //     std::vector<int> pieces(ids.begin(), ids.end());
    //     auto status = sp->Decode(pieces, &text);
    //     if (!status.ok()) {
    //         LOGE("[TOKENIZER] Decode failed: %s", status.ToString().c_str());
    //         return "";
    //     }
    // }

    return text;
}

std::string TokenizerWrapper::decode_token(int32_t id) {
    return decode({id});
}

} // namespace halfhex
