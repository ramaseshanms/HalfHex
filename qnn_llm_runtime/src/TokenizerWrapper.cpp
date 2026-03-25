// ============================================================================
// TokenizerWrapper.cpp — SentencePiece Tokenizer Implementation
// ============================================================================
//
// Uses Google SentencePiece for BPE tokenization (Qwen3 uses this format).
// The sentencepiece library is cross-compiled as a static library
// (libsentencepiece.a) for Android arm64 and linked at build time.
//
// Qwen3-1.7B tokenizer specifics:
//   - Vocab size: 151,936
//   - BOS token: 151643
//   - EOS token: 151645 (<|endoftext|>)
//   - Model file: tokenizer.model (SentencePiece binary format)
//
// ============================================================================

#include "TokenizerWrapper.h"

#ifdef HALFHEX_HAS_SENTENCEPIECE
#include "sentencepiece_processor.h"
#endif

namespace halfhex {

TokenizerWrapper::~TokenizerWrapper() {
#ifdef HALFHEX_HAS_SENTENCEPIECE
    if (sp_processor_) {
        delete static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
        sp_processor_ = nullptr;
    }
#endif
    loaded_ = false;
}

bool TokenizerWrapper::load(const std::string& model_path) {
    PROFILE_SCOPE("tokenizer_load");

#ifdef HALFHEX_HAS_SENTENCEPIECE
    auto* sp = new sentencepiece::SentencePieceProcessor();
    auto status = sp->Load(model_path);
    if (!status.ok()) {
        LOGE("[TOKENIZER] Failed to load: %s", status.ToString().c_str());
        delete sp;
        return false;
    }

    sp_processor_ = sp;
    vocab_size_ = sp->GetPieceSize();

    // Qwen3 uses custom special token IDs that differ from SentencePiece defaults.
    bos_id_ = 151643;
    eos_id_ = 151645;
    pad_id_ = 151643;

    loaded_ = true;
    LOGI("[TOKENIZER] Loaded from %s (vocab_size=%d, eos=%d)",
         model_path.c_str(), vocab_size_, eos_id_);
    return true;
#else
    // SentencePiece not linked — stub mode for testing without tokenizer.
    LOGW("[TOKENIZER] SentencePiece not available (built without HALFHEX_HAS_SENTENCEPIECE)");
    LOGW("[TOKENIZER] Tokenizer operations will return empty results");
    loaded_ = true;
    LOGI("[TOKENIZER] Loaded in stub mode from %s", model_path.c_str());
    return true;
#endif
}

std::vector<int32_t> TokenizerWrapper::encode(const std::string& text) {
    PROFILE_SCOPE("tokenizer_encode");

    std::vector<int32_t> ids;

#ifdef HALFHEX_HAS_SENTENCEPIECE
    if (sp_processor_) {
        auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
        std::vector<int> pieces;
        auto status = sp->Encode(text, &pieces);
        if (!status.ok()) {
            LOGE("[TOKENIZER] Encode failed: %s", status.ToString().c_str());
            return {};
        }
        ids.reserve(pieces.size());
        for (int p : pieces) ids.push_back(static_cast<int32_t>(p));
    }
#endif

    LOGD("[TOKENIZER] Encoded %zu chars -> %zu tokens", text.size(), ids.size());
    return ids;
}

std::string TokenizerWrapper::decode(const std::vector<int32_t>& ids) {
    PROFILE_SCOPE("tokenizer_decode");

    std::string text;

#ifdef HALFHEX_HAS_SENTENCEPIECE
    if (sp_processor_) {
        auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
        std::vector<int> pieces(ids.begin(), ids.end());
        auto status = sp->Decode(pieces, &text);
        if (!status.ok()) {
            LOGE("[TOKENIZER] Decode failed: %s", status.ToString().c_str());
            return "";
        }
    }
#endif

    return text;
}

std::string TokenizerWrapper::decode_token(int32_t id) {
    return decode({id});
}

} // namespace halfhex
