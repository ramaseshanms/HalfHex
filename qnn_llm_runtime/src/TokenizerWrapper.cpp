// TokenizerWrapper.cpp — SentencePiece tokenizer wrapper
#include "TokenizerWrapper.h"
#include "Profiler.h"

// In production, link against sentencepiece library:
// #include "sentencepiece_processor.h"

TokenizerWrapper::~TokenizerWrapper() {
    // if (sp_processor_) {
    //     delete static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
    //     sp_processor_ = nullptr;
    // }
}

bool TokenizerWrapper::load(const std::string& model_path) {
    PROFILE_SCOPE("tokenizer_load");

    // sentencepiece::SentencePieceProcessor* sp = new sentencepiece::SentencePieceProcessor();
    // auto status = sp->Load(model_path);
    // if (!status.ok()) {
    //     LOGE("[TOKENIZER] Failed to load model: %s", status.ToString().c_str());
    //     delete sp;
    //     return false;
    // }
    // sp_processor_ = sp;
    // vocab_size_ = sp->GetPieceSize();
    // bos_id_ = sp->bos_id();
    // eos_id_ = sp->eos_id();
    // pad_id_ = sp->pad_id();

    LOGI("[TOKENIZER] Loaded model from %s (vocab_size=%d)", model_path.c_str(), vocab_size_);
    return true;
}

std::vector<int32_t> TokenizerWrapper::encode(const std::string& text) {
    PROFILE_SCOPE("tokenizer_encode");

    std::vector<int32_t> ids;
    // if (sp_processor_) {
    //     auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
    //     std::vector<int> pieces;
    //     sp->Encode(text, &pieces);
    //     ids.assign(pieces.begin(), pieces.end());
    // }

    LOGI("[TOKENIZER] Encoded %zu chars → %zu tokens", text.size(), ids.size());
    return ids;
}

std::string TokenizerWrapper::decode(const std::vector<int32_t>& ids) {
    PROFILE_SCOPE("tokenizer_decode");

    std::string text;
    // if (sp_processor_) {
    //     auto* sp = static_cast<sentencepiece::SentencePieceProcessor*>(sp_processor_);
    //     std::vector<int> pieces(ids.begin(), ids.end());
    //     sp->Decode(pieces, &text);
    // }
    return text;
}

std::string TokenizerWrapper::decode_token(int32_t id) {
    return decode({id});
}
