#include "tokenization/bert_tokenizer/bert_tokenizer.h"

#include "tokenization/bert_tokenizer/basic_tokenizer.h"
#include "tokenization/bert_tokenizer/wordpiece_tokenizer.h"
#include <algorithm>
#include <utility>

namespace {
/**
 * Special tokens for the tokenizer
 */
const char* const kClsToken = "[CLS]";
const char* const kUnkToken = "[UNK]";
const char* const kSepToken = "[SEP]";
const char* const kPadToken = "[PAD]";
}  // namespace

namespace tokenization {

BertTokenizer::BertTokenizer(
    const VocabT& vocab, int64_t max_sequence_length, bool do_lower_case,
    bool tokenize_chinese_chars,
    const tokenization::BasicTokenizer::TokenSetT& never_split)
    : vocab_(vocab),
      max_sequence_length_(max_sequence_length),
      unk_token_id_(TokenToId(kUnkToken, true)),
      sep_token_id_(TokenToId(kSepToken, true)),
      cls_token_id_(TokenToId(kClsToken, true)),
      pad_token_id_(TokenToId(kPadToken, true)),
      basic_tokenizer_(do_lower_case, tokenize_chinese_chars, never_split),
      wordpiece_tokenizer_(vocab_, kUnkToken) {}

int64_t BertTokenizer::TokenToId(const icu::UnicodeString& token,
                                 bool errorIfMissing) const {
  const auto it = vocab_.find(token);
  if (it != vocab_.end()) return it->second;

  if (!errorIfMissing) return unk_token_id_;

  std::string tok;
  token.toUTF8String(tok);
#ifdef GLOG_ENABLED
  LOG(ERROR) << "Missing token in vocabulary: " << tok;
#endif
  throw std::runtime_error("Missing token in vocabulary: " + tok);
}

std::vector<std::int64_t> BertTokenizer::Encode(
    const icu::UnicodeString& text) const {
  const std::vector<icu::UnicodeString> basic_tokenized =
      basic_tokenizer_.Tokenize(text);
  std::vector<std::int64_t> out;
  out.reserve(std::min(static_cast<int64_t>(basic_tokenized.size()) * 2,
                       max_sequence_length_));
  out.push_back(cls_token_id_);
  for (const auto& token : basic_tokenized) {
    if (basic_tokenizer_.IsNeverSplitToken(token)) {
      out.push_back(TokenToId(token));
    } else {
      const std::vector<icu::UnicodeString> wps =
          wordpiece_tokenizer_.Tokenize(token);
      for (const auto& wp : wps) {
        out.push_back(TokenToId(wp));
      }
    }
    size_t size_limit = std::max(0L, max_sequence_length_ - 1);
    if (out.size() > size_limit) {
      out.resize(size_limit);
      break;
    }
  }
  out.push_back(sep_token_id_);
  return out;
}
}  // namespace tokenization
