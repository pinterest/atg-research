#include "tokenization/bert_tokenizer/wordpiece_tokenizer.h"

#ifdef GLOG_ENABLED
#include "common/log.h"
#endif
#include <vector>

#include "tokenization/bert_tokenizer/tokenization_bert_utils.h"
#include "unicode/schriter.h"

namespace tokenization {

std::vector<icu::UnicodeString> WordpieceTokenizer::Tokenize(
    const icu::UnicodeString& text) const {
  const std::vector<icu::UnicodeString> split_tokens = WhitespaceTokenize(text);
  std::vector<icu::UnicodeString> output;
  output.reserve(split_tokens.size());
  for (const auto& token : split_tokens) {
    if (token.length() > max_input_chars_per_word_) {
      output.push_back(unk_token_);
      continue;
    }
    bool is_bad = false;
    int start = 0;
    std::vector<icu::UnicodeString> sub_tokens;
    while (start < token.length()) {
      icu::UnicodeString cur_substr;
      icu::UnicodeString substr;

      int end = token.length();
      while (start < end) {
        token.extractBetween(start, end, substr);
        if (start > 0) {
          substr = "##" + substr;
        }

        if (vocab_.find(substr) != vocab_.end()) {
          cur_substr = substr;
          break;
        }
        --end;
      }

      if (cur_substr.isEmpty()) {
        is_bad = true;
        break;
      }
      sub_tokens.push_back(cur_substr);
      start = end;
    }
    if (is_bad) {
      output.push_back(unk_token_);
    } else {
      output.insert(output.end(), sub_tokens.begin(), sub_tokens.end());
    }
  }
  return output;
}
}  // namespace tokenization
