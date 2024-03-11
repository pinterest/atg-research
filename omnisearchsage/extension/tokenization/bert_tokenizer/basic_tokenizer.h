#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tokenization/bert_tokenizer/tokenization_bert_utils.h"
#include "unicode/unistr.h"

namespace tokenization {

/**
 * Implementation of the BertTokenizer.BasicTokenizer from huggingface
 * transformers library.
 */
class BasicTokenizer {
 public:
  using TokenSetT = absl::flat_hash_set<icu::UnicodeString, Hasher>;
  /**
   * Constructs a BasicTokenizer.
   *
   * @param do_lower_case Whether to lower case the input.
   * @param tokenize_chinese_chars Whether to Tokenize Chinese characters. This
   * should likely be deactivated for Japanese.
   * @param never_split Tokens to avoid splitting.
   */
  BasicTokenizer(bool do_lower_case, bool tokenize_chinese_chars,
                 const TokenSetT& never_split)
      : never_split_(never_split),
        do_lower_case_(do_lower_case),
        tokenize_chinese_chars_(tokenize_chinese_chars) {}

  BasicTokenizer(BasicTokenizer const&) = delete;
  BasicTokenizer& operator=(BasicTokenizer const&) = delete;

  /**
   * Basic Tokenization of a piece of text. Splits on "white spaces" only; for
   * sub-word tokenization, see WordpieceTokenizer.
   */
  std::vector<icu::UnicodeString> Tokenize(
      const icu::UnicodeString& text) const;
#if __cplusplus >= 202001L  // C++20
  // C++20 introduces type char8_t and changes the type of string literal
  // u8"foo" from const char[] to const char8_t[]. This is a breaking change.
  // Define this overload to allow Tokenize(u8"foo") to work with C++20.
  std::vector<icu::UnicodeString> Tokenize(const char8_t* text) const {
    return Tokenize(std::reinterpret_cast<const char*>(text));
  }
#endif

  /**
   * Same as `Tokenize`, but as as a static function (so we can use this class
   * statelessly).
   */
  inline static std::vector<icu::UnicodeString> TokenizeStatic(
      const icu::UnicodeString& text, bool do_lower_case,
      bool tokenize_chinese_chars, const TokenSetT& never_split) {
    return BasicTokenizer(do_lower_case, tokenize_chinese_chars, never_split)
        .Tokenize(text);
  }

  /**
   * Returns true if the token is one of the never_split tokens.
   */
  inline bool IsNeverSplitToken(const icu::UnicodeString& token) const {
    return never_split_.find(token) != never_split_.end();
  }

 private:
  const TokenSetT never_split_;
  const bool do_lower_case_;
  const bool tokenize_chinese_chars_;
};
}  // namespace tokenization
