#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tokenization/bert_tokenizer/basic_tokenizer.h"
#include "tokenization/bert_tokenizer/wordpiece_tokenizer.h"
#include "unicode/unistr.h"

namespace tokenization {

/**
 * C++ Implementation of BertTokenizer.
 */
class BertTokenizer {
 public:
  using VocabT = WordpieceTokenizer::VocabT;
  /**
   * Constructs a BertTokenizer from a given vocab.
   * This should almost output identical results to
   * `transformers.BertTokenizer`.
   *
   * @param vocab Vocabulary of the tokenizer, map of unicode string to token
   * id.
   * @param max_sequence_length Maximum sequence length of the text after which
   * truncation is applied.
   * @param do_lowercase Whether or not to lowercase the input when tokenizing.
   * @param tokenize_chinese_chars Whether or not to tokenize Chinese
   * characters. This should likely be deactivated for Japanese (see this `issue
   * <https://github.com/huggingface/transformers/issues/328>`__).
   * @param never_split Collection of tokens which will never be split during
   * tokenization.
   */
  BertTokenizer(const VocabT& vocab, int64_t max_sequence_length,
                bool do_lower_case = true, bool tokenize_chinese_chars = true,
                const tokenization::BasicTokenizer::TokenSetT& never_split =
                    tokenization::BasicTokenizer::TokenSetT());

  BertTokenizer(BertTokenizer const&) = delete;
  BertTokenizer& operator=(BertTokenizer const&) = delete;

  /**
   * Given a string, encodes it as a series of ints. Should be equivalent to
   * `transformers BertTokenizer.encode(text, add_special_tokens=True)` in
   * python.
   */
  std::vector<std::int64_t> Encode(const icu::UnicodeString& text) const;
#if __cplusplus >= 202001L  // C++20
  // C++20 introduces type char8_t and changes the type of string literal
  // u8"foo" from const char[] to const char8_t[]. This is a breaking change.
  // Define this overload to allow Encode(u8"foo") to work with C++20.
  std::vector<std::int64_t> Encode(const char8_t* text) const {
    return Encode(std::reinterpret_cast<const char*>(text));
  }
#endif

  /**
   * Size of the vocabulary.
   */
  inline size_t size() const { return vocab_.size(); }

  /**
   * Id of the pad token.
   */
  inline int64_t pad_token_id() const { return pad_token_id_; }

 private:
  /**
   * Returns the id corresponding to the token in vocab.
   * If the token is not present in the the vocab, the unk_token_id is returned
   * if errorIfMissing is false, else an exception is thrown.
   */
  int64_t TokenToId(const icu::UnicodeString& token,
                    bool errorIfMissing = false) const;

  const VocabT vocab_;
  const int64_t max_sequence_length_;
  const int64_t unk_token_id_;
  const int64_t sep_token_id_;
  const int64_t cls_token_id_;
  const int64_t pad_token_id_;

  const BasicTokenizer basic_tokenizer_;
  const WordpieceTokenizer wordpiece_tokenizer_;
};

}  // namespace tokenization
