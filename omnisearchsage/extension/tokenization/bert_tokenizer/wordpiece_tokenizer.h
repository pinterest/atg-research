#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tokenization/bert_tokenizer/tokenization_bert_utils.h"
#include "unicode/unistr.h"

namespace tokenization {

/**
 * Implementation of BertTokenizer.WordpieceTokenizer.
 */
class WordpieceTokenizer {
 public:
  using VocabT = absl::flat_hash_map<icu::UnicodeString, int64_t, Hasher>;
  /**
   *
   * @param vocab Vocabulary of the tokenizer, map of unicode string to token
   * id.
   * @param unk_token UNK token.
   * @param max_input_chars_per_word Maximum input characters to consider per
   * word. Returns UNK for longer words.
   */
  WordpieceTokenizer(const VocabT& vocab, icu::UnicodeString unk_token,
                     int32_t max_input_chars_per_word = 100)
      : vocab_(vocab),
        unk_token_(unk_token),
        max_input_chars_per_word_(max_input_chars_per_word) {}

  WordpieceTokenizer(WordpieceTokenizer const&) = delete;
  WordpieceTokenizer& operator=(WordpieceTokenizer const&) = delete;

  /**
   * Tokenizes a piece of text into its word pieces.
   * This uses a greedy longest-match-first algorithm to perform tokenization
   * using the given vocabulary.
   *
   * For example, :obj:`input = "unaffable"` wil return as output :obj:`["un",
   * "##aff", "##able"]`.
   *
   * @param text: A single token or whitespace separated tokens.
   * This should have already been passed through `BasicTokenizer`.
   *
   * @returns A vector of wordpiece tokens.
   */
  std::vector<icu::UnicodeString> Tokenize(
      const icu::UnicodeString& text) const;

 private:
  const VocabT vocab_;
  const icu::UnicodeString unk_token_;
  const int32_t max_input_chars_per_word_;
};
}  // namespace tokenization
