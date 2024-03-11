#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenization/bert_tokenizer/tokenization_bert_utils.h"
#include "unicode/unistr.h"

namespace tokenization {
/**
 * Character N-Gram tokenization
 *
 * Sanitization done are NFKC normalization, case folding, duplicate whitespace
 * removal.
 *
 * @param text input text string
 * @param n number of consecutive characters for N-gram
 * @return vector of strings, each of which is an individual token
 */
void CharNGramTokenize(const icu::UnicodeString& text, size_t n,
                       std::vector<icu::UnicodeString>* output);

/**
 * N-Gram tokenization
 *
 * Sanitization done are NFKC normalization, case folding, accent removal,
 * duplicate whitespace removal.
 *
 * @param text input text string
 * @param n number of consecutive characters for N-gram
 * @return vector of strings, each of which is an individual token
 */
void NGramTokenize(const icu::UnicodeString& text, size_t n,
                   std::vector<icu::UnicodeString>* output);

}  // namespace tokenization
