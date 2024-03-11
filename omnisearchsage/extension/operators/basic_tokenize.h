#pragma once

#include <ATen/core/ivalue.h>
#include <torch/script.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"

namespace torchscript {
namespace operators {
/**
 * Apply basic sanitization to `text` and split the inputs on punctuation.
 * This is a reimplementation of `tokenization_bert.BasicTokenizer` to work
 * with Torchscript, as the original implementation has `unicodedata` python
 * calls (which cannot be scripted)
 *
 * @param text text to tokenize and split
 * @param do_lower_case if true, convert letters to lower case and strip accents
 * @param tokenize_chinese_chars if true, split each cjk character into its own
 * token
 * @return vector of strings, each of which is an individual token as defined by
 * `BasicTokenizer`
 */
std::vector<std::string> basic_tokenize(const std::string& text,
                                        bool do_lower_case,
                                        bool tokenize_chinese_chars);

/**
 * Character Trigram tokenization
 *
 * Input: "artistic iphone 6s case"
 *
 * Output: ["#ar", "art", "rti", "tis", "ist", "sti", "tic", "ic#", "c#i",
 * "#ip", "iph", "pho", "hon", "one", "ne#", "e#6", "#6s", "6s#", "s#c", "#ca",
 * "cas", "ase", "se#"].
 *
 * We sanitize with NFKC normalization, case folding, and duplicate whitespace
 * removal.
 *
 * @param text input text string
 * @return vector of strings, each of which is an individual token
 */
std::vector<std::string> char_trigram_tokenize(const std::string& text);

/**
 * Ngram tokenization
 *
 * Input: "artistic iphone 6s case"
 *
 * N = 1 Output: ["artistic", "iphone", "6s", "case"].
 * N = 2 Output: ["artistic#iphone", "iphone#6s", "6s#case"].
 *
 * We sanitize with NFKC normalization, case folding, and
 * duplicate (whitespace | punctuation) removal.
 *
 * @param text input text string
 * @return vector of strings, each of which is an individual token
 */
std::vector<std::string> ngram_tokenize(int64_t n, const std::string& text);

/*
 * Tokenize a vector of strings by hashing them. Empty string will be assigned
 * to 0, all other strings will be hashed to a int32_t
 *
 * Torch modulo behaves like python, so this won't cause any issues:
 * In [4]: torch.tensor([-10, -5]) % 3
 * Out[4]: tensor([2, 1])
 *
 * Given an input of length N, this will return a tensor with shape (N,)
 */
torch::Tensor hash_tokenize(std::vector<std::string> raw_data);

/**
 * Simple string hash function
 */
int32_t simple_hash(const std::string& s);

/**
 * Tokenizer that has some vocabulary, and some oov hash size. It has 2
 * supported operations: CharTrigramTokenize, and NgramTokenize. Rather than
 * have separate tokenizers for each case, we just use this one, then in python
 * can have a wrapper to translate this to a unigram/bigram/trigram/ character
 * trigram tokenizer
 */
class VocabTokenizer : public torch::CustomClassHolder {
 public:
  VocabTokenizer(torch::Dict<std::string, int64_t> vocab, int64_t oov_size);

  torch::Tensor NgramTokenize(std::string s, int64_t n) const;
  torch::Tensor CharTrigramTokenize(std::string s) const;

  using StateT = std::tuple<torch::Dict<std::string, int64_t>, int64_t>;

  StateT state() const;
  int64_t size() const { return vocab_.size() + oov_size_; }

 private:
  torch::Tensor TokensToTensor(const std::vector<std::string>& tokens) const;

  absl::flat_hash_map<std::string, int64_t> vocab_;

  int64_t oov_size_;
};

}  // namespace operators
}  // namespace torchscript
