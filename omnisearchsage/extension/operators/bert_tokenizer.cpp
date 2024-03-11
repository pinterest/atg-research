#include "operators/bert_tokenizer.h"

#include <ATen/core/ivalue.h>
#include <torch/script.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "unicode/unistr.h"

namespace {

/**
 * Converts the keys in the vocab from UTF8 strings to unicode strings.
 */
tokenization::BertTokenizer::VocabT ConvertVocab(
    const torch::Dict<std::string, int64_t>& vocab) {
  auto result = tokenization::BertTokenizer::VocabT();
  result.reserve(vocab.size());
  for (const auto& item : vocab) {
    size_t item_id = item.value();
    TORCH_CHECK(item_id < vocab.size());
    result[icu::UnicodeString::fromUTF8(item.key())] = item.value();
  }
  return result;
}

/**
 * Creates the set of text normalization options from their names.
 */
std::vector<tokenization::TextNormalizer::TextNormalizationOption>
OptionsFromName(const torch::List<std::string>& options) {
  std::vector<tokenization::TextNormalizer::TextNormalizationOption> result;
  result.reserve(options.size());
  for (const std::string& option : options.vec()) {
    if (option == "LOWERCASE")
      result.push_back(tokenization::TextNormalizer::kLowercase);
    else if (option == "TRIM_SPACE")
      result.push_back(tokenization::TextNormalizer::kTrimSpace);
    else if (option == "UNICODE_NORMALIZE")
      result.push_back(tokenization::TextNormalizer::kUnicodeNormalize);
    else if (option == "COLLAPSE_WHITESPACE")
      result.push_back(tokenization::TextNormalizer::kCollapseWhitespace);
    else
      throw std::invalid_argument("Invalid option: " + option);
  }
  return result;
}

/**
 * Converts the special tokens from UTF8 strings to unicode strings.
 */
tokenization::BasicTokenizer::TokenSetT TokenSetFromList(
    const torch::List<std::string>& inp) {
  tokenization::BasicTokenizer::TokenSetT out;
  out.reserve(inp.size());
  for (const std::string& t : inp.vec()) {
    out.insert(icu::UnicodeString::fromUTF8(t));
  }
  return out;
}
}  // namespace

namespace torchscript::operators {

BertTokenizer::BertTokenizer(
    const torch::Dict<std::string, int64_t>& vocab, int64_t max_sequence_length,
    const torch::List<std::string>& text_normalization_options,
    bool do_lowercase, bool tokenize_chinese_chars,
    const torch::List<std::string>& never_split)
    : text_normalizer_(OptionsFromName(text_normalization_options)),
      tokenizer_(ConvertVocab(vocab), max_sequence_length, do_lowercase,
                 tokenize_chinese_chars, TokenSetFromList(never_split)),
      init_args_(vocab, max_sequence_length, text_normalization_options,
                 do_lowercase, tokenize_chinese_chars, never_split) {}

torch::Dict<std::string, torch::Tensor> BertTokenizer::BatchEncode(
    const torch::List<std::string>& inputs, bool normalize) const {
  std::vector<torch::Tensor> output;
  output.reserve(inputs.size());
  int64_t max_size = 0;
  for (const std::string& input : inputs.vec()) {
    const auto unicode_input = icu::UnicodeString::fromUTF8(input);
    const auto normalized_input =
        normalize ? text_normalizer_.Normalize(unicode_input) : unicode_input;

    std::vector<int64_t> token_ids = tokenizer_.Encode(normalized_input);
    auto size = static_cast<int64_t>(token_ids.size());
    output.push_back(
        torch::from_blob(token_ids.data(), {size}, torch::kInt64).clone());
    max_size = std::max(max_size, size);
  }
  torch::Tensor token_ids =
      torch::full({static_cast<int64_t>(output.size()), max_size},
                  tokenizer_.pad_token_id(), torch::kInt64);
  torch::Tensor mask = torch::zeros_like(token_ids);
  for (size_t i = 0; i < output.size(); i++) {
    const torch::Tensor& t = output[i];
    const std::initializer_list<torch::indexing::TensorIndex> index = {
        static_cast<int>(i),
        torch::indexing::Slice(torch::indexing::None, t.size(0))};
    token_ids.index_put_(index, t);
    mask.index_put_(index, 1);
  }
  auto result = torch::Dict<std::string, torch::Tensor>();
  result.insert("input_ids", token_ids);
  result.insert("attention_mask", mask);
  return result;
}

}  // namespace torchscript::operators
