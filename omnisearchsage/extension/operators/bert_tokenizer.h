#pragma once

#include <ATen/core/ivalue.h>
#include <torch/script.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tokenization/bert_tokenizer/bert_tokenizer.h"
#include "tokenization/text_normalizer.h"

namespace torchscript::operators {

/**
 * Implementation of BERTTokenizer with pre-tokenization text normalization
 * option. It always truncates to the given maximum sequence length and uses
 * LONGEST padding. It is scriptable and faster than the Huggingface tokenizer
 * and recommended tokenizer to use.
 */
class BertTokenizer : public torch::CustomClassHolder {
 public:
  /**
   * The state of the tokenizer is the tuple of constructor args. For details
   * see the doc for the constructor.
   */
  using StateT =
      std::tuple<torch::Dict<std::string, int64_t>,  // vocab
                 int64_t,                            // max_sequence_length
                 torch::List<std::string>,   // text_normalization_options
                 bool,                       // do_lowercase
                 bool,                       // tokenize_chinese_chars
                 torch::List<std::string>>;  // never_split
  /**
   * @param vocab Vocabulary of the tokenizer, map of string to token id.
   * @param max_sequence_length Maximum sequence length of the text after which
   * truncation is applied.
   * @param text_normalization_options List of normalization operation to carry
   * out on the string before tokenization.
   * @param do_lowercase Whether or not to lowercase the input when tokenizing.
   * @param tokenize_chinese_chars Whether or not to tokenize Chinese
   * characters. This should likely be deactivated for Japanese (see this `issue
   * <https://github.com/huggingface/transformers/issues/328>`__).
   * @param never_split Collection of tokens which will never be split during
   * tokenization.
   */
  BertTokenizer(const torch::Dict<std::string, int64_t>& vocab,
                int64_t max_sequence_length,
                const torch::List<std::string>& text_normalization_options,
                bool do_lowercase, bool tokenize_chinese_chars,
                const torch::List<std::string>& never_split);

  /***
   * Creates a tokenizer using saved state.
   * @param state State of the tokenizer
   */
  explicit BertTokenizer(const StateT& state)
      : BertTokenizer(std::get<0>(state), std::get<1>(state),
                      std::get<2>(state), std::get<3>(state),
                      std::get<4>(state), std::get<5>(state)) {}

  BertTokenizer(BertTokenizer const&) = delete;
  BertTokenizer& operator=(BertTokenizer const&) = delete;

  /**
   * Tokenizes and encodes the input text (and optionally applied
   * pre-tokenization normalization).
   *
   * @param inputs List of strings to tokenize.
   * @param normalize Whether to normalize the text before tokenization.
   * @return Dict of tensor with input_ids representing the padded token_ids and
   * attention_mask representing the corresponding mask.
   */
  torch::Dict<std::string, torch::Tensor> BatchEncode(
      const torch::List<std::string>& inputs, bool normalize = true) const;
  /**
   * @return Size of the vocabulary.
   */
  inline int64_t size() const {
    return static_cast<int64_t>(tokenizer_.size());
  }

  /**
   * @return Serializable state of the tokenizer.
   */
  inline const StateT& state() const { return init_args_; }

 private:
  const tokenization::TextNormalizer text_normalizer_;
  const tokenization::BertTokenizer tokenizer_;
  const StateT init_args_;
};

}  // namespace torchscript::operators
