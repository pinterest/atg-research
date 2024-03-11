// This source file is necessary to register the operators for PyPI package
// build.

#include <ATen/core/ivalue.h>
#include <torch/script.h>

#include "operators/basic_tokenize.h"
#include "operators/bert_tokenizer.h"

using torchscript::operators::BertTokenizer;
using torchscript::operators::VocabTokenizer;

TORCH_LIBRARY(pinterest_ops, m) {
  m.def("basic_tokenize", &torchscript::operators::basic_tokenize)
      .def("char_trigram_tokenize",
           &torchscript::operators::char_trigram_tokenize)
      .def("ngram_tokenize", &torchscript::operators::ngram_tokenize)
      .def("hash_tokenize", &torchscript::operators::hash_tokenize);

  m.class_<VocabTokenizer>("VocabTokenizer")
      .def(torch::init<torch::Dict<std::string, int64_t>, int64_t>())
      .def("ngram_tokenize", &VocabTokenizer::NgramTokenize)
      .def("char_trigram_tokenize", &VocabTokenizer::CharTrigramTokenize)
      .def_pickle([](const c10::intrusive_ptr<VocabTokenizer>& self)
                      -> VocabTokenizer::StateT { return self->state(); },
                  [](VocabTokenizer::StateT state)
                      -> c10::intrusive_ptr<VocabTokenizer> {
                    return c10::make_intrusive<VocabTokenizer>(
                        std::move(std::get<0>(state)),
                        std::move(std::get<1>(state)));
                  })
      .def("__len__", &VocabTokenizer::size);

  m.class_<BertTokenizer>("BertTokenizer")
      .def(torch::init<torch::Dict<std::string, int64_t>, int64_t,
                       torch::List<std::string>, bool, bool,
                       torch::List<std::string>>(),
           "/**\n"
           "   *\n"
           "   * @param vocab Vocabulary of the tokenizer, map of string to "
           "token id.\n"
           "   * @param max_sequence_length Maximum sequence length of the "
           "text after which "
           "truncation is applied.\n"
           "   * @param text_normalization_options List of normalization "
           "operation to carry out\n"
           "   *    on the string before tokenization.\n"
           "   * @param do_lowercase Whether or not to lowercase the input "
           "when tokenizing.\n"
           "   * @param tokenize_chinese_chars Whether or not to tokenize "
           "Chinese characters.\n"
           "   *    This should likely be deactivated for Japanese\n"
           "   *    (see this `issue "
           "<https://github.com/huggingface/transformers/issues/328>`__).\n"
           "   * @param never_split Collection of tokens which will never be "
           "split during "
           "tokenization.\n"
           "   */")
      .def("__len__", &BertTokenizer::size)
      .def("batch_encode", &BertTokenizer::BatchEncode)
      .def_pickle([](const c10::intrusive_ptr<BertTokenizer>& self)
                      -> BertTokenizer::StateT { return self->state(); },
                  [](const BertTokenizer::StateT& state)
                      -> c10::intrusive_ptr<BertTokenizer> {
                    return c10::make_intrusive<BertTokenizer>(state);
                  });
}
