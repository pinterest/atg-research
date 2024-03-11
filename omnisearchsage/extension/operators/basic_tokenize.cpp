#include "operators/basic_tokenize.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "tokenization/bert_tokenizer/basic_tokenizer.h"
#include "tokenization/ngram.h"
#include "unicode/unistr.h"

#ifndef FALLTHROUGH_INTENDED
#define FALLTHROUGH_INTENDED \
  do {                       \
  } while (0)
#endif

namespace {

inline uint32_t DecodeFixed32(const char* ptr) {
  const uint8_t* const buffer = reinterpret_cast<const uint8_t*>(ptr);

  // Recent clang and gcc optimize this to a single mov / ldr instruction.
  return (static_cast<uint32_t>(buffer[0])) |
         (static_cast<uint32_t>(buffer[1]) << 8) |
         (static_cast<uint32_t>(buffer[2]) << 16) |
         (static_cast<uint32_t>(buffer[3]) << 24);
}

absl::flat_hash_map<std::string, int64_t> CheckAndConvertDictToFastMap(
    const torch::Dict<std::string, int64_t>& dict) {
  absl::flat_hash_map<std::string, int64_t> map;
  int64_t num_elems = static_cast<int64_t>(dict.size());
  map.reserve(dict.size());
  for (const auto& kv : dict) {
    int64_t value = kv.value();
    TORCH_CHECK(value < num_elems, "Value ", value, " is out of range [0, ",
                dict.size(), ")");
    map[kv.key()] = kv.value();
  }
  return map;
}

torch::Dict<std::string, int64_t> FastMapToDict(
    const absl::flat_hash_map<std::string, int64_t>& map) {
  torch::Dict<std::string, int64_t> dict;
  dict.reserve(map.size());
  for (const auto& kv : map) {
    dict.insert(kv.first, kv.second);
  }
  return dict;
}

// some arbitrary hash function, copy pasted from
// https://github.com/google/leveldb/blob/master/util/hash.cc (BSD 3-clause
// license) but moving seed to a fixed parameter
/**
Copyright (c) 2011 The LevelDB Authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
   * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
**/
inline uint32_t Hash(const char* data, size_t n) {
  const uint32_t seed = 0xbc9f1d34;

  // Similar to murmur hash
  const uint32_t m = 0xc6a4a793;
  const uint32_t r = 24;
  const char* limit = data + n;
  uint32_t h = seed ^ (n * m);

  // Pick up four bytes at a time
  while (data + 4 <= limit) {
    uint32_t w = DecodeFixed32(data);
    data += 4;
    h += w;
    h *= m;
    h ^= (h >> 16);
  }

  // Pick up remaining bytes
  switch (limit - data) {
    case 3:
      h += static_cast<uint8_t>(data[2]) << 16;
      FALLTHROUGH_INTENDED;
    case 2:
      h += static_cast<uint8_t>(data[1]) << 8;
      FALLTHROUGH_INTENDED;
    case 1:
      h += static_cast<uint8_t>(data[0]);
      h *= m;
      h ^= (h >> r);
      break;
  }
  return h;
}

}  // namespace

namespace torchscript::operators {

std::vector<std::string> basic_tokenize(const std::string& text,
                                        bool do_lower_case,
                                        bool tokenize_chinese_chars) {
  icu::UnicodeString icu_str = icu::UnicodeString::fromUTF8(text);

  const std::vector<icu::UnicodeString> out_unistr =
      tokenization::BasicTokenizer::TokenizeStatic(
          icu_str, do_lower_case, tokenize_chinese_chars,
          tokenization::BasicTokenizer::TokenSetT());
  std::vector<std::string> out;
  out.reserve(out_unistr.size());
  for (const auto& item : out_unistr) {
    std::string s;
    item.toUTF8String(s);
    out.push_back(s);
  }
  return out;
}

std::vector<std::string> char_trigram_tokenize(const std::string& text) {
  std::vector<icu::UnicodeString> out_unistr;
  icu::UnicodeString icu_str = icu::UnicodeString::fromUTF8(text);
  tokenization::CharNGramTokenize(icu_str, /*n=*/3, &out_unistr);
  std::vector<std::string> out;
  out.reserve(out_unistr.size());
  for (const auto& item : out_unistr) {
    std::string s;
    item.toUTF8String(s);
    out.push_back(s);
  }
  return out;
}

std::vector<std::string> ngram_tokenize(int64_t n, const std::string& text) {
  std::vector<icu::UnicodeString> out_unistr;
  icu::UnicodeString icu_str = icu::UnicodeString::fromUTF8(text);
  tokenization::NGramTokenize(icu_str, /*n=*/n, &out_unistr);
  std::vector<std::string> out;
  out.reserve(out_unistr.size());
  for (const auto& item : out_unistr) {
    std::string s;
    item.toUTF8String(s);
    out.push_back(std::move(s));
  }
  return out;
}

torch::Tensor hash_tokenize(std::vector<std::string> raw_data) {
  torch::Tensor out = torch::zeros({static_cast<int64_t>(raw_data.size())},
                                   at::TensorOptions(at::kLong));
  for (size_t i = 0; i < raw_data.size(); ++i) {
    const std::string& s = raw_data.at(i);
    if (!s.empty()) {
      // Hash returns a uint32_t, so 1 + Hash(...) is never equal to zero. This
      // allows us to have  a unique token for empty string
      out[i] = 1 + static_cast<int64_t>(simple_hash(s));
    }
  }
  return out;
}

int32_t simple_hash(const std::string& s) { return Hash(s.data(), s.size()); }

VocabTokenizer::VocabTokenizer(torch::Dict<std::string, int64_t> vocab,
                               int64_t oov_size)
    : vocab_(CheckAndConvertDictToFastMap(vocab)), oov_size_(oov_size) {}

VocabTokenizer::StateT VocabTokenizer::state() const {
  return std::make_tuple(FastMapToDict(vocab_), oov_size_);
}

torch::Tensor VocabTokenizer::NgramTokenize(std::string s, int64_t n) const {
  return TokensToTensor(ngram_tokenize(n, s));
}

torch::Tensor VocabTokenizer::CharTrigramTokenize(std::string s) const {
  return TokensToTensor(char_trigram_tokenize(s));
}

torch::Tensor VocabTokenizer::TokensToTensor(
    const std::vector<std::string>& tokens) const {
  torch::Tensor out = torch::zeros({static_cast<int64_t>(tokens.size())},
                                   at::TensorOptions(at::kLong));
  size_t i = 0;
  auto out_access = out.accessor<int64_t, 1>();
  for (const auto& token : tokens) {
    auto it = vocab_.find(token);
    if ((it == vocab_.end()) && (oov_size_ == 0)) {
      continue;
    } else if ((it == vocab_.end())) {
      const int32_t oov_idx = simple_hash(token) % oov_size_;
      out_access[i++] = static_cast<int64_t>(oov_idx + vocab_.size());
    } else {
      out_access[i++] = it->second;
    }
  }
  return out.slice(0, 0, i);
}

}  // namespace torchscript::operators
