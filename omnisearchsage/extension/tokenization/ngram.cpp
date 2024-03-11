#include "tokenization/ngram.h"

#ifdef GLOG_ENABLED
#include "common/log.h"
#endif

#include <deque>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "tokenization/bert_tokenizer/tokenization_bert_utils.h"
#include "unicode/errorcode.h"
#include "unicode/normalizer2.h"
#include "unicode/schriter.h"
#include "unicode/ustring.h"

namespace tokenization {
namespace {

const char32_t kSpaceDelim = '#';

/**
 * Buffer of n characters that can output the buffer as a UnicodeString
 **/
class CharNGramQueue {
 public:
  // n = buffer size
  explicit CharNGramQueue(size_t n) : max_length_(n) {}

  // push _value_ into buffer. remove element (FIFO) to maintain buffer size
  void push(char32_t value) {
    if (buffer_.size() == max_length_) {
      buffer_.pop_front();
    }
    buffer_.push_back(value);
  }

  // value of last character put into buffer
  char32_t back() { return buffer_.back(); }

  // string value of internal buffer
  icu::UnicodeString str() {
    icu::UnicodeString s(max_length_, 0, 0);
    for (char32_t c : buffer_) {
      s.append(static_cast<int32_t>(c));
    }
    return s;
  }

  // size of buffer
  size_t size() { return buffer_.size(); }

 private:
  std::deque<char32_t> buffer_;
  size_t max_length_;
};

/**
 * Buffer of n words that can output the buffer as a UnicodeString
 **/
class NGramQueue {
 public:
  // n = buffer size
  explicit NGramQueue(size_t n) : max_length_(n) {}

  // push _value_ into buffer. remove element (FIFO) to maintain buffer size
  void push(icu::UnicodeString&& value) {
    if (buffer_.size() == max_length_) {
      buffer_.pop_front();
    }
    buffer_.push_back(value);
  }

  // string value of internal buffer
  icu::UnicodeString str() {
    int length = 0;
    for (const icu::UnicodeString& c : buffer_) {
      // +1 for the separator character
      length += c.length() + 1;
    }
    length -= 1;  // don't account for separator character at end

    icu::UnicodeString s(length, 0, 0);
    for (size_t i = 0; i < buffer_.size(); ++i) {
      s.append(buffer_.at(i));
      if (i != buffer_.size() - 1) {
        s.append(static_cast<int32_t>(kSpaceDelim));
      }
    }
    return s;
  }

  // size of buffer
  size_t size() { return buffer_.size(); }

 private:
  std::deque<icu::UnicodeString> buffer_;
  size_t max_length_;
};

/**
 * character n-gram the given text into tokens
 **/
void CharNgramify(const icu::UnicodeString& text, size_t n,
                  std::vector<icu::UnicodeString>* tokens) {
  // +2 because we append and prepend # symbols for word boundaries
  if (static_cast<size_t>(text.length()) + 2 < n) {
    return;
  }

  // num char trigrams is roughly (text.length() - n + 1) (assuming no duplicate
  // spaces) but we also prepend and append kSpaceDelim to the sentence so +2
  // more char trigrams
  tokens->reserve(text.length() - n + 1 + 2);

  CharNGramQueue char_buffer(n);

  int startIndex = 0;
  char_buffer.push(kSpaceDelim);

  while (startIndex <= text.length()) {
    // end
    if (startIndex == text.length()) {
      if (char_buffer.back() != kSpaceDelim) {
        char_buffer.push(kSpaceDelim);
        if (char_buffer.size() == n) {
          tokens->emplace_back(char_buffer.str());
        }
      }
      ++startIndex;
      continue;
    }

    if (IsWhitespace(text.char32At(startIndex))) {
      // if multiple white spaces in a row just add # once.
      if (char_buffer.back() != kSpaceDelim) {
        char_buffer.push(kSpaceDelim);
        if (char_buffer.size() == n) {
          tokens->emplace_back(char_buffer.str());
        }
      }
      ++startIndex;
      continue;
    }

    // guaranteed not whitespace
    char_buffer.push(text.char32At(startIndex));
    if (char_buffer.size() == n) {
      tokens->emplace_back(char_buffer.str());
    }
    ++startIndex;
  }
}

bool IsWhitespaceOrPunctuation(const char32_t c) {
  return IsWhitespace(c) || IsPunctuation(c);
}

void Ngramify(const icu::UnicodeString& text, size_t n,
              std::vector<icu::UnicodeString>* tokens) {
  if (static_cast<size_t>(text.length()) < n) {
    return;
  }

  tokens->reserve(text.length() - n + 1);
  NGramQueue buffer(n);

  int32_t length = text.length();
  int32_t word_begin = 0;
  int32_t word_end = 0;

  while (word_end < length) {
    // find beginning of word
    while ((word_begin < length) &&
           IsWhitespaceOrPunctuation(text.char32At(word_begin))) {
      word_begin++;
    }

    // find first whitespace or punctuation after word
    word_end = word_begin;
    while ((word_end < length) &&
           !IsWhitespaceOrPunctuation(text.char32At(word_end))) {
      word_end++;
    }

    if (word_end > word_begin) {
      icu::UnicodeString s(word_end - word_begin, 0, 0);
      for (int32_t i = word_begin; i < word_end; ++i) {
        s.append(static_cast<int32_t>(text.char32At(i)));
      }

      if (n == 1) {
        // for unigram we can just output directly
        tokens->emplace_back(std::move(s));
      } else {
        buffer.push(std::move(s));
        if (buffer.size() == n) {
          tokens->emplace_back(buffer.str());
        }
      }
    }
    word_begin = word_end;
  }
}

icu::UnicodeString Normalize(const icu::UnicodeString& text) {
  icu::ErrorCode icu_error;
  const icu::Normalizer2* nfkc_cf =
      icu::Normalizer2::getNFKCCasefoldInstance(icu_error);

  if (!icu_error.isSuccess()) {
    throw std::invalid_argument(
        std::string(icu_error.errorName()) +
        ": Could not retrieve ICU NFKC_CaseFold normalizer");
  }

  icu::UnicodeString ntext = nfkc_cf->normalize(text, icu_error);
  if (U_FAILURE(icu_error)) {
    throw std::invalid_argument(std::string(icu_error.errorName()) +
                                ": Could not normalize input string");
  }
  return ntext;
}
}  // namespace

void CharNGramTokenize(const icu::UnicodeString& text, size_t n,
                       std::vector<icu::UnicodeString>* output) {
  CharNgramify(Normalize(text), n, output);
}

void NGramTokenize(const icu::UnicodeString& text, size_t n,
                   std::vector<icu::UnicodeString>* output) {
  Ngramify(TokenizeChineseChars(Normalize(text)), n, output);
}
}  // namespace tokenization
