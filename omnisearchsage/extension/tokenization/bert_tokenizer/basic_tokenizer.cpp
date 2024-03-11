#include "tokenization/bert_tokenizer/basic_tokenizer.h"

#include "tokenization/bert_tokenizer/tokenization_bert_utils.h"

#ifdef GLOG_ENABLED
#include "common/log.h"
#endif
#include <numeric>
#include <vector>

#include "unicode/errorcode.h"
#include "unicode/normalizer2.h"
#include "unicode/schriter.h"

namespace {

/**
 * Performs invalid character removal and whitespace cleanup on text.
 * `output' should be an empty string and the clean text is stored in it.
 */
icu::UnicodeString CleanText(const icu::UnicodeString& text) {
  icu::StringCharacterIterator it(text);
  icu::UnicodeString output(text.length(), 0, 0);
  while (it.hasNext()) {
    const UChar32 c = it.next32PostInc();
    if (c == 0 || c == 0xFFFD || tokenization::IsControl(c)) {
    } else if (tokenization::IsWhitespace(c)) {
      output.append(' ');
    } else {
      output.append(c);
    }
  }
  return output;
}

/**
 * Strips accents from a piece of text.
 */
icu::UnicodeString StripAccents(const icu::UnicodeString& text) {
  icu::ErrorCode status;
  const auto normalizer = icu::Normalizer2::getNFDInstance(status);
  if (U_FAILURE(status)) {
    throw std::invalid_argument(std::string(status.errorName()) +
                                ": Failed to get instance of NFD normalizer");
  }
  const icu::UnicodeString r = normalizer->normalize(text, status);
  if (U_FAILURE(status)) {
    throw std::invalid_argument(std::string(status.errorName()) +
                                ": Failed to normalize text");
  }

  icu::UnicodeString out(r.length(), 0, 0);
  for (int i = 0; i < r.length(); ++i) {
    const auto cp = r.char32At(i);
    if (!tokenization::IsAccent(cp)) {
      out.append(cp);
    }
  }
  return out;
}

/**
 * Splits punctuation on a piece of text. The tokens are appended to the
 * `output' vector.
 */
void SplitOnPunc(const icu::UnicodeString& text,
                 std::vector<icu::UnicodeString>* output) {
  icu::StringCharacterIterator it(text);
  bool start_new_word = true;
  while (it.hasNext()) {
    const UChar32 cp = it.next32PostInc();
    if (tokenization::IsPunctuation(cp)) {
      output->push_back(cp);
      start_new_word = true;
    } else {
      if (start_new_word) {
        output->push_back(cp);
      } else {
        output->back().append(cp);
      }
      start_new_word = false;
    }
  }
}

}  // namespace

namespace tokenization {

std::vector<icu::UnicodeString> BasicTokenizer::Tokenize(
    const icu::UnicodeString& text) const {
  const icu::UnicodeString cleaned_text = CleanText(text);
  const icu::UnicodeString cjk_split_text =
      tokenize_chinese_chars_ ? TokenizeChineseChars(cleaned_text)
                              : cleaned_text;
  const std::vector<icu::UnicodeString> orig_tokens =
      WhitespaceTokenize(cjk_split_text);
  std::vector<icu::UnicodeString> split_tokens;
  split_tokens.reserve(orig_tokens.size());
  for (const auto& token : orig_tokens) {
    const bool should_lowercase = !IsNeverSplitToken(token) && do_lower_case_;
    const auto lowercased_token =
        should_lowercase ? StripAccents(icu::UnicodeString(token).toLower())
                         : token;
    if (IsNeverSplitToken(lowercased_token))
      split_tokens.push_back(lowercased_token);
    else
      SplitOnPunc(lowercased_token, &split_tokens);
  }

  icu::UnicodeString joined = std::accumulate(
      split_tokens.begin(), split_tokens.end(), icu::UnicodeString(),
      [](icu::UnicodeString s1, const icu::UnicodeString& s2) {
        s1.append(' ');
        s1 += s2;
        return s1;
      });

  return WhitespaceTokenize(joined);
}

}  // namespace tokenization
