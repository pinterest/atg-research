#include "tokenization/text_normalizer.h"
#include <stdexcept>
#include "unicode/schriter.h"

namespace {

/**
 * Replaces continuous whitespace characters (e.g., \n, \r) by a single white
 * space.
 */
icu::UnicodeString CollapseWhitespace(const icu::UnicodeString& text) {
  icu::UnicodeString result(text.length(), 0, 0);
  bool is_last_space = false;
  icu::StringCharacterIterator it(text);
  while (it.hasNext()) {
    const auto c = it.next32PostInc();
    if (u_isUWhiteSpace(c)) {
      is_last_space = true;
    } else {
      if (is_last_space) result.append(' ');
      result.append(c);
      is_last_space = false;
    }
  }
  if (is_last_space) result.append(' ');
  return result;
}

/**
 * Throws exception if icu_error is a failure. Useful to assert that icu:: calls
 * succeeds.
 */
inline void CheckStatus(icu::ErrorCode icu_error, const std::string& message) {
  if (U_FAILURE(icu_error)) {
    throw std::invalid_argument(std::string(icu_error.errorName()) + ": " +
                                message);
  }
}

/**
 * Returns the singleton NFKC normalizer from icu.
 */
const icu::Normalizer2* const GetNormalizer() {
  icu::ErrorCode icu_error;
  const icu::Normalizer2* const nfkc =
      icu::Normalizer2::getNFKCInstance(icu_error);
  CheckStatus(icu_error, "Failed to get NFKC normalizer");
  return nfkc;
}

/**
 * Applies Unicode Normalization NFKC to text.
 */
icu::UnicodeString UnicodeNormalize(const icu::UnicodeString& text,
                                    const icu::Normalizer2* const nfkc_) {
  icu::ErrorCode icu_error;

  icu::UnicodeString ntext = nfkc_->normalize(text, icu_error);
  CheckStatus(icu_error, "Could not Normalize input string");
  return ntext;
}

}  // namespace

namespace tokenization {

TextNormalizer::TextNormalizer(
    const std::vector<TextNormalizationOption>& options)
    : options_(options.begin(), options.end()), nfkc_(GetNormalizer()) {}

icu::UnicodeString TextNormalizer::Normalize(
    const icu::UnicodeString& text) const {
  if (text.isEmpty()) return text;
  icu::UnicodeString out = text;
  if (HasOption(kUnicodeNormalize)) out = UnicodeNormalize(out, nfkc_);

  if (HasOption(kLowercase)) out = out.toLower();

  if (HasOption(kCollapseWhitespace)) out = CollapseWhitespace(out);

  if (HasOption(kTrimSpace)) out = out.trim();
  return out;
}
}  // namespace tokenization
