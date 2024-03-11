#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "unicode/errorcode.h"
#include "unicode/normalizer2.h"
#include "unicode/unistr.h"

namespace tokenization {

/**
 * Normalizes a text.
 *
 * This class supports the following normalizations.
 *
 *  - Apply Unicode Normalization NFKC.
 *  - Lowercase.
 *  - Trim the spaces at the beginning and in the end.
 *  - Replace continuous whitespace characters (e.g., \n, \r) by a single white
 * space.
 */
class TextNormalizer {
 public:
  enum TextNormalizationOption {
    kLowercase = 0,
    kUnicodeNormalize = 1,
    kTrimSpace = 2,
    kCollapseWhitespace = 3,
  };

  /**
   * @param options The normalization to apply to a given text.
   */
  explicit TextNormalizer(const std::vector<TextNormalizationOption>& options);

  /**
   * Normalizes a text using the set options.
   * @param text Input text.
   * @return Normalized text.
   */
  icu::UnicodeString Normalize(const icu::UnicodeString& text) const;

 private:
  const std::unordered_set<TextNormalizationOption> options_;
  // The pointer points to the singleton instance owned by ICU. We keep a
  // pointer in the class as it is recommended by ICU to cache the resources.
  const icu::Normalizer2* const nfkc_;

  /**
   * Returns true of the `option` is set in normalization options.
   */
  bool HasOption(TextNormalizationOption option) const {
    return options_.find(option) != options_.end();
  }
};
}  // namespace tokenization
