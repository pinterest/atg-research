#pragma once

#include <string>
#include <vector>

#include "unicode/uchar.h"
#include "unicode/unistr.h"

namespace tokenization {

/**
 * Hash function for icu::UnicodeString. This is needed because
 * `icu::UnicodeString` has a non-standard, java-like interface.
 */
class Hasher {
 public:
  int64_t operator()(const icu::UnicodeString& text) const {
    return text.hashCode();
  }
};

/**
 * Returns true if a string contains valid utf-8 code points.
 */
bool ValidateUTF8(const std::string& text);

/**
 * Checks if a unicode code point is of type "Zs" (space separator).
 */
inline bool IsWhitespace(const char32_t cp) {
  return cp == '\t' || cp == '\n' || cp == ' ' || cp == '\r' ||
         u_charType(cp) == U_SPACE_SEPARATOR;
}

/**
 * Checks if a unicode code point is a control character (excluding whitespace).
 */
inline bool IsControl(const char32_t cp) {
  if (IsWhitespace(cp)) {
    return false;
  }
  int8_t char_type = u_charType(cp);

  return char_type == U_PRIVATE_USE_CHAR || char_type == U_FORMAT_CHAR ||
         char_type == U_CONTROL_CHAR || char_type == U_SURROGATE;
}

/**
 * Checks if a unicode code point is punctuation.
 */
inline bool IsPunctuation(const char32_t cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
    return true;
  }
  return u_ispunct(cp);
}

/**
 * Checks if a unicode code point is of type "Mn" (non-spacing mark).
 */
inline bool IsAccent(const char32_t cp) {
  return u_charType(cp) == U_NON_SPACING_MARK;
}

/**
 * Checks whether CP is the codepoint of a CJK character
 * This defines a "chinese character" as anything in the CJK Unicode block:
 *   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
 *
 * Note that the CJK Unicode block is NOT all Japanese and Korean characters,
 * despite its name. The modern Korean Hangul alphabet is a different block,
 * as is Japanese Hiragana and Katakana. Those alphabets are used to write
 * space-separated words, so they are not treated specially and handled
 * like the all of the other languages.
 */
inline bool IsCjk(const char32_t cp) {
  return (
      (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F));
}

/**
 * Prints out a vector of `UnicodeString`s, delimiting them by space.
 */
void PrintStrings(const std::vector<icu::UnicodeString>& v);

/**
 * Reimplementation of `whitespace_tokenize` from `transformers`.
 * In python this would be text.strip().split()
 */
std::vector<icu::UnicodeString> WhitespaceTokenize(
    const icu::UnicodeString& text);

/**
 * Adds whitespace around any CJK character.
 */
icu::UnicodeString TokenizeChineseChars(const icu::UnicodeString& text);

}  // namespace tokenization
