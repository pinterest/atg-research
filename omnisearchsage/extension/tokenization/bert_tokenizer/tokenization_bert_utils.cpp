#include "tokenization/bert_tokenizer/tokenization_bert_utils.h"

#ifdef GLOG_ENABLED
#include "common/log.h"
#endif
#include <memory>
#include <sstream>
#include <vector>

#include "unicode/errorcode.h"
#include "unicode/schriter.h"
#include "unicode/ustring.h"

namespace tokenization {

bool ValidateUTF8(const std::string& text) {
  UErrorCode status = U_ZERO_ERROR;
  // check if valid utf8
  u_strFromUTF8(nullptr, 0, nullptr, text.data(), text.length(), &status);
  return status != U_INVALID_CHAR_FOUND;
}

void PrintStrings(const std::vector<icu::UnicodeString>& v) {
#ifdef GLOG_ENABLED
  icu::UnicodeString s;
  std::stringstream ss;
  for (auto& item : v) {
    std::string cs;
    item.toUTF8String(cs);
    ss << cs << ' ';
  }
  LOG(INFO) << ss.str();
#endif
}

std::vector<icu::UnicodeString> WhitespaceTokenize(
    const icu::UnicodeString& text) {
  std::vector<icu::UnicodeString> output;
  const int32_t length = text.length();
  int32_t word_begin = 0;
  int32_t word_end = 0;

  while (word_end < length) {
    // find beginning of word
    while ((word_begin < length) && IsWhitespace(text.char32At(word_begin))) {
      word_begin++;
    }

    // find first whitespace after word
    word_end = word_begin;
    while ((word_end < length) && !IsWhitespace(text.char32At(word_end))) {
      word_end++;
    }

    if (word_end > word_begin) {
      output.push_back(icu::UnicodeString());
      text.extractBetween(word_begin, word_end, output.back());
    }
    word_begin = word_end;
  }
  return output;
}

icu::UnicodeString TokenizeChineseChars(const icu::UnicodeString& text) {
  icu::StringCharacterIterator it(text);
  icu::UnicodeString output(text.length(), 0, 0);
  while (it.hasNext()) {
    const UChar32 c = it.next32PostInc();
    if (IsCjk(c)) {
      output.append(' ');
      output.append(c);
      output.append(' ');
    } else {
      output.append(c);
    }
  }
  return output;
}
}  // namespace tokenization
