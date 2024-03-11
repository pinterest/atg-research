cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(torchscript LANGUAGES CXX)

add_subdirectory(third_party)
add_subdirectory(third_party/abseil-cpp)

find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Development)

set(CMAKE_CXX_FLAGS "--std=c++20")

if(TORCH_CXX_FLAGS)
  message(WARNING "setting global flags to ${TORCH_CXX_FLAGS}")
  add_definitions(${TORCH_CXX_FLAGS})
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# Set source directory
set(SOURCE_DIR ".")
# Tell CMake that headers are also in SOURCE_DIR
include_directories("${SOURCE_DIR}")

# bert tokenizer
set(TOKENIZER_SOURCES
        tokenization/ngram.h
        tokenization/ngram.cpp
        tokenization/text_normalizer.cpp
        tokenization/text_normalizer.h

        tokenization/bert_tokenizer/tokenization_bert_utils.h
        tokenization/bert_tokenizer/tokenization_bert_utils.cpp
        tokenization/bert_tokenizer/basic_tokenizer.h
        tokenization/bert_tokenizer/basic_tokenizer.cpp
        tokenization/bert_tokenizer/wordpiece_tokenizer.h
        tokenization/bert_tokenizer/wordpiece_tokenizer.cpp
        tokenization/bert_tokenizer/bert_tokenizer.h
        tokenization/bert_tokenizer/bert_tokenizer.cpp
)
set(OPERATORS_SOURCES
    operators/basic_tokenize.h
    operators/basic_tokenize.cpp
    operators/bert_tokenizer.h
    operators/bert_tokenizer.cpp
    operators/registry.cpp
    )

add_library(operators SHARED ${OPERATORS_SOURCES} ${TOKENIZER_SOURCES}
                             $<TARGET_OBJECTS:icuuc-object>)

# === linker opt for operators ===
# Enable C++11
target_compile_features(operators PRIVATE cxx_range_for)
# Link against LibTorch
target_link_libraries(operators "${TORCH_LIBRARIES}")
# Link against dl (for icu)
target_link_libraries(operators dl)
target_link_libraries(operators absl::flat_hash_map)
target_include_directories(
  operators
  PRIVATE $<TARGET_PROPERTY:icuuc-object,INTERFACE_INCLUDE_DIRECTORIES>)

install(
  TARGETS operators
  ARCHIVE DESTINATION .
  LIBRARY DESTINATION .)
