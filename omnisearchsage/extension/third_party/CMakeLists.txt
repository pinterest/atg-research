cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 11)

project(icuuc)

set(CMAKE_CXX_FLAGS "-DU_COMMON_IMPLEMENTATION")

set(SRC_FILES normalization_data.cc)

# Collect other source files using GLOB
file(GLOB COMMON_SRC_FILES
  icu/icu4c/source/common/*.c
  icu/icu4c/source/common/*.cpp
  icu/icu4c/source/stubdata/*.cpp
)

list(APPEND SRC_FILES ${COMMON_SRC_FILES})

add_library(icuuc-object OBJECT ${SRC_FILES})
target_include_directories(icuuc-object PUBLIC icu/icu4c/source/common)

set_property(TARGET icuuc-object PROPERTY POSITION_INDEPENDENT_CODE ON)
