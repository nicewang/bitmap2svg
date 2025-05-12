cmake_minimum_required(VERSION 3.15)
project(bitmap2svg LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

find_library(POTRACE_LIBRARY NAMES potrace
             HINTS /usr/lib/x86_64-linux-gnu)
find_path(POTRACE_INCLUDE_DIR NAMES potracelib.h
          HINTS /usr/include)

pybind11_add_module(bitmap2svg_cpp bitmap2svg/cpp/bindings.cpp)

if(POTRACE_LIBRARY AND POTRACE_INCLUDE_DIR)
    target_include_directories(bitmap2svg_cpp PRIVATE ${POTRACE_INCLUDE_DIR})
    target_link_libraries(bitmap2svg_cpp PRIVATE ${POTRACE_LIBRARY})
    target_link_libraries(bitmap2svg_cpp PRIVATE potrace)
else()
    message(FATAL_ERROR "Could not find potrace library or header file.")
endif()