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
             HINTS /usr/lib /usr/local/lib)
find_path(POTRACE_INCLUDE_DIR NAMES potracelib.h
          HINTS /usr/include /usr/local/include)

if(NOT POTRACE_LIBRARY OR NOT POTRACE_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find Potrace library or header file.")
endif()

add_library(bitmap2svg_cpp MODULE
    bitmap2svg/cpp/bindings.cpp
    bitmap2svg/cpp/bitmap_to_svg.cpp
)

target_include_directories(bitmap2svg_cpp
    PRIVATE
        ${POTRACE_INCLUDE_DIR}
        ${pybind11_INCLUDE_DIRS}
)

target_link_libraries(bitmap2svg_cpp
    PRIVATE
        ${POTRACE_LIBRARY}
        pybind11::module
)

set_target_properties(bitmap2svg_cpp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
