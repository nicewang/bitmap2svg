cmake_minimum_required(VERSION 3.15)
project(bitmap2svg LANGUAGES CXX)

# Fetch pybind11 from GitHub
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find Python 3 interpreter and development libraries
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find potrace library and header
find_library(POTRACE_LIBRARY NAMES potrace HINTS /usr/lib /usr/lib/x86_64-linux-gnu)
find_path(POTRACE_INCLUDE_DIR NAMES potracelib.h HINTS /usr/include)

# Fail early if not found
if(NOT POTRACE_LIBRARY OR NOT POTRACE_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find potrace library or headers")
endif()

# Add Python module target using pybind11
pybind11_add_module(bitmap2svg_cpp
    bitmap2svg/cpp/bindings.cpp
    bitmap2svg/cpp/bitmap_to_svg.cpp
)

# Include potrace headers
target_include_directories(bitmap2svg_cpp PRIVATE
    ${POTRACE_INCLUDE_DIR}
)

# Link with potrace and Python
target_link_libraries(bitmap2svg_cpp PRIVATE
    ${POTRACE_LIBRARY}
    Python3::Python
)
