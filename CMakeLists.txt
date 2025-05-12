cmake_minimum_required(VERSION 3.15)
project(bitmap2svg LANGUAGES CXX)

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find potrace
find_library(POTRACE_LIBRARY NAMES potrace
             HINTS /usr/lib /usr/lib/x86_64-linux-gnu)
find_path(POTRACE_INCLUDE_DIR NAMES potracelib.h
          HINTS /usr/include /usr/local/include)

if(NOT POTRACE_LIBRARY OR NOT POTRACE_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find potrace library or header file.")
endif()

# Add C++ extension module
add_library(bitmap2svg_cpp MODULE
    bitmap2svg/cpp/bindings.cpp
    bitmap2svg/cpp/bitmap_to_svg.cpp
)

# Set correct extension for Python module
set_target_properties(bitmap2svg_cpp PROPERTIES
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

# Include paths and link libraries
target_include_directories(bitmap2svg_cpp PRIVATE
    ${POTRACE_INCLUDE_DIR}
)

target_link_libraries(bitmap2svg_cpp PRIVATE
    pybind11::module
    ${POTRACE_LIBRARY}
)
