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

add_library(bitmap2svg_cpp MODULE bitmap2svg/cpp/bindings.cpp bitmap2svg/cpp/bitmap_to_svg.cpp)

if(POTRACE_LIBRARY AND POTRACE_INCLUDE_DIR)
    target_include_directories(bitmap2svg_cpp PRIVATE ${POTRACE_INCLUDE_DIR} ${pybind11_INCLUDE_DIRS})
    target_link_libraries(bitmap2svg_cpp PRIVATE ${POTRACE_LIBRARY} ${PYTHON_LIBRARIES})
    set_target_properties(bitmap2svg_cpp PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})
else()
    message(FATAL_ERROR "Could not find potrace library or header file.")
endif()
