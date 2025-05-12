cmake_minimum_required(VERSION 3.15)
project(bitmap2svg LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

find_package(PkgConfig REQUIRED)
pkg_check_modules(POTRACE REQUIRED potrace)

pybind11_add_module(bitmap2svg_cpp bitmap2svg/cpp/bindings.cpp)

target_include_directories(bitmap2svg_cpp PRIVATE ${POTRACE_INCLUDE_DIRS})
target_link_libraries(bitmap2svg_cpp PRIVATE ${POTRACE_LIBRARIES})

set_target_properties(bitmap2svg_cpp PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})
