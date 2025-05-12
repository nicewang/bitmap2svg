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
pkg_check_modules(POTRACE REQUIRED IMPORTED_TARGET potrace)

pybind11_add_module(bitmap2svg_cpp bitmap2svg/cpp/bindings.cpp)
target_link_libraries(bitmap2svg_cpp PRIVATE PkgConfig::POTRACE)
