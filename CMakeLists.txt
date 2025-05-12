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

# Find Python
find_package(Potrace REQUIRED
    HINTS
        ${CMAKE_INSTALL_PREFIX}
        /usr
        /usr/local
        /opt/local
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
    PATH_SUFFIXES
        lib
        lib64
        cmake/Potrace
        share/Potrace/cmake
        Potrace/cmake
)

# If find_package fails, try find_library and find_path as a fallback
if(NOT Potrace_FOUND)
    find_library(POTRACE_LIBRARY NAMES potrace
                 HINTS
                     /usr/lib
                     /usr/lib/x86_64-linux-gnu
                     /usr/local/lib)
    find_path(POTRACE_INCLUDE_DIR NAMES potracelib.h
              HINTS
                  /usr/include
                  /usr/local/include
                  /usr/include/potrace) # Some distributions might put the header in a subdirectory

    if(POTRACE_LIBRARY AND POTRACE_INCLUDE_DIR)
        message(STATUS "Potrace found using find_library/find_path")
        set(Potrace_FOUND TRUE)
        set(Potrace_INCLUDE_DIRS ${POTRACE_INCLUDE_DIR})
        set(Potrace_LIBRARIES ${POTRACE_LIBRARY})
    else()
        message(FATAL_ERROR "Could not find Potrace library. Please ensure it is installed and CMake can find it.")
    endif()
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
    ${CMAKE_CURRENT_SOURCE_DIR}/bitmap2svg/cpp
    ${Potrace_INCLUDE_DIRS}
)

target_link_libraries(bitmap2svg_cpp PRIVATE
    pybind11::module
    ${Potrace_LIBRARIES}
)
