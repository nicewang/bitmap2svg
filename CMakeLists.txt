cmake_minimum_required(VERSION 3.15)
project(bitmap_to_svg C CXX)

# Find the Potrace library
# This requires Potrace development files to be installed on the build system.
find_package(Potrace REQUIRED)

# Add the C++ source files to create a shared library
add_library(bitmap_to_svg SHARED
    src/bitmap_to_svg.cpp
    src/bitmap_to_svg.h
)

# Link the shared library against Potrace
target_link_libraries(bitmap_to_svg PRIVATE Potrace::potrace)

# Set output directory for easier access from setuptools
# We'll build directly into a directory that setup.py can easily include.
# The exact path might need adjustment based on the setup.py build dir structure.
# Let's set it relative to the build directory. setup.py will handle the rest.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # For DLLs on Windows

# Handle vcpkg integration for Windows - relies on VCPKG_ROOT env var set by CI/user
if (MSVC)
   # Use vcpkg toolchain file if VCPKG_ROOT is set
   if (DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "Vcpkg toolchain file")
   endif()
   # Ensure MSVC runtime is linked statically or dynamically as needed
   # default is usually dynamic which is fine for wheels if VC runtime is present
endif()

# Optional: Set C++ standard
# set_property(TARGET bitmap_to_svg PROPERTY CXX_STANDARD 11)
# set_property(TARGET bitmap_to_svg PROPERTY CXX_STANDARD_REQUIRED ON)