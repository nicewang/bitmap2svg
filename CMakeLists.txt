cmake_minimum_required(VERSION 3.15)
project(bitmap_to_svg C CXX)

# Find the PkgConfig tool
find_package(PkgConfig REQUIRED)

# Use PkgConfig to find the Potrace library based on its .pc file (potrace.pc)
# This is usually more reliable for libraries installed via package managers.
pkg_check_modules(POTRACE REQUIRED potrace)
# If the above line fails, it means pkg-config can't find 'potrace.pc'.
# Potential causes: pkg-config not installed, PKG_CONFIG_PATH not set correctly, or potrace.pc is missing/in a non-standard location.

# Add the include directories and link the libraries found by PkgConfig
# POTRACE_INCLUDE_DIRS and POTRACE_LIBRARIES are variables populated by pkg_check_modules
include_directories(${POTRACE_INCLUDE_DIRS}) # Add Potrace header paths
# Note: PkgConfig finds the library file(s), pkg_check_modules populates POTRACE_LIBRARIES
# We link to the libraries found by PkgConfig
link_libraries(${POTRACE_LIBRARIES})


add_library(bitmap_to_svg SHARED
    src/bitmap_to_svg.cpp
    src/bitmap_to_svg.h
)

# Link our target library against the Potrace libraries found by PkgConfig
# Use PRIVATE because your library links to Potrace, but users of your Python library don't directly link to Potrace
target_link_libraries(bitmap_to_svg PRIVATE Potrace::potrace) # Original approach, might still work after find, but PkgConfig vars are more direct
# OR more directly use the PkgConfig variables:
# target_link_libraries(bitmap_to_svg PRIVATE ${POTRACE_LIBRARIES})

# Let's stick to using the PkgConfig variables found by pkg_check_modules, it's more explicit
target_link_libraries(bitmap_to_svg PRIVATE ${POTRACE_LIBRARIES})


# Set output directory for easier access from setuptools
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # For DLLs on Windows

# Handle vcpkg integration for Windows - relies on VCPKG_ROOT env var set by CI/user
if (MSVC)
   # Use vcpkg toolchain file if VCPKG_ROOT is set
   if (DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "Vcpkg toolchain file")
   endif()
endif()

# Optional: Set C++ standard
# set_property(TARGET bitmap_to_svg PROPERTY CXX_STANDARD 11)
# set_property(TARGET bitmap_to_svg PROPERTY CXX_STANDARD_REQUIRED ON)