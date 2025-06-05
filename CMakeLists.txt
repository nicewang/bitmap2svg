cmake_minimum_required(VERSION 3.18) # Faiss might require 3.18+
project(bitmap2svg_cpp LANGUAGES CXX) # Removed CUDA language here, will add conditionally if true CUDA toolkit is found

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# For CUDA, ensure its standard matches or is compatible
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA Architectures for Kaggle (T4: 75, P100: 60)
# Check scikit-build-core docs for how it handles this; often set via env var or cmake args.
# If not set externally, provide a default.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, defaulting to '60;75' for common Kaggle GPUs.")
    set(CMAKE_CUDA_ARCHITECTURES "60;75")
endif()

# --- Options for local development / system search fallback ---
# For Kaggle, we'll generally disable system CUDA/Faiss GPU search by default
option(DEVSYS_USE_CUDA "Enable CUDA if found on system (for OpenCV, and Faiss if system Faiss is used)" OFF) # Default OFF for build environment
option(DEVSYS_USE_FAISS_GPU "Enable FAISS GPU if system Faiss is found and supports GPU (requires DEVSYS_USE_CUDA)" OFF) # Default OFF for build environment

# --- Variables from Python build environment (used for signaling GPU status) ---
# This variable is expected to be set by the Python build process if faiss-gpu is installed.
set(PYTHON_FAISS_IS_GPU_ENABLED OFF CACHE BOOL "Whether the Python Faiss package is GPU enabled")

# --- Final decision flags ---
set(WITH_CUDA_ENABLED FALSE)
set(WITH_FAISS_LIBRARY_FOUND FALSE)
set(WITH_FAISS_GPU_ENABLED FALSE)
set(FAISS_INCLUDE_DIRS_TO_USE "")
set(FAISS_LIBRARIES_TO_LINK "")

# --- CUDA Toolkit Detection ---
# This section tries to ensure CUDA is found if any component (Python Faiss GPU, System Faiss GPU, OpenCV CUDA) might need it.
set(CUDA_REQUESTED FALSE)
if(PYTHON_FAISS_IS_GPU_ENABLED)
    # If Python Faiss is GPU enabled, we need CUDA Toolkit for it.
    set(CUDA_REQUESTED TRUE)
    message(STATUS "Python Faiss is GPU enabled, requesting CUDA Toolkit.")
endif()
if(DEVSYS_USE_CUDA) # User explicitly wants system CUDA for OpenCV or system Faiss
    set(CUDA_REQUESTED TRUE)
    message(STATUS "DEVSYS_USE_CUDA is ON, requesting CUDA Toolkit.")
endif()

if(CUDA_REQUESTED)
    # Explicitly enable CUDA language only if needed and CUDA can be found.
    # This ensures CMAKE_CUDA_COMPILER is set and related variables are populated.
    find_package(CUDA QUIET) # Try to find CUDA Toolkit
    if(CUDA_FOUND) # If CUDA Toolkit is found, then enable the language
        enable_language(CUDA) # Enables CUDA compiler for the project
        set(WITH_CUDA_ENABLED TRUE)
        message(STATUS "CUDA Toolkit found and configured (via find_package and enable_language).")
        message(STATUS "  CUDA Root Dir (approx): ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "  CUDA Version (approx): ${CUDA_VERSION}")
        message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        if(NOT CUDA_INCLUDE_DIRS AND DEFINED CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES)
            set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
        endif()
        message(STATUS "  CUDA Include Dirs (approx): ${CUDA_INCLUDE_DIRS}")
    else()
        message(WARNING "CUDA Toolkit not found, but was requested (e.g., by Python Faiss GPU). CUDA-dependent features will be disabled.")
        if(PYTHON_FAISS_IS_GPU_ENABLED)
            message(WARNING "Python Faiss is GPU enabled, but CUDA Toolkit not found by CMake. FAISS GPU will likely fail.")
            set(PYTHON_FAISS_IS_GPU_ENABLED FALSE) # Disable GPU features if CUDA not found
        endif()
    endif()
else()
    message(STATUS "CUDA support is not requested for C++ compilation.")
endif()


# --- Faiss Detection (prioritizing manual headers and _swigfaiss.so) ---

# 1. Locate Faiss C++ Headers
# We assume headers are manually placed in a known location relative to the project root.
# Example: my_project/external/faiss_headers/faiss/Index.h
set(MANUAL_FAISS_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/faiss_headers")
message(STATUS "Attempting to find Faiss C++ headers in: ${MANUAL_FAISS_HEADERS_DIR}")

find_path(FAISS_INCLUDE_DIR_LOCATED
          NAMES faiss/Index.h
          PATHS "${MANUAL_FAISS_HEADERS_DIR}" # Prioritize manual location
          "/usr/local/include" # Fallback to common system paths
          "/usr/local/cuda/include"
          "/opt/conda/include"
          DOC "Path to Faiss C++ include directory containing faiss/Index.h"
)

if(FAISS_INCLUDE_DIR_LOCATED)
    set(FAISS_INCLUDE_DIRS_TO_USE "${FAISS_INCLUDE_DIR_LOCATED}")
    set(WITH_FAISS_LIBRARY_FOUND TRUE) # At least headers are found
    message(STATUS "Found Faiss C++ Headers at: ${FAISS_INCLUDE_DIRS_TO_USE}")
else()
    message(FATAL_ERROR "Faiss C++ Headers (faiss/Index.h) not found. Please ensure they are correctly placed in '${MANUAL_FAISS_HEADERS_DIR}' or a system path.")
endif()


# 2. Locate Faiss C++ Library (_swigfaiss.cpython-310-x86_64-linux-gnu.so)
# Based on previous debugging, the actual library to link is often _swigfaiss.cpython-310-x86_64-linux-gnu.so
# which is located within the pip installed faiss package.
set(FAISS_LIB_CANDIDATES
    "/usr/local/lib/python3.10/dist-packages/faiss"           # Specific path for _swigfaiss.so
    "/usr/local/lib/python3.10/dist-packages/faiss_gpu.libs"  # Common for underlying libfaiss.so if exists
    "/usr/local/lib"
    "/usr/local/cuda/lib64"
    "/opt/conda/lib"
    "/usr/lib"
)

# Use find_library to locate the _swigfaiss.cpython-310-x86_64-linux-gnu.so
# Note: The exact name including Python version and arch is crucial.
# We explicitly specify the full name of the .so file.
# CMake's find_library expects names without 'lib' prefix and '.so' suffix, but for exact files,
# we can provide the full filename if it's the only name we're looking for.
# However, using NAMES just 'faiss', 'faiss_gpu', or '_swigfaiss' is more conventional if CMake can resolve the full name.
# Given your specific output, let's try to match the exact file.
# For .so files, `find_library` usually prefers to find 'libNAME.so' if you give it 'NAME'.
# If the file is literally '_swigfaiss.cpython-310-x86_64-linux-gnu.so', we need to pass that exact string.
# A more robust way is to use `find_file` for an exact file or try different names with find_library.

# Let's try to find it by its full name first using find_file
find_file(FAISS_SWIG_LIBRARY_PATH
          NAMES _swigfaiss.cpython-310-x86_64-linux-gnu.so
          PATHS ${FAISS_LIB_CANDIDATES}
          DOC "Path to the _swigfaiss.cpython-310-x86_64-linux-gnu.so library"
)

if(FAISS_SWIG_LIBRARY_PATH)
    set(FAISS_LIBRARIES_TO_LINK "${FAISS_SWIG_LIBRARY_PATH}")
    message(STATUS "Found Faiss SWIG Library at: ${FAISS_LIBRARIES_TO_LINK}")
    set(WITH_FAISS_LIBRARY_FOUND TRUE) # Library found
    
    # Determine if it's GPU-enabled based on Python signal and CUDA availability
    if(PYTHON_FAISS_IS_GPU_ENABLED AND WITH_CUDA_ENABLED)
        set(WITH_FAISS_GPU_ENABLED TRUE)
        message(STATUS "Faiss library is SWIG (GPU enabled) and CUDA is available.")
    else()
        message(STATUS "Faiss SWIG library found, but either Python indicated CPU only, or CUDA Toolkit is NOT available. Faiss GPU features will be disabled for C++ compilation.")
    endif()
else()
    message(FATAL_ERROR "Faiss SWIG Library '_swigfaiss.cpython-310-x86_64-linux-gnu.so' not found. "
                        "Please verify its path or your faiss-gpu installation. Searched in: ${FAISS_LIB_CANDIDATES}")
endif()

# --- OpenCV ---
set(OpenCV_REQUIRED_COMPONENTS core imgproc)
find_package(OpenCV REQUIRED COMPONENTS ${OpenCV_REQUIRED_COMPONENTS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- pybind11 ---
find_package(pybind11 REQUIRED CONFIG)
message(STATUS "Pybind11 version: ${pybind11_VERSION}")

# --- Create Python Module ---
pybind11_add_module(bitmap2svg_core
    bitmap2svg/cpp/bindings.cpp
    bitmap2svg/cpp/bitmap_to_svg.cpp
)

# --- Include Directories for the Module ---
target_include_directories(bitmap2svg_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmap2svg/cpp" # For local headers like bitmap_to_svg.h
    ${OpenCV_INCLUDE_DIRS}
    # pybind11 include dirs are handled by pybind11_add_module / linking pybind11::module
)
if(WITH_FAISS_LIBRARY_FOUND AND FAISS_INCLUDE_DIRS_TO_USE)
    target_include_directories(bitmap2svg_core PUBLIC ${FAISS_INCLUDE_DIRS_TO_USE})
    message(STATUS "Adding FAISS include directory to target: ${FAISS_INCLUDE_DIRS_TO_USE}")
endif()

# Explicitly add CUDA include directories to the target
if(WITH_CUDA_ENABLED AND CUDA_INCLUDE_DIRS)
    message(STATUS "Explicitly adding CUDA_INCLUDE_DIRS to target bitmap2svg_core: ${CUDA_INCLUDE_DIRS}")
    target_include_directories(bitmap2svg_core SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
endif()


# --- Compile Definitions for C++ ---
if(WITH_CUDA_ENABLED)
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_CUDA)
endif()
if(WITH_FAISS_GPU_ENABLED) # True if Faiss lib found, is GPU, and CUDA is available
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_FAISS_GPU)
endif()

# --- Link Libraries to the Module ---
target_link_libraries(bitmap2svg_core PRIVATE
    pybind11::module
    ${OpenCV_LIBS}
)

if(WITH_FAISS_LIBRARY_FOUND AND FAISS_LIBRARIES_TO_LINK)
    target_link_libraries(bitmap2svg_core PRIVATE ${FAISS_LIBRARIES_TO_LINK})
    message(STATUS "Linking target with FAISS library/target: ${FAISS_LIBRARIES_TO_LINK}")

    # If Faiss GPU is enabled, we need to link to CUDA Runtime library
    if(WITH_FAISS_GPU_ENABLED AND WITH_CUDA_ENABLED)
        message(STATUS "Linking Faiss GPU features to CUDA Runtime.")
        find_package(CUDAToolkit QUIET COMPONENTS Runtime) # CUDAToolkit module is newer and might create these targets
        if(CUDAToolkit_FOUND AND TARGET CUDA::cudart)
            target_link_libraries(bitmap2svg_core PRIVATE CUDA::cudart)
            message(STATUS "  Linked with CUDA::cudart (modern CMake target).")
        else()
            # Fallback: Find cudart library directly and link
            find_library(CUDART_LIBRARY cudart PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib NO_DEFAULT_PATH)
            if(CUDART_LIBRARY)
                target_link_libraries(bitmap2svg_core PRIVATE ${CUDART_LIBRARY})
                message(STATUS "  Linked with cudart library found at: ${CUDART_LIBRARY} (fallback).")
            else()
                message(WARNING "  CUDA is enabled, but CUDA::cudart target was not found, and direct cudart library could not be found. Faiss GPU features might fail.")
            endif()
        endif()
    endif()
endif()

# --- Installation ---
install(TARGETS bitmap2svg_core LIBRARY DESTINATION bitmap2svg)

# --- Final Configuration Summary (condensed) ---
message(STATUS "--- Configuration Summary ---")
message(STATUS "Effective CUDA Toolkit Enabled (WITH_CUDA_ENABLED): ${WITH_CUDA_ENABLED}")
if(WITH_CUDA_ENABLED)
    message(STATUS "  CUDA Version (approx): ${CUDA_VERSION}, Dirs: ${CUDA_INCLUDE_DIRS}")
endif()
message(STATUS "Effective Faiss Library Found (WITH_FAISS_LIBRARY_FOUND): ${WITH_FAISS_LIBRARY_FOUND}")
if(WITH_FAISS_LIBRARY_FOUND)
    message(STATUS "  Effective Faiss GPU Enabled (WITH_FAISS_GPU_ENABLED): ${WITH_FAISS_GPU_ENABLED}")
    message(STATUS "  Faiss Include Dirs Used: ${FAISS_INCLUDE_DIRS_TO_USE}")
    message(STATUS "  Faiss Libraries Linked: ${FAISS_LIBRARIES_TO_LINK}")
endif()
message(STATUS "---------------------------")
