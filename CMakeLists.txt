cmake_minimum_required(VERSION 3.18)
project(bitmap2svg_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, defaulting to '60;75' for common Kaggle GPUs.")
    set(CMAKE_CUDA_ARCHITECTURES "60;75")
endif()

option(DEVSYS_USE_CUDA "Enable CUDA if found on system (for OpenCV, and Faiss if system Faiss is used)" OFF)
option(DEVSYS_USE_FAISS_GPU "Enable FAISS GPU if system Faiss is found and supports GPU (requires DEVSYS_USE_CUDA)" OFF)

set(PYTHON_FAISS_IS_GPU_ENABLED ON CACHE BOOL "Whether the Python Faiss package is GPU enabled")

set(WITH_CUDA_ENABLED FALSE)
set(WITH_FAISS_LIBRARY_FOUND FALSE)
set(WITH_FAISS_GPU_ENABLED FALSE)
set(FAISS_INCLUDE_DIRS_TO_USE "")
# FAISS_LIBRARIES_TO_LINK will now be the name of our IMPORTED target
set(FAISS_LIBRARIES_TO_LINK "") # Initialize to empty


set(CUDA_REQUESTED FALSE)
if(PYTHON_FAISS_IS_GPU_ENABLED)
    set(CUDA_REQUESTED TRUE)
    message(STATUS "Python Faiss is GPU enabled, requesting CUDA Toolkit.")
endif()
if(DEVSYS_USE_CUDA)
    set(CUDA_REQUESTED TRUE)
    message(STATUS "DEVSYS_USE_CUDA is ON, requesting CUDA Toolkit.")
endif()

if(CUDA_REQUESTED)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
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
            set(PYTHON_FAISS_IS_GPU_ENABLED FALSE)
        endif()
    endif()
else()
    message(STATUS "CUDA support is not requested for C++ compilation.")
endif()


# --- Faiss Detection ---

# 1. Locate Faiss C++ Headers
set(MANUAL_FAISS_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/faiss_headers")
message(STATUS "Attempting to find Faiss C++ headers in: ${MANUAL_FAISS_HEADERS_DIR}")

find_path(FAISS_INCLUDE_DIR_LOCATED
          NAMES faiss/Index.h
          PATHS "${MANUAL_FAISS_HEADERS_DIR}"
          "/usr/local/include"
          "/usr/local/cuda/include"
          "/opt/conda/include"
          DOC "Path to Faiss C++ include directory containing faiss/Index.h"
)

if(FAISS_INCLUDE_DIR_LOCATED)
    set(FAISS_INCLUDE_DIRS_TO_USE "${FAISS_INCLUDE_DIR_LOCATED}")
    message(STATUS "Found Faiss C++ Headers at: ${FAISS_INCLUDE_DIRS_TO_USE}")
else()
    message(FATAL_ERROR "Faiss C++ Headers (faiss/Index.h) not found. Please ensure they are correctly placed in '${MANUAL_FAISS_HEADERS_DIR}' or a system path.")
endif()


# 2. Locate Faiss C++ Library (_swigfaiss.cpython-310-x86_64-linux-gnu.so)
set(FAISS_LIB_SEARCH_PATHS
    "/usr/local/lib/python3.10/dist-packages/faiss" # Specific path for _swigfaiss.so
    "/usr/local/lib/python3.10/dist-packages/faiss_gpu.libs" # Common for underlying libfaiss.so if exists
    "/usr/local/lib"
    "/usr/local/cuda/lib64"
    "/opt/conda/lib"
    "/usr/lib"
)

# Use find_file to locate the exact .so file by its full name
find_file(FAISS_SWIG_LIBRARY_PATH
          NAMES _swigfaiss.cpython-310-x86_64-linux-gnu.so
          PATHS ${FAISS_LIB_SEARCH_PATHS}
          DOC "Path to the _swigfaiss.cpython-310-x86_64-linux-gnu.so library"
)

if(FAISS_SWIG_LIBRARY_PATH)
    message(STATUS "Found Faiss SWIG Library at: ${FAISS_SWIG_LIBRARY_PATH}")

    # --- Define an IMPORTED Library Target for Faiss ---
    # This is the key to solving the linking issue with non-standard names.
    add_library(FaissPythonSwigLib SHARED IMPORTED)
    set_property(TARGET FaissPythonSwigLib PROPERTY IMPORTED_LOCATION "${FAISS_SWIG_LIBRARY_PATH}")
    # You might also need to set IMPORTED_LINK_INTERFACE_LIBRARIES if _swigfaiss.so itself has further dependencies
    # For now, let's assume it encapsulates its dependencies or they are implicitly handled.

    set(FAISS_LIBRARIES_TO_LINK "FaissPythonSwigLib") # Now link by the target name
    set(WITH_FAISS_LIBRARY_FOUND TRUE) # Library found

    if(PYTHON_FAISS_IS_GPU_ENABLED AND WITH_CUDA_ENABLED)
        set(WITH_FAISS_GPU_ENABLED TRUE)
        message(STATUS "Faiss library is SWIG (GPU enabled) and CUDA is available.")
        # If this imported library has explicit CUDA runtime dependencies that are not
        # resolved by the linker (e.g., if it expects -lcudart), you might need to
        # make it a dependency of this imported library itself, or link it later.
        # For now, we link cudart explicitly to bitmap2svg_core.
    else()
        message(STATUS "Faiss SWIG library found, but either Python indicated CPU only, or CUDA Toolkit is NOT available. Faiss GPU features will be disabled for C++ compilation.")
    endif()
else()
    message(FATAL_ERROR "Faiss SWIG Library '_swigfaiss.cpython-310-x86_64-linux-gnu.so' not found. "
                        "Please verify its path or your faiss-gpu installation. Searched in: ${FAISS_LIB_SEARCH_PATHS}")
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
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmap2svg/cpp"
    ${OpenCV_INCLUDE_DIRS}
)
if(WITH_FAISS_LIBRARY_FOUND AND FAISS_INCLUDE_DIRS_TO_USE)
    target_include_directories(bitmap2svg_core PUBLIC ${FAISS_INCLUDE_DIRS_TO_USE})
    message(STATUS "Adding FAISS include directory to target: ${FAISS_INCLUDE_DIRS_TO_USE}")
endif()

if(WITH_CUDA_ENABLED AND CUDA_INCLUDE_DIRS)
    message(STATUS "Explicitly adding CUDA_INCLUDE_DIRS to target bitmap2svg_core: ${CUDA_INCLUDE_DIRS}")
    target_include_directories(bitmap2svg_core SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
endif()


# --- Compile Definitions for C++ ---
if(WITH_CUDA_ENABLED)
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_CUDA)
endif()
if(WITH_FAISS_GPU_ENABLED)
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_FAISS_GPU)
endif()

# --- Link Libraries to the Module ---
target_link_libraries(bitmap2svg_core PRIVATE
    pybind11::module
    ${OpenCV_LIBS}
)

# Now, link by the name of the IMPORTED target.
# CMake will handle the full path and ensure proper linking without -l prefix issues.
if(WITH_FAISS_LIBRARY_FOUND AND FAISS_LIBRARIES_TO_LINK)
    target_link_libraries(bitmap2svg_core PRIVATE ${FAISS_LIBRARIES_TO_LINK})
    message(STATUS "Linking target with FAISS imported library: ${FAISS_LIBRARIES_TO_LINK}")

    if(WITH_FAISS_GPU_ENABLED AND WITH_CUDA_ENABLED)
        message(STATUS "Linking Faiss GPU features to CUDA Runtime.")
        # Ensure CUDA::cudart is linked.
        # It's crucial for any CUDA-dependent code, including Faiss's GPU usage.
        find_package(CUDAToolkit QUIET COMPONENTS Runtime)
        if(CUDAToolkit_FOUND AND TARGET CUDA::cudart)
            target_link_libraries(bitmap2svg_core PRIVATE CUDA::cudart)
            message(STATUS "  Linked with CUDA::cudart (modern CMake target).")
        else()
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

# --- Final Configuration Summary ---
message(STATUS "--- Configuration Summary ---")
message(STATUS "Effective CUDA Toolkit Enabled (WITH_CUDA_ENABLED): ${WITH_CUDA_ENABLED}")
if(WITH_CUDA_ENABLED)
    message(STATUS "  CUDA Root Dir (approx): ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "  CUDA Version (approx): ${CUDA_VERSION}, Dirs: ${CUDA_INCLUDE_DIRS}")
endif()
message(STATUS "Effective Faiss Library Found (WITH_FAISS_LIBRARY_FOUND): ${WITH_FAISS_LIBRARY_FOUND}")
if(WITH_FAISS_LIBRARY_FOUND)
    message(STATUS "  Effective Faiss GPU Enabled (WITH_FAISS_GPU_ENABLED): ${WITH_FAISS_GPU_ENABLED}")
    message(STATUS "  Faiss Include Dirs Used: ${FAISS_INCLUDE_DIRS_TO_USE}")
    message(STATUS "  Faiss Libraries Linked (Target Name): ${FAISS_LIBRARIES_TO_LINK}")
    message(STATUS "  Actual Faiss SWIG Library Path: ${FAISS_SWIG_LIBRARY_PATH}")
endif()
message(STATUS "---------------------------")
