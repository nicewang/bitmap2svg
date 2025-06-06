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

option(DEVSYS_USE_CUDA "Enable CUDA if found on system (for OpenCV, and Faiss if system Faiss is used)" ON)

set(WITH_CUDA_ENABLED FALSE)
set(WITH_FAISS_LIBRARY_FOUND FALSE)
set(WITH_FAISS_GPU_ENABLED FALSE)
set(FAISS_INCLUDE_DIRS_TO_USE "")
set(FAISS_LIBRARIES_TO_LINK "") # Initialize to empty


# --- CUDA Toolkit Detection and Enablement ---
if(DEVSYS_USE_CUDA)
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
        message(WARNING "CUDA Toolkit not found, but DEVSYS_USE_CUDA is ON. CUDA-dependent features will be disabled.")
        set(DEVSYS_USE_CUDA FALSE) # 禁用 CUDA
    endif()
else()
    message(STATUS "CUDA support is explicitly disabled for C++ compilation.")
endif()


# --- Faiss Integration (using add_subdirectory) ---
set(FAISS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/faiss_sources")

if(EXISTS "${FAISS_SOURCE_DIR}/CMakeLists.txt") # 确保 Faiss 源码目录存在
    message(STATUS "Found Faiss source directory: ${FAISS_SOURCE_DIR}")

    # Pass relevant options to Faiss's CMakeLists.txt
    # IMPORTANT: These options are passed as cached variables to the subdirectory
    # They control how Faiss itself is built.
    set(FAISS_ENABLE_GPU TRUE CACHE BOOL "Enable GPU support in Faiss" FORCE)
    set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Python bindings build for Faiss (handled by pip)" FORCE)
    set(FAISS_BUILD_TESTING OFF CACHE BOOL "Disable Faiss testing build" FORCE)
    set(FAISS_BUILD_BENCHMARKS OFF CACHE BOOL "Disable Faiss benchmarks build" FORCE)
    set(FAISS_OPT_LEVEL "O3" CACHE STRING "Faiss optimization level" FORCE) # Optional: match your project's O3

    # If you need specific BLAS libraries for Faiss, you might set them here:
    # set(BLAS_LIBRARIES "-lmkl_rt" CACHE STRING "BLAS library for Faiss" FORCE) # Example for MKL
    # set(LAPACK_LIBRARIES "-lmkl_rt" CACHE STRING "LAPACK library for Faiss" FORCE) # Example for MKL

    # # Add Faiss as a subdirectory. This will execute Faiss's CMakeLists.txt
    # add_subdirectory(${FAISS_SOURCE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/faiss_build")

    # ===== Debug =====
    set(FAISS_PERSISTENT_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/faiss_build_logs_and_cache")
    message(STATUS "Faiss will build into a persistent directory for debugging: ${FAISS_PERSISTENT_BUILD_DIR}")
    file(MAKE_DIRECTORY "${FAISS_PERSISTENT_BUILD_DIR}")

    add_subdirectory(${FAISS_SOURCE_DIR} "${FAISS_PERSISTENT_BUILD_DIR}")
    # ===== Debug End =====

    # After add_subdirectory, Faiss's targets should be available.
    # The GPU target name is typically 'faiss_gpu', CPU is 'faiss'
    if(TARGET faiss_gpu AND WITH_CUDA_ENABLED)
        set(FAISS_TARGET_TO_LINK faiss_gpu)
        set(WITH_FAISS_GPU_ENABLED TRUE)
        message(STATUS "Faiss GPU target 'faiss_gpu' found and will be linked.")
    elseif(TARGET faiss)
        set(FAISS_TARGET_TO_LINK faiss)
        message(STATUS "Faiss CPU target 'faiss' found and will be linked.")
    else()
        message(FATAL_ERROR "Faiss targets (faiss_gpu or faiss) not found after adding subdirectory. Check Faiss build configuration.")
    endif()

    # Set common variables for linking
    set(FAISS_LIBRARIES_TO_LINK ${FAISS_TARGET_TO_LINK})
    set(WITH_FAISS_LIBRARY_FOUND TRUE)

    # Faiss's include directories are usually handled by its targets, but
    # for explicit inclusion in your C++ files, you might still need to add them.
    # You might need to adjust this based on how Faiss exports its includes.
    # A common pattern is to include the source directory itself if headers are relative to it.
    set(FAISS_INCLUDE_DIRS_TO_USE "${FAISS_SOURCE_DIR}/faiss") # e.g., for #include <faiss/Index.h>
    message(STATUS "Faiss C++ Headers (from source) at: ${FAISS_INCLUDE_DIRS_TO_USE}")

else()
    message(FATAL_ERROR "Faiss source directory not found at: ${FAISS_SOURCE_DIR}. Please ensure Faiss source is placed there.")
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
    ${FAISS_INCLUDE_DIRS_TO_USE} # Add Faiss includes from source
)

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
    ${FAISS_LIBRARIES_TO_LINK} # Link the internally built Faiss target
)

# If Faiss GPU is enabled, link CUDA runtime library explicitly.
# This is crucial as bitmap2svg_core directly uses Faiss GPU.
if(WITH_FAISS_GPU_ENABLED AND WITH_CUDA_ENABLED)
    message(STATUS "Linking Faiss GPU features to CUDA Runtime.")
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
endif()
message(STATUS "---------------------------")
