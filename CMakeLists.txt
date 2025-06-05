cmake_minimum_required(VERSION 3.18)
project(bitmap2svg_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA Architectures for Kaggle (T4: 75, P100: 60)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set, defaulting to '60;75' for common Kaggle GPUs.")
    set(CMAKE_CUDA_ARCHITECTURES "60;75")
endif()

# --- Options for local development / system search fallback ---
option(DEVSYS_USE_CUDA "Enable CUDA if found on system (for OpenCV, and Faiss if system Faiss is used)" OFF)
option(DEVSYS_USE_FAISS_GPU "Enable FAISS GPU if system Faiss is found and supports GPU (requires DEVSYS_USE_CUDA)" OFF)

# --- Python Build Environment Variables ---
set(PYTHON_FAISS_IS_GPU_ENABLED OFF CACHE BOOL "Whether the Python Faiss package is GPU enabled (from external script)")

# --- Final decision flags ---
set(WITH_CUDA_ENABLED FALSE)
set(WITH_FAISS_LIBRARY_FOUND FALSE)
set(WITH_FAISS_GPU_ENABLED FALSE)
set(FAISS_INCLUDE_DIRS_TO_USE "")
set(FAISS_LIBRARIES_TO_LINK "") 

# --- CUDA Toolkit Detection ---
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

find_package(Faiss QUIET COMPONENTS GPU)

if(Faiss_FOUND)
    message(STATUS "Found Faiss (via find_package): ${Faiss_VERSION_STRING}")
    set(FAISS_INCLUDE_DIRS_TO_USE ${Faiss_INCLUDE_DIRS})
    set(FAISS_LIBRARIES_TO_LINK Faiss::Faiss)
    set(WITH_FAISS_LIBRARY_FOUND TRUE)
    if(Faiss_GPU_FOUND OR ("${Faiss_VERSION_STRING}" MATCHES ".*gpu.*" AND Faiss_VERSION_MAJOR GREATER_EQUAL 1) AND WITH_CUDA_ENABLED)
        set(WITH_FAISS_GPU_ENABLED TRUE)
        message(STATUS "Faiss found is GPU enabled and CUDA is available.")
    else()
        message(STATUS "Faiss found is CPU only, or GPU components not detected/CUDA not available.")
    endif()
else()
    message(STATUS "Faiss not found via CMake find_package. Attempting manual paths (common for Kaggle/pip-installed Faiss-GPU setups).")

    set(FAISS_INCLUDE_CANDIDATES
        "${CMAKE_CURRENT_SOURCE_DIR}/external/faiss_headers"
        "/usr/local/include" # Faiss headers might be here: /usr/local/include/faiss/Index.h
        "/usr/include"
    )
    set(FAISS_LIB_CANDIDATES
        "/usr/local/lib"
        "/usr/lib"
    )

    find_path(FAISS_INCLUDE_DIR NAMES faiss/Index.h
              PATHS ${FAISS_INCLUDE_CANDIDATES}
              REQUIRED
              DOC "Path to Faiss C++ include directory containing faiss/Index.h"
    )

    find_library(FAISS_LIBRARY_PATH NAMES faiss_gpu faiss
                 PATHS ${FAISS_LIB_CANDIDATES}
                 REQUIRED
                 DOC "Path to Faiss C++ library file (e.g., libfaiss_gpu.so or libfaiss.so)"
    )

    if (FAISS_INCLUDE_DIR AND FAISS_LIBRARY_PATH)
        set(FAISS_INCLUDE_DIRS_TO_USE "${FAISS_INCLUDE_DIR}")
        set(FAISS_LIBRARIES_TO_LINK "${FAISS_LIBRARY_PATH}")
        set(WITH_FAISS_LIBRARY_FOUND TRUE)
        message(STATUS "Manually found Faiss C++ Include Dir: ${FAISS_INCLUDE_DIR}")
        message(STATUS "Manually found Faiss C++ Library Path: ${FAISS_LIBRARY_PATH}")

        if(PYTHON_FAISS_IS_GPU_ENABLED AND WITH_CUDA_ENABLED)
             set(WITH_FAISS_GPU_ENABLED TRUE)
             message(STATUS "Python Faiss indicated GPU enabled, and CUDA is available. Assuming GPU Faiss.")
        elseif("${FAISS_LIBRARY_PATH}" MATCHES "faiss_gpu" AND WITH_CUDA_ENABLED) # 检查库名
            set(WITH_FAISS_GPU_ENABLED TRUE)
            message(STATUS "Faiss library name indicates GPU, and CUDA is available. Assuming GPU Faiss.")
        else()
            message(STATUS "Faiss C++ library detected as CPU only, or CUDA not available.")
        endif()

        if(WITH_FAISS_GPU_ENABLED)
            target_compile_definitions(bitmap2svg_core PRIVATE FAISS_USE_GPU)
        endif()
    else()
        message(FATAL_ERROR "Faiss C++ library or include directory not found. Please ensure Faiss C++ development files are installed or adjust search paths.")
    endif()
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

if(WITH_FAISS_LIBRARY_FOUND AND FAISS_LIBRARIES_TO_LINK)
    target_link_libraries(bitmap2svg_core PRIVATE ${FAISS_LIBRARIES_TO_LINK})
    message(STATUS "Linking target with FAISS library/target: ${FAISS_LIBRARIES_TO_LINK}")

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
endif()

# --- Installation ---
install(TARGETS bitmap2svg_core LIBRARY DESTINATION bitmap2svg)

# --- Final Configuration Summary ---
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
