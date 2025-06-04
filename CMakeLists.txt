cmake_minimum_required(VERSION 3.18) # Faiss might require 3.18+
project(bitmap2svg_cpp LANGUAGES CXX CUDA) # Add CUDA as a language here if not already

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
option(DEVSYS_USE_CUDA "Enable CUDA if found on system (for OpenCV, and Faiss if system Faiss is used)" ON)
option(DEVSYS_USE_FAISS_GPU "Enable FAISS GPU if system Faiss is found and supports GPU (requires DEVSYS_USE_CUDA)" ON)

# --- Variables from Python build environment ---
set(PYTHON_FAISS_INCLUDE_DIR "" CACHE PATH "Path to Faiss include directory from Python package")
set(PYTHON_FAISS_LIBRARY_PATH "" CACHE FILEPATH "Path to Faiss library file from Python package")
set(PYTHON_FAISS_IS_GPU_ENABLED OFF CACHE BOOL "Whether the Python Faiss package is GPU enabled")
option(MYPROJECT_PREFER_PYTHON_FAISS "Prefer Faiss from Python environment if paths are provided" ON)

# --- Final decision flags ---
set(WITH_CUDA_ENABLED FALSE)
set(WITH_FAISS_LIBRARY_FOUND FALSE)
set(WITH_FAISS_GPU_ENABLED FALSE)
set(FAISS_INCLUDE_DIRS_TO_USE "")
set(FAISS_LIBRARIES_TO_LINK "")

# --- CUDA Toolkit Detection ---
# This section tries to ensure CUDA is found if any component (Python Faiss GPU, System Faiss GPU, OpenCV CUDA) might need it.
set(CUDA_REQUESTED FALSE)
if(MYPROJECT_PREFER_PYTHON_FAISS AND PYTHON_FAISS_IS_GPU_ENABLED)
    set(CUDA_REQUESTED TRUE)
endif()
if(DEVSYS_USE_CUDA) # User explicitly wants system CUDA for OpenCV or system Faiss
    set(CUDA_REQUESTED TRUE)
endif()

if(CUDA_REQUESTED)
    # Enabling CUDA language at the project level handles find_package(CUDA) implicitly in modern CMake.
    # However, an explicit find_package can still be useful for messages or specific checks.
    if(NOT CMAKE_CUDA_COMPILER) # If CUDA language was not enabled at project level or failed
        find_package(CUDA QUIET) # Try to find it
    endif()

    if(CUDA_FOUND OR CMAKE_CUDA_COMPILER_LOADED) # CMAKE_CUDA_COMPILER_LOADED is true if enable_language(CUDA) worked
        set(WITH_CUDA_ENABLED TRUE)
        # CUDA_TOOLKIT_ROOT_DIR and CUDA_VERSION should be set by FindCUDA or enable_language(CUDA)
        if(NOT CUDA_TOOLKIT_ROOT_DIR AND DEFINED CMAKE_CUDA_TOOLKIT_ROOT_DIR) # CMake 3.20+ sets this
           set(CUDA_TOOLKIT_ROOT_DIR ${CMAKE_CUDA_TOOLKIT_ROOT_DIR})
        endif()
        if(NOT CUDA_VERSION AND DEFINED CMAKE_CUDA_COMPILER_VERSION)
            set(CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION})
        endif()
        message(STATUS "CUDA Toolkit appears to be configured.")
        message(STATUS "  CUDA Root Dir (approx): ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "  CUDA Version (approx): ${CUDA_VERSION}")
        message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        # CUDA_INCLUDE_DIRS should be set by now.
        if(NOT CUDA_INCLUDE_DIRS AND DEFINED CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES)
            set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES})
        endif()
        message(STATUS "  CUDA Include Dirs (approx): ${CUDA_INCLUDE_DIRS}")

    else()
        message(WARNING "CUDA Toolkit not found, but was requested. CUDA-dependent features will be disabled.")
        if(PYTHON_FAISS_IS_GPU_ENABLED)
            message(WARNING "Python Faiss is GPU enabled, but CUDA Toolkit not found by CMake. FAISS GPU will likely fail.")
        endif()
    endif()
else()
    message(STATUS "CUDA support is not requested.")
endif()


# --- Faiss Detection (prioritizing Python environment) ---
# ensure it correctly sets WITH_FAISS_LIBRARY_FOUND, FAISS_INCLUDE_DIRS_TO_USE,
# FAISS_LIBRARIES_TO_LINK, and WITH_FAISS_GPU_ENABLED based on Python paths or system fallback)
# Make sure that if Python Faiss is GPU enabled, WITH_FAISS_GPU_ENABLED is only TRUE if WITH_CUDA_ENABLED is also TRUE.
if(MYPROJECT_PREFER_PYTHON_FAISS AND PYTHON_FAISS_INCLUDE_DIR AND PYTHON_FAISS_LIBRARY_PATH)
    message(STATUS "Attempting to use Faiss from Python environment.")
    if(EXISTS "${PYTHON_FAISS_INCLUDE_DIR}" AND EXISTS "${PYTHON_FAISS_LIBRARY_PATH}")
        set(FAISS_INCLUDE_DIRS_TO_USE "${PYTHON_FAISS_INCLUDE_DIR}")
        set(FAISS_LIBRARIES_TO_LINK "${PYTHON_FAISS_LIBRARY_PATH}")
        set(WITH_FAISS_LIBRARY_FOUND TRUE)
        message(STATUS "Found Faiss from Python environment:")
        message(STATUS "  Include Dir: ${PYTHON_FAISS_INCLUDE_DIR}")
        message(STATUS "  Library Path: ${PYTHON_FAISS_LIBRARY_PATH}")
        if(PYTHON_FAISS_IS_GPU_ENABLED)
            if(WITH_CUDA_ENABLED) # Crucial check!
                set(WITH_FAISS_GPU_ENABLED TRUE)
                message(STATUS "  Python Faiss is GPU enabled and CUDA is available.")
            else()
                message(WARNING "Python Faiss is GPU enabled, but CUDA is NOT available/configured. Disabling FAISS GPU features.")
            endif()
        else()
            message(STATUS "  Python Faiss is CPU only.")
        endif()
    else()
        message(WARNING "Python Faiss paths provided but invalid/not found. Will attempt system search if enabled.")
    endif()
# Ensure this fallback also checks WITH_CUDA_ENABLED before setting WITH_FAISS_GPU_ENABLED.
# Example for system Faiss fallback (simplified):
elif(DEVSYS_USE_FAISS_GPU) # Fallback to system Faiss search
    if(WITH_CUDA_ENABLED)
        find_package(Faiss QUIET)
        if(FAISS_FOUND)
            # set(FAISS_INCLUDE_DIRS_TO_USE ${Faiss_INCLUDE_DIRS})
            # set(FAISS_LIBRARIES_TO_LINK ${FAISS_SYSTEM_LINK_TARGET})
            # set(WITH_FAISS_LIBRARY_FOUND TRUE)
            # set(WITH_FAISS_GPU_ENABLED TRUE) # If system Faiss GPU target is found
            message(STATUS "System Faiss found and configured for GPU (ensure your logic here is complete).")
        else()
            message(WARNING "DEVSYS_USE_FAISS_GPU is ON but system FAISS library not found for GPU.")
        endif()
    else()
        message(WARNING "DEVSYS_USE_FAISS_GPU is ON but CUDA is not available. Cannot use system FAISS GPU.")
    endif()
endif()


# --- OpenCV ---
# (Largely same, ensure it respects WITH_CUDA_ENABLED for requesting CUDA components)
set(OpenCV_REQUIRED_COMPONENTS core imgproc)
if(WITH_CUDA_ENABLED) # Only request OpenCV CUDA modules if our CUDA is enabled
    list(APPEND OpenCV_REQUIRED_COMPONENTS cudaarithm cudaimgproc cudev)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${OpenCV_REQUIRED_COMPONENTS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
if(WITH_CUDA_ENABLED AND NOT OpenCV_cudaarithm_FOUND) 
    message(WARNING "OpenCV CUDA components (e.g., cudaarithm) not found in your OpenCV build, despite CUDA being available. OpenCV operations will be CPU-only.")
    # Note: We don't set WITH_CUDA_ENABLED to FALSE here, as Faiss GPU might still be usable independently.
    # The C++ code might have #ifdefs for OpenCV CUDA parts.
endif()


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

# *** Explicitly add CUDA include directories to the target ***
# This is a key addition to try and resolve the missing cuda_fp16.h / cuda_bf16.h
if(WITH_CUDA_ENABLED AND CUDA_INCLUDE_DIRS)
    message(STATUS "Explicitly adding CUDA_INCLUDE_DIRS to target bitmap2svg_core: ${CUDA_INCLUDE_DIRS}")
    target_include_directories(bitmap2svg_core SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
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
endif()

if(WITH_CUDA_ENABLED) # If general CUDA is active and configured
    # Linking to CUDA::cudart should bring in necessary CUDA runtime libraries
    # and also often helps with include paths for CUDA headers.
    target_link_libraries(bitmap2svg_core PRIVATE CUDA::cudart)
    message(STATUS "Linking target with CUDA::cudart.")
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
endif()
message(STATUS "---------------------------")
