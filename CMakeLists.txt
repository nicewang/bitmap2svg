cmake_minimum_required(VERSION 3.18) # Faiss might require 3.18+
project(bitmap2svg_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Options for local development / system search fallback ---
# These options primarily guide how CMake searches for dependencies if they are NOT provided by the Python build environment.
option(DEVSYS_USE_CUDA "Enable CUDA if found on system (for OpenCV, and Faiss if system Faiss is used)" ON)
option(DEVSYS_USE_FAISS_GPU "Enable FAISS GPU if system Faiss is found and supports GPU (requires DEVSYS_USE_CUDA)" ON)

# --- Variables to be potentially set by Python build environment (e.g., setup.py) ---
# These are paths and flags derived from the Python 'faiss' package.
# Example usage in setup.py:
# cmake_args.append(f"-DPYTHON_FAISS_INCLUDE_DIR={faiss.get_include()}")
# cmake_args.append(f"-DPYTHON_FAISS_LIBRARY_PATH={faiss.get_library_path()}")
# cmake_args.append(f"-DPYTHON_FAISS_IS_GPU_ENABLED={'ON' if is_gpu_version else 'OFF'}")
set(PYTHON_FAISS_INCLUDE_DIR "" CACHE PATH "Path to Faiss include directory from Python package")
set(PYTHON_FAISS_LIBRARY_PATH "" CACHE FILEPATH "Path to Faiss library file from Python package")
set(PYTHON_FAISS_IS_GPU_ENABLED OFF CACHE BOOL "Whether the Python Faiss package is GPU enabled")
option(MYPROJECT_PREFER_PYTHON_FAISS "Prefer Faiss from Python environment if paths are provided" ON)

# --- Final decision flags ---
set(WITH_CUDA_ENABLED FALSE)         # Will be true if CUDA toolkit is found and enabled for use.
set(WITH_FAISS_LIBRARY_FOUND FALSE)  # Will be true if any Faiss library (Python or system) is found.
set(WITH_FAISS_GPU_ENABLED FALSE)    # Will be true if the found Faiss library is GPU enabled and CUDA is available.
set(FAISS_INCLUDE_DIRS_TO_USE "")
set(FAISS_LIBRARIES_TO_LINK "")


# --- CUDA Toolkit Detection (primarily for OpenCV CUDA and system Faiss GPU) ---
if(DEVSYS_USE_CUDA OR (MYPROJECT_PREFER_PYTHON_FAISS AND PYTHON_FAISS_IS_GPU_ENABLED))
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDA_TOOLKIT_ROOT_DIR} (Version: ${CUDA_VERSION_STRING})")
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(WITH_CUDA_ENABLED TRUE)
    else()
        message(STATUS "CUDA Toolkit not found. CUDA-dependent features will be disabled.")
        if(PYTHON_FAISS_IS_GPU_ENABLED)
            message(WARNING "Python Faiss is GPU enabled, but CUDA Toolkit not found by CMake. FAISS GPU might fail.")
        endif()
    endif()
else()
    message(STATUS "CUDA support is not requested (DEVSYS_USE_CUDA=OFF and Python Faiss not GPU or not preferred).")
endif()


# --- Faiss Detection ---
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
            if(WITH_CUDA_ENABLED)
                set(WITH_FAISS_GPU_ENABLED TRUE)
                message(STATUS "  Python Faiss is GPU enabled and CUDA is available.")
            else()
                message(WARNING "Python Faiss is GPU enabled, but CUDA is NOT available/enabled. Disabling FAISS GPU features.")
            endif()
        else()
            message(STATUS "  Python Faiss is CPU only.")
        endif()
    else()
        message(WARNING "Python Faiss paths provided but invalid/not found. Will attempt system search if enabled.")
    endif()
endif()

if(NOT WITH_FAISS_LIBRARY_FOUND AND DEVSYS_USE_FAISS_GPU) # Fallback to system Faiss search if Python Faiss not used/found
    message(STATUS "Attempting to find system-installed FAISS for GPU usage (DEVSYS_USE_FAISS_GPU=ON).")
    if(NOT WITH_CUDA_ENABLED)
        message(WARNING "DEVSYS_USE_FAISS_GPU is ON, but CUDA is not available. System FAISS GPU search will likely fail or be ineffective.")
    else()
        # Standard find_package for system Faiss
        find_package(Faiss QUIET)
        if(FAISS_FOUND)
            message(STATUS "System FAISS found.")
            set(FAISS_INCLUDE_DIRS_TO_USE ${Faiss_INCLUDE_DIRS}) # Standard variable from FindFaiss.cmake
            
            # Logic to find a suitable GPU-enabled FAISS CMake target
            set(FAISS_SYSTEM_LINK_TARGET "")
            if(TARGET faiss::faiss_gpu)           set(FAISS_SYSTEM_LINK_TARGET faiss::faiss_gpu)
            elseif(TARGET faiss_gpu)              set(FAISS_SYSTEM_LINK_TARGET faiss_gpu)
            # Add more known targets if necessary, or check Faiss_LIBRARIES variable
            elseif(TARGET faiss::faiss) # Generic, might be GPU
                # Heuristic: if Faiss_LIBRARIES contains "gpu", assume it's a GPU build
                if("${Faiss_LIBRARIES}" MATCHES "[Gg][Pp][Uu]")
                    set(FAISS_SYSTEM_LINK_TARGET faiss::faiss)
                    message(STATUS "System target faiss::faiss found, and Faiss_LIBRARIES ('${Faiss_LIBRARIES}') suggests GPU support.")
                else()
                     message(STATUS "System target faiss::faiss found, but Faiss_LIBRARIES ('${Faiss_LIBRARIES}') does not clearly indicate GPU support. Assuming CPU.")
                endif()
            endif()

            if(NOT FAISS_SYSTEM_LINK_TARGET STREQUAL "")
                set(FAISS_LIBRARIES_TO_LINK ${FAISS_SYSTEM_LINK_TARGET})
                set(WITH_FAISS_LIBRARY_FOUND TRUE)
                set(WITH_FAISS_GPU_ENABLED TRUE) # Assumed if system GPU search was intended and a target found
                message(STATUS "Will link system FAISS against '${FAISS_SYSTEM_LINK_TARGET}' for GPU support.")
            else()
                message(WARNING "System FAISS found, but no recognizable GPU-enabled CMake target identified. Disabling FAISS GPU for system Faiss.")
            endif()
        else()
            message(WARNING "DEVSYS_USE_FAISS_GPU is ON but system FAISS library not found.")
        endif()
    endif()
elif(NOT WITH_FAISS_LIBRARY_FOUND) # If still not found, and system search for GPU was not explicitly ON.
    message(STATUS "Faiss library not provided by Python environment and DEVSYS_USE_FAISS_GPU is OFF. No Faiss will be used unless found by a generic find_package(Faiss) if your C++ code has a non-GPU Faiss path (not present in current C++).")
    # For now, if not found via Python or specific system GPU search, assume no Faiss.
endif()


# --- OpenCV ---
set(OpenCV_REQUIRED_COMPONENTS core imgproc)
if(WITH_CUDA_ENABLED) # Only request OpenCV CUDA modules if our CUDA is enabled
    list(APPEND OpenCV_REQUIRED_COMPONENTS cudaarithm cudaimgproc cudev)
endif()

find_package(OpenCV REQUIRED COMPONENTS ${OpenCV_REQUIRED_COMPONENTS})
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV (with required components) not found!")
endif()
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Refined check: if WITH_CUDA_ENABLED was true, but OpenCV didn't find its CUDA modules,
# then we cannot use OpenCV's CUDA capabilities.
if(WITH_CUDA_ENABLED AND NOT OpenCV_cudaarithm_FOUND) # Check if a core OpenCV CUDA module was actually found
    message(WARNING "OpenCV found, but required CUDA components (e.g., cudaarithm) seem missing in your OpenCV build. Disabling WITH_CUDA_ENABLED for OpenCV specific paths.")
    set(WITH_CUDA_ENABLED FALSE) # Downgrade our flag if OpenCV CUDA parts are missing
    if(WITH_FAISS_GPU_ENABLED)
         message(WARNING "FAISS GPU (if found and enabled) might still function if it has its own CUDA handling, but OpenCV operations won't be CUDA accelerated.")
    endif()
endif()


# --- pybind11 ---
find_package(pybind11 REQUIRED CONFIG) # Use CONFIG mode for modern pybind11
message(STATUS "Pybind11 version: ${pybind11_VERSION}")


# --- Create Python Module ---
# Module name is 'bitmap2svg_core' as per __init__.py
pybind11_add_module(bitmap2svg_core
    bitmap2svg/cpp/bindings.cpp
    bitmap2svg/cpp/bitmap_to_svg.cpp
)

# --- Include Directories for the Module ---
target_include_directories(bitmap2svg_core PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmap2svg/cpp" # For local headers like bitmap_to_svg.h
)
if(WITH_FAISS_LIBRARY_FOUND AND FAISS_INCLUDE_DIRS_TO_USE)
    target_include_directories(bitmap2svg_core PUBLIC ${FAISS_INCLUDE_DIRS_TO_USE})
    message(STATUS "Adding FAISS include directory: ${FAISS_INCLUDE_DIRS_TO_USE}")
endif()
# pybind11_INCLUDE_DIRS are automatically handled by pybind11_add_module() in modern pybind11.

# --- Compile Definitions for C++ ---
if(WITH_CUDA_ENABLED) # This refers to general CUDA toolkit availability for OpenCV/Faiss
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_CUDA)
endif()
if(WITH_FAISS_GPU_ENABLED) # This means Faiss lib is found, it's GPU, and CUDA is available
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_FAISS_GPU)
endif()

# --- Link Libraries to the Module ---
target_link_libraries(bitmap2svg_core PRIVATE
    ${OpenCV_LIBS}      # OpenCV libraries (includes CUDA modules if found by OpenCV)
    pybind11::module    # Pybind11 CMake target
    # pybind11::embed   # Only if embedding Python
)

if(WITH_FAISS_LIBRARY_FOUND AND FAISS_LIBRARIES_TO_LINK)
    target_link_libraries(bitmap2svg_core PRIVATE ${FAISS_LIBRARIES_TO_LINK})
    message(STATUS "Linking with FAISS library/target: ${FAISS_LIBRARIES_TO_LINK}")
endif()

if(WITH_CUDA_ENABLED) # If general CUDA is active, link CUDA runtime
    # Modern CMake provides these as IMPORTED targets like CUDA::cudart
    target_link_libraries(bitmap2svg_core PRIVATE CUDA::cudart)
endif()

# --- Installation ---
# scikit-build-core or a setup.py using CMakeBuild (as previously discussed)
# will typically handle placing the module in the correct location within the Python package.
# This install rule is more for traditional CMake `make install`.
# Ensure the DESTINATION aligns with Python package structure.
install(TARGETS bitmap2svg_core LIBRARY DESTINATION bitmap2svg) 
# For a flat module structure (e.g. if __init__.py is at project root and imports 'bitmap2svg_core'):
# install(TARGETS bitmap2svg_core LIBRARY DESTINATION .)


# --- Final Configuration Summary ---
message(STATUS "--- Configuration Summary ---")
message(STATUS "Preferring Faiss from Python env (MYPROJECT_PREFER_PYTHON_FAISS): ${MYPROJECT_PREFER_PYTHON_FAISS}")
if(MYPROJECT_PREFER_PYTHON_FAISS)
    message(STATUS "  Python Faiss Include Dir: ${PYTHON_FAISS_INCLUDE_DIR}")
    message(STATUS "  Python Faiss Library Path: ${PYTHON_FAISS_LIBRARY_PATH}")
    message(STATUS "  Python Faiss Reported as GPU: ${PYTHON_FAISS_IS_GPU_ENABLED}")
endif()
message(STATUS "System Dev Search for CUDA (DEVSYS_USE_CUDA): ${DEVSYS_USE_CUDA}")
message(STATUS "System Dev Search for Faiss GPU (DEVSYS_USE_FAISS_GPU): ${DEVSYS_USE_FAISS_GPU}")
message(STATUS "---------------------------")
message(STATUS "Effective CUDA Toolkit Enabled (WITH_CUDA_ENABLED): ${WITH_CUDA_ENABLED}")
message(STATUS "Effective Faiss Library Found (WITH_FAISS_LIBRARY_FOUND): ${WITH_FAISS_LIBRARY_FOUND}")
if(WITH_FAISS_LIBRARY_FOUND)
    message(STATUS "  Effective Faiss GPU Enabled (WITH_FAISS_GPU_ENABLED): ${WITH_FAISS_GPU_ENABLED}")
    message(STATUS "  Using FAISS Includes: ${FAISS_INCLUDE_DIRS_TO_USE}")
    message(STATUS "  Linking FAISS Libs: ${FAISS_LIBRARIES_TO_LINK}")
endif()
message(STATUS "OpenCV CUDA modules effectively used: ${OpenCV_cudaarithm_FOUND} (indicator, implies others if OpenCV built with them)")
message(STATUS "---------------------------")
