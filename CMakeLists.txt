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

# --- Variables from Python build environment (now potentially set by probe script) ---
# These initial CACHE definitions act as defaults or placeholders
set(PYTHON_FAISS_INCLUDE_DIR "" CACHE PATH "Path to Faiss include directory from Python package (auto-detected or manual)")
set(PYTHON_FAISS_LIBRARY_PATH "" CACHE FILEPATH "Path to Faiss library file from Python package (auto-detected or manual)")
set(PYTHON_FAISS_IS_GPU_ENABLED OFF CACHE BOOL "Whether the Python Faiss package is GPU enabled (auto-detected or manual)")
option(MYPROJECT_PREFER_PYTHON_FAISS "Prefer Faiss from Python environment if paths are provided" ON)

# --- Attempt to auto-detect Faiss paths using a Python script ---
# This section will try to overwrite the above CACHE variables if the script succeeds.
if(MYPROJECT_PREFER_PYTHON_FAISS) # Only run probe if we prefer Python Faiss
    set(FAISS_PROBE_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_faiss_paths.py")
    if(EXISTS "${FAISS_PROBE_SCRIPT}")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        if(Python3_EXECUTABLE)
            message(STATUS "Attempting to auto-detect Faiss paths using Python script: ${FAISS_PROBE_SCRIPT}")
            execute_process(
                COMMAND "${Python3_EXECUTABLE}" "${FAISS_PROBE_SCRIPT}"
                RESULT_VARIABLE _faiss_probe_result
                OUTPUT_VARIABLE _faiss_probe_output # KEY=VALUE;KEY=VALUE
                ERROR_VARIABLE _faiss_probe_error
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )

            if(_faiss_probe_result EQUAL 0 AND _faiss_probe_output)
                message(STATUS "Faiss probe script successful. Output: ${_faiss_probe_output}")
                # Split the output string by semicolon into a list of KEY=VALUE pairs
                string(REPLACE ";" _list_separator "${_faiss_probe_output}" _key_value_list)

                foreach(_key_value_pair ${_key_value_list})
                    if(_key_value_pair MATCHES "([^=]+)=(.+)")
                        set(_key "${CMAKE_MATCH_1}")
                        set(_value "${CMAKE_MATCH_2}")
                        message(STATUS "  Probe found: ${_key} = ${_value}")
                        # Determine CACHE type based on key name (optional, could just use STRING)
                        if(_key STREQUAL "PYTHON_FAISS_INCLUDE_DIR")
                            set(${_key} "${_value}" CACHE PATH "Path from Faiss probe script" FORCE)
                        elseif(_key STREQUAL "PYTHON_FAISS_LIBRARY_PATH")
                            set(${_key} "${_value}" CACHE FILEPATH "Path from Faiss probe script" FORCE)
                        elseif(_key STREQUAL "PYTHON_FAISS_IS_GPU_ENABLED")
                            # CMake set(CACHE BOOL ...) expects ON/OFF or TRUE/FALSE
                            if("${_value}" STREQUAL "ON" OR "${_value}" STREQUAL "OFF" OR
                               "${_value}" STREQUAL "TRUE" OR "${_value}" STREQUAL "FALSE" OR
                               "${_value}" STREQUAL "1" OR "${_value}" STREQUAL "0")
                                set(${_key} "${_value}" CACHE BOOL "Status from Faiss probe script" FORCE)
                            else()
                                message(WARNING "  Faiss probe script returned invalid BOOL for ${_key}: ${_value}. Using OFF.")
                                set(${_key} OFF CACHE BOOL "Status from Faiss probe script (fallback)" FORCE)
                            endif()
                        else()
                            # For any other unexpected keys, just set as string
                            set(${_key} "${_value}" CACHE STRING "Value from Faiss probe script" FORCE)
                        endif()
                    else()
                        message(WARNING "  Faiss probe script returned malformed pair: ${_key_value_pair}")
                    endif()
                endforeach()
                if(_faiss_probe_error) # Print stderr from script if any
                    message(WARNING "Faiss probe script stderr:\n${_faiss_probe_error}")
                endif()
            else()
                message(WARNING "Faiss probe script execution failed or produced no output.")
                if(NOT _faiss_probe_result EQUAL 0)
                    message(WARNING "  Result: ${_faiss_probe_result}")
                endif()
                if(NOT _faiss_probe_output)
                     message(WARNING "  Output: <empty>")
                endif()
                if(_faiss_probe_error)
                    message(WARNING "  Error output:\n${_faiss_probe_error}")
                endif()
                message(WARNING "Proceeding with default/manual Faiss path settings if any.")
            endif()
        else()
            message(WARNING "Python3 interpreter not found. Cannot run Faiss probe script.")
        endif()
    else()
        message(STATUS "Faiss probe script not found at ${FAISS_PROBE_SCRIPT}. Skipping auto-detection.")
    endif()
endif()

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
    # If Python Faiss is GPU enabled, we need CUDA Toolkit for it.
    set(CUDA_REQUESTED TRUE)
    message(STATUS "Python Faiss is GPU enabled, requesting CUDA Toolkit.")
endif()
# DEVSYS_USE_CUDA is now OFF by default for pip build, only ON for local dev.
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
    message(STATUS "CUDA support is not requested for C++ compilation (only if Python Faiss GPU is used or DEVSYS_USE_CUDA is ON).")
endif()


# --- Faiss Detection (prioritizing Python environment) ---
set(FAISS_FOUND_FROM_PYTHON FALSE)
if(MYPROJECT_PREFER_PYTHON_FAISS AND PYTHON_FAISS_INCLUDE_DIR AND PYTHON_FAISS_LIBRARY_PATH)
    message(STATUS "Attempting to use Faiss from Python environment.")
    if(EXISTS "${PYTHON_FAISS_INCLUDE_DIR}" AND EXISTS "${PYTHON_FAISS_LIBRARY_PATH}")
        set(FAISS_INCLUDE_DIRS_TO_USE "${PYTHON_FAISS_INCLUDE_DIR}")
        set(FAISS_LIBRARIES_TO_LINK "${PYTHON_FAISS_LIBRARY_PATH}")
        set(WITH_FAISS_LIBRARY_FOUND TRUE)
        set(FAISS_FOUND_FROM_PYTHON TRUE) # Mark that it was found from Python
        message(STATUS "Found Faiss from Python environment:")
        message(STATUS "  Include Dir: ${PYTHON_FAISS_INCLUDE_DIR}")
        message(STATUS "  Library Path: ${PYTHON_FAISS_LIBRARY_PATH}")
        if(PYTHON_FAISS_IS_GPU_ENABLED)
            if(WITH_CUDA_ENABLED) # Crucial check! Ensure CUDA Toolkit was actually found
                set(WITH_FAISS_GPU_ENABLED TRUE)
                message(STATUS "  Python Faiss is GPU enabled and CUDA is available.")
            else()
                message(WARNING "Python Faiss is GPU enabled, but CUDA is NOT available/configured by CMake. Disabling FAISS GPU features for C++ compilation.")
            endif()
        else()
            message(STATUS "  Python Faiss is CPU only.")
        endif()
    else()
        message(WARNING "Python Faiss paths provided but invalid/not found. Will attempt system search if enabled.")
    endif()
endif()

# Only try system Faiss if Python Faiss was not preferred or not found
if(NOT WITH_FAISS_LIBRARY_FOUND AND DEVSYS_USE_FAISS_GPU)
    if(WITH_CUDA_ENABLED)
        message(STATUS "Attempting to find system Faiss with GPU support.")
        # Try to find Faiss GPU specifically
        find_package(Faiss QUIET COMPONENTS GPU)
        if(Faiss_FOUND)
            set(FAISS_INCLUDE_DIRS_TO_USE ${Faiss_INCLUDE_DIRS})
            set(FAISS_LIBRARIES_TO_LINK ${FAISS_SYSTEM_LINK_TARGET}) # Use the provided target
            set(WITH_FAISS_LIBRARY_FOUND TRUE)
            set(WITH_FAISS_GPU_ENABLED TRUE)
            message(STATUS "System Faiss found and configured for GPU.")
        else()
            message(WARNING "DEVSYS_USE_FAISS_GPU is ON but system FAISS library not found for GPU.")
        endif()
    else()
        message(WARNING "DEVSYS_USE_FAISS_GPU is ON but CUDA is not available. Cannot use system FAISS GPU.")
    endif()
endif()


# --- OpenCV ---
set(OpenCV_REQUIRED_COMPONENTS core imgproc)
# if(WITH_CUDA_ENABLED)
#     list(APPEND OpenCV_REQUIRED_COMPONENTS cudaarithm cudaimgproc cudev)
# endif()
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

# *** Explicitly add CUDA include directories to the target ***
# This is a key addition to try and resolve the missing cuda_fp16.h / cuda_bf16.h
if(WITH_CUDA_ENABLED AND CUDA_INCLUDE_DIRS)
    message(STATUS "Explicitly adding CUDA_INCLUDE_DIRS to target bitmap2svg_core: ${CUDA_INCLUDE_DIRS}")
    # Use SYSTEM PUBLIC to ensure it's propagated and treated as system headers
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
endif()

if(WITH_CUDA_ENABLED) # If general CUDA is active and configured
    # Try linking to the modern CMake target first
    find_package(CUDAToolkit QUIET COMPONENTS Runtime) # CUDAToolkit module is newer and might create these targets more reliably
    if(CUDAToolkit_FOUND AND TARGET CUDA::cudart)
        target_link_libraries(bitmap2svg_core PRIVATE CUDA::cudart)
        message(STATUS "Linking target with CUDA::cudart (modern CMake target).")
    else()
        # Fallback: Find cudart library directly and link
        find_library(CUDART_LIBRARY cudart PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib NO_DEFAULT_PATH)
        if(CUDART_LIBRARY)
            target_link_libraries(bitmap2svg_core PRIVATE ${CUDART_LIBRARY})
            message(STATUS "Linking target with cudart library found at: ${CUDART_LIBRARY} (fallback).")
        else()
            message(WARNING "CUDA is enabled, but CUDA::cudart target was not found, and direct cudart library could not be found. CUDA features might fail.")
            # If C++ code depends heavily on CUDA runtime, you might want to set WITH_CUDA_ENABLED to FALSE here,
            # or ensure this error stops the build if it's critical.
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
