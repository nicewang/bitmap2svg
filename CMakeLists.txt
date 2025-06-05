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
option(DEVSYS_USE_CUDA "Enable CUDA if found on system (for OpenCV, and Faiss if system Faiss is used)" OFF) # Set to OFF by default for Kaggle
option(DEVSYS_USE_FAISS_GPU "Enable FAISS GPU if system Faiss is found and supports GPU (requires DEVSYS_USE_CUDA)" OFF) # Set to OFF by default for Kaggle

# --- Variables from Python build environment ---
# These are typically set by the Python build backend (e.g., scikit-build-core)
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
    # If Python Faiss is GPU enabled, we need CUDA Toolkit for it.
    set(CUDA_REQUESTED TRUE)
    message(STATUS "Python Faiss is GPU enabled, requesting CUDA Toolkit.")
endif()
if(DEVSYS_USE_CUDA) # User explicitly wants system CUDA for OpenCV or system Faiss
    set(CUDA_REQUESTED TRUE)
    message(STATUS "DEVSYS_USE_CUDA is ON, requesting CUDA Toolkit.")
endif()

if(CUDA_REQUESTED)
    message(STATUS "Attempting to find CUDA Toolkit...")
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
        endif()
    endif()
else()
    message(STATUS "CUDA support is not requested for C++ compilation (only if Python Faiss GPU is NOT used and DEVSYS_USE_CUDA is OFF).")
endif()

---

### Faiss Detection (Prioritizing Python Environment)

```cmake
# --- Faiss Detection (prioritizing Python environment) ---
# Ensure it correctly sets WITH_FAISS_LIBRARY_FOUND, FAISS_INCLUDE_DIRS_TO_USE,
# FAISS_LIBRARIES_TO_LINK, and WITH_FAISS_GPU_ENABLED based on Python paths or system fallback)
# Make sure that if Python Faiss is GPU enabled, WITH_FAISS_GPU_ENABLED is only TRUE if WITH_CUDA_ENABLED is also TRUE.
if(MYPROJECT_PREFER_PYTHON_FAISS AND PYTHON_FAISS_INCLUDE_DIR AND PYTHON_FAISS_LIBRARY_PATH)
    message(STATUS "Attempting to use Faiss from Python environment.")
    if(EXISTS "<span class="math-inline">\{PYTHON\_FAISS\_INCLUDE\_DIR\}" AND EXISTS "</span>{PYTHON_FAISS_LIBRARY_PATH}")
        set(FAISS_INCLUDE_DIRS_TO_USE "<span class="math-inline">\{PYTHON\_FAISS\_INCLUDE\_DIR\}"\)
set\(FAISS\_LIBRARIES\_TO\_LINK "</span>{PYTHON_FAISS_LIBRARY_PATH}")
        set(WITH_FAISS_LIBRARY_FOUND TRUE)
        message(STATUS "Found Faiss from Python environment:")
        message(STATUS "  Include Dir: ${PYTHON_FAISS_INCLUDE_DIR}")
        message(STATUS "  Library Path: <span class="math-inline">\{PYTHON\_FAISS\_LIBRARY\_PATH\}"\)
if\(PYTHON\_FAISS\_IS\_GPU\_ENABLED\)
if\(WITH\_CUDA\_ENABLED\) \# Crucial check\! Ensure CUDA Toolkit was actually found
set\(WITH\_FAISS\_GPU\_ENABLED TRUE\)
message\(STATUS "  Python Faiss is GPU enabled and CUDA is available\. C\+\+ will use Faiss GPU\."\)
else\(\)
message\(WARNING "Python Faiss is GPU enabled, but CUDA is NOT available/configured by CMake\. Disabling FAISS GPU features for C\+\+ compilation\."\)
endif\(\)
else\(\)
message\(STATUS "  Python Faiss is CPU only\. C\+\+ will use Faiss CPU\."\)
endif\(\)
else\(\)
message\(WARNING "Python Faiss paths provided but invalid/not found \('</span>{PYTHON_FAISS_INCLUDE_DIR}' or '${PYTHON_FAISS_LIBRARY_PATH}' not found). Will attempt system search if enabled.")
    endif()
elseif(DEVSYS_USE_FAISS_GPU) # Fallback to system Faiss search
    message(STATUS "DEVSYS_USE_FAISS_GPU is ON. Attempting to find system Faiss.")
    if(WITH_CUDA_ENABLED)
        message(STATUS "CUDA is available, searching for GPU-enabled system Faiss.")
        # Find Faiss and explicitly request the CUDA component for GPU support
        find_package(Faiss QUIET COMPONENTS CUDA)
        if(FAISS_FOUND)
            set(FAISS_INCLUDE_DIRS_TO_USE ${Faiss_INCLUDE_DIRS})
            set(FAISS_LIBRARIES_TO_LINK ${FAISS_SYSTEM_LINK_TARGET}) # Or use specific Faiss targets if found
            set(WITH_FAISS_LIBRARY_FOUND TRUE)
            set(WITH_FAISS_GPU_ENABLED TRUE) # Assume system Faiss found via COMPONENTS CUDA is GPU enabled
            message(STATUS "System Faiss (GPU-enabled) found: ${FAISS_INCLUDE_DIRS_TO_USE}, ${FAISS_LIBRARIES_TO_LINK}")
        else()
            message(WARNING "DEVSYS_USE_FAISS_GPU is ON, but system FAISS library with GPU support not found.")
            message(WARNING "  Consider checking system Faiss installation or setting Faiss_DIR.")
        endif()
    else()
        message(WARNING "DEVSYS_USE_FAISS_GPU is ON but CUDA is not available. Cannot use system FAISS GPU.")
    endif()
else()
    message(STATUS "Faiss support (either Python or system) is not requested or not found.")
endif()

---

### OpenCV Integration

```cmake
# --- OpenCV ---
set(OpenCV_REQUIRED_COMPONENTS core imgproc)

# Re-enable request for OpenCV CUDA modules IF CUDA is enabled.
# IMPORTANT: This assumes your Kaggle environment provides an OpenCV C++ build with CUDA support.
# If these are not found, find_package will fail (because REQUIRED is used).
if(WITH_CUDA_ENABLED)
    message(STATUS "CUDA is enabled. Requesting OpenCV CUDA components: cudaarithm, cudaimgproc, cudev.")
    list(APPEND OpenCV_REQUIRED_COMPONENTS cudaarithm cudaimgproc cudev)
else()
    message(STATUS "CUDA is NOT enabled. Requesting only CPU OpenCV components: core, imgproc.")
endif()

# Try to find OpenCV with the specified components.
# If CUDA components are requested but not found, this will cause an error if they are REQUIRED.
# Using 'REQUIRED' here means the build will fail if specified components (including CUDA ones if requested) are not found.
find_package(OpenCV REQUIRED COMPONENTS ${OpenCV_REQUIRED_COMPONENTS})
message(STATUS "OpenCV found. Version: ${OpenCV_VERSION}")

# Check if CUDA components were successfully found, if requested
if(WITH_CUDA_ENABLED)
    if(NOT OpenCV_cudaarithm_FOUND OR NOT OpenCV_cudaimgproc_FOUND OR NOT OpenCV_cudev_FOUND)
        message(WARNING "OpenCV CUDA components (e.g., cudaarithm, cudaimgproc, cudev) were requested but NOT fully found in your OpenCV build.")
        message(WARNING "  This build will proceed with CPU-only OpenCV operations for image processing, even though CUDA is available.")
        # We don't set WITH_CUDA_ENABLED to FALSE here, as Faiss GPU might still be usable independently.
        # The C++ code should handle #ifdefs for OpenCV CUDA parts.
        # You might need to add a flag here to inform your C++ code that OpenCV CUDA is not fully available.
        # set(WITH_OPENCV_CUDA_ENABLED FALSE) # Example
    else()
        message(STATUS "OpenCV CUDA components found and configured.")
        # set(WITH_OPENCV_CUDA_ENABLED TRUE) # Example
    endif()
else()
    message(STATUS "OpenCV is configured for CPU-only (CUDA not requested/available).")
endif()

---

### pybind11 Integration

```cmake
# --- pybind11 ---
find_package(pybind11 REQUIRED CONFIG)
message(STATUS "Pybind11 found. Version: ${pybind11_VERSION}")

---

### Create Python Module

```cmake
# --- Create Python Module ---
pybind11_add_module(bitmap2svg_core
    bitmap2svg/cpp/bindings.cpp
    bitmap2svg/cpp/bitmap_to_svg.cpp
)

---

### Include Directories

```cmake
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

---

### Compile Definitions

```cmake
# --- Compile Definitions for C++ ---
if(WITH_CUDA_ENABLED)
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_CUDA)
    message(STATUS "Defining WITH_CUDA preprocessor macro.")
endif()
if(WITH_FAISS_GPU_ENABLED) # True if Faiss lib found, is GPU, and CUDA is available
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_FAISS_GPU)
    message(STATUS "Defining WITH_FAISS_GPU preprocessor macro.")
endif()

---

### Link Libraries

```cmake
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

---

### Installation

```cmake
# --- Installation ---
install(TARGETS bitmap2svg_core LIBRARY DESTINATION bitmap2svg)

---

### Final Configuration Summary

```cmake
# --- Final Configuration Summary ---
message(STATUS "--- Build Configuration Summary ---")
message(STATUS "CUDA Toolkit Enabled (WITH_CUDA_ENABLED): ${WITH_CUDA_ENABLED}")
if(WITH_CUDA_ENABLED)
    message(STATUS "  CUDA Version (approx): ${CUDA_VERSION}")
    message(STATUS "  CUDA Include Dirs (approx): ${CUDA_INCLUDE_DIRS}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Faiss Library Found (WITH_FAISS_LIBRARY_FOUND): ${WITH_FAISS_LIBRARY_FOUND}")
if(WITH_FAISS_LIBRARY_FOUND)
    message(STATUS "  Faiss GPU Enabled (WITH_FAISS_GPU_ENABLED): ${WITH_FAISS_GPU_ENABLED}")
    message(STATUS "  Faiss Include Dirs: ${FAISS_INCLUDE_DIRS_TO_USE}")
    message(STATUS "  Faiss Libraries: ${FAISS_LIBRARIES_TO_LINK}")
endif()
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Includes: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
# Add checks for specific OpenCV CUDA modules if you want to be very precise about their presence
if(WITH_CUDA_ENABLED)
    if(OpenCV_cudaarithm_FOUND AND OpenCV_cudaimgproc_FOUND AND OpenCV_cudev_FOUND)
        message(STATUS "  OpenCV CUDA modules (cudaarithm, cudaimgproc, cudev) are found.")
    else()
        message(STATUS "  OpenCV CUDA modules NOT fully found. OpenCV operations may be CPU-only.")
    endif()
endif()
message(STATUS "Pybind11 Version: ${pybind11_VERSION}")
message(STATUS "-----------------------------------")
