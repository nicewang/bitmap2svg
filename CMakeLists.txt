cmake_minimum_required(VERSION 3.18)
project(bitmap2svg_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Options for GPU Acceleration ---
option(USE_CUDA "Enable CUDA support for OpenCV and FAISS" ON)
option(USE_FAISS_GPU "Enable FAISS GPU for K-Means (requires USE_CUDA and FAISS library)" ON)

set(WITH_CUDA_ENABLED FALSE)
set(WITH_FAISS_GPU_ENABLED FALSE)

# --- CUDA Toolkit Detection ---
if(USE_CUDA)
    # Find the CUDA toolkit.
    # Forcing specific architectures can be done with CMAKE_CUDA_ARCHITECTURES, e.g., "75;86" for T4;P100
    # Example: set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86") # From P100, T4, up to Ampere
    find_package(CUDA QUIET) # Use QUIET to handle gracefully if not found
    if(CUDA_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "CUDA version: ${CUDA_VERSION_STRING}")
        enable_language(CUDA) # Enable CUDA language support
        set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(WITH_CUDA_ENABLED TRUE)
    else()
        message(WARNING "USE_CUDA is ON but CUDA Toolkit not found. Disabling CUDA features.")
        set(USE_CUDA OFF) # Effectively disable if not found
    endif()
else()
    message(STATUS "CUDA support is disabled by user (USE_CUDA=OFF).")
endif()

if(NOT USE_CUDA AND USE_FAISS_GPU)
    message(WARNING "USE_FAISS_GPU is ON but USE_CUDA is OFF. FAISS GPU requires CUDA. Disabling FAISS GPU.")
    set(USE_FAISS_GPU OFF)
endif()

# --- OpenCV ---
set(OpenCV_REQUIRED_COMPONENTS core imgproc)
if(WITH_CUDA_ENABLED)
    # Add OpenCV CUDA modules if CUDA is enabled
    list(APPEND OpenCV_REQUIRED_COMPONENTS cudaarithm cudaimgproc cudev)
    # You might need other modules like opencv_cudafilters, opencv_cudawarping depending on usage
    # list(APPEND OpenCV_REQUIRED_COMPONENTS cudafilters)
endif()

find_package(OpenCV REQUIRED COMPONENTS ${OpenCV_REQUIRED_COMPONENTS})
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV (with required components) not found!")
endif()
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
if(WITH_CUDA_ENABLED AND NOT OpenCV_cudaarithm_FOUND) # Check if a core CUDA module was actually found
    message(WARNING "OpenCV found, but required CUDA components (e.g., cudaarithm) seem to be missing in your OpenCV build. Disabling CUDA-specific OpenCV paths.")
    set(WITH_CUDA_ENABLED FALSE) # Downgrade if OpenCV CUDA modules are not present
    if(USE_FAISS_GPU)
         message(WARNING "FAISS GPU might still work if it does not rely on OpenCV CUDA GpuMat conversions directly in your code, but OpenCV operations won't be CUDA accelerated.")
    endif()
endif()


# --- FAISS ---
if(USE_FAISS_GPU AND WITH_CUDA_ENABLED) # FAISS GPU requires CUDA
    # Users might need to set FAISS_ROOT or CMAKE_PREFIX_PATH if FAISS is not in standard locations.
    # Example: set(FAISS_ROOT "/opt/faiss" CACHE PATH "Path to FAISS installation")
    find_package(FAISS QUIET) # Try to find FAISS
    if(FAISS_FOUND)
        # FAISS exports different targets depending on its build configuration (e.g., faiss::faiss_gpu, faiss::faiss, faiss::faiss_avx2).
        # We need to link against a target that includes GPU support.
        # This logic tries to find a suitable target.
        set(FAISS_LINK_TARGET "")
        if(TARGET faiss::faiss_gpu)
            set(FAISS_LINK_TARGET faiss::faiss_gpu)
        elseif(TARGET faiss_gpu) # Some older FAISS might just export 'faiss_gpu'
             set(FAISS_LINK_TARGET faiss_gpu)
        elseif(TARGET faiss::faiss_avx2) # Often includes GPU if built with it
             set(FAISS_LINK_TARGET faiss::faiss_avx2)
        elseif(TARGET faiss::faiss) # Generic target, hope it includes GPU
             set(FAISS_LINK_TARGET faiss::faiss)
        endif()

        if(NOT FAISS_LINK_TARGET STREQUAL "")
            message(STATUS "FAISS found. Will link against '${FAISS_LINK_TARGET}'. Assumed to be built with GPU support.")
            set(WITH_FAISS_GPU_ENABLED TRUE)
        else()
            message(WARNING "FAISS found, but no recognizable GPU-enabled CMake target (e.g., faiss::faiss_gpu) was identified. Disabling FAISS GPU support.")
            set(USE_FAISS_GPU OFF)
        endif()
    else()
        message(WARNING "USE_FAISS_GPU is ON but FAISS library not found. Disabling FAISS GPU support.")
        set(USE_FAISS_GPU OFF)
    endif()
elif(USE_FAISS_GPU AND NOT WITH_CUDA_ENABLED)
    message(STATUS "FAISS GPU support disabled because CUDA is not available or not enabled.")
    set(USE_FAISS_GPU OFF)
endif()


# --- pybind11 ---
find_package(pybind11 REQUIRED)
message(STATUS "Pybind11 version: ${pybind11_VERSION}")
message(STATUS "Pybind11 include dirs: ${pybind11_INCLUDE_DIRS}") # For reference

# --- Create Python Module ---
# Module sources are relative to CMAKE_CURRENT_SOURCE_DIR
pybind11_add_module(bitmap2svg_core
    bitmap2svg/cpp/bindings.cpp
    bitmap2svg/cpp/bitmap_to_svg.cpp
)

# --- Include Directories for the Module ---
target_include_directories(bitmap2svg_core PUBLIC
    ${OpenCV_INCLUDE_DIRS}      # OpenCV headers
    # pybind11_INCLUDE_DIRS are typically handled by pybind11_add_module()
    # If FAISS headers are not globally installed, you might need to add ${FAISS_INCLUDE_DIRS}
    # if(FAISS_FOUND AND WITH_FAISS_GPU_ENABLED)
    #    target_include_directories(bitmap2svg_core PUBLIC ${FAISS_INCLUDE_DIRS}) # FAISS_INCLUDE_DIRS must be set by find_package(FAISS)
    # endif()
    "${CMAKE_CURRENT_SOURCE_DIR}/bitmap2svg/cpp" # For local headers like bitmap_to_svg.h
)

# --- Compile Definitions for C++ ---
if(WITH_CUDA_ENABLED)
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_CUDA)
    message(STATUS "Compiling with WITH_CUDA definition for OpenCV CUDA paths.")
endif()
if(WITH_FAISS_GPU_ENABLED)
    target_compile_definitions(bitmap2svg_core PRIVATE WITH_FAISS_GPU)
    message(STATUS "Compiling with WITH_FAISS_GPU definition for FAISS GPU K-Means.")
else()
     message(STATUS "FAISS GPU K-Means specific code paths will be disabled.")
endif()

# --- Link Libraries to the Module ---
target_link_libraries(bitmap2svg_core PRIVATE
    ${OpenCV_LIBS}      # OpenCV libraries (includes CUDA modules if found)
    pybind11::module    # Pybind11 library
)

if(WITH_FAISS_GPU_ENABLED AND NOT FAISS_LINK_TARGET STREQUAL "")
    target_link_libraries(bitmap2svg_core PRIVATE ${FAISS_LINK_TARGET})
    message(STATUS "Linking with FAISS target: ${FAISS_LINK_TARGET}")
endif()

if(WITH_CUDA_ENABLED)
    # Explicitly link CUDA runtime and other necessary CUDA libraries
    # Modern CMake usually provides these asIMPORTED targets like CUDA::cudart
    target_link_libraries(bitmap2svg_core PRIVATE CUDA::cudart)
    # If using cuBLAS or cuFFT directly or if FAISS needs them explicitly:
    # target_link_libraries(bitmap2svg_core PRIVATE CUDA::cublas CUDA::cufft)
    message(STATUS "Linking with CUDA Runtime Library (CUDA::cudart).")
endif()

# --- Installation ---
# This will install the compiled Python module into a 'bitmap2svg' subdirectory
# of the CMAKE_INSTALL_PREFIX.
# For example, if CMAKE_INSTALL_PREFIX is /usr/local, it installs to /usr/local/bitmap2svg/
# If installing into a Python environment's site-packages, adjust DESTINATION accordingly or use setup.py.
install(TARGETS bitmap2svg_core DESTINATION bitmap2svg)

message(STATUS "--- Configuration Summary ---")
message(STATUS "USE_CUDA: ${USE_CUDA} (User option)")
message(STATUS "CUDA Toolkit Found & Enabled (WITH_CUDA_ENABLED): ${WITH_CUDA_ENABLED}")
if(WITH_CUDA_ENABLED)
    message(STATUS "OpenCV CUDA modules enabled: YES (if OpenCV was built with them and they were found)")
else()
    message(STATUS "OpenCV CUDA modules enabled: NO")
endif()
message(STATUS "USE_FAISS_GPU: ${USE_FAISS_GPU} (User option)")
message(STATUS "FAISS Library Found & GPU Enabled (WITH_FAISS_GPU_ENABLED): ${WITH_FAISS_GPU_ENABLED}")
message(STATUS "---------------------------")
