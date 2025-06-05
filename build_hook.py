import faiss
import os
import sys

def get_cmake_args():
    """
    Dynamically finds Faiss installation paths in the build environment
    and returns them as a list of CMake arguments.
    """
    faiss_include_dir = ""
    faiss_library_path = ""
    is_gpu_enabled = "OFF"
    cmake_args = []

    # Print debug info to stderr so it shows up in pip --verbose output
    print("DEBUG: Executing custom build_hook.py:get_cmake_args()", file=sys.stderr)

    try:
        # Get Faiss include directory
        faiss_include_dir = os.path.join(os.path.dirname(faiss.__file__), 'include')
        print(f"DEBUG: Faiss discovered include dir: {faiss_include_dir}", file=sys.stderr)

        # Get Faiss library file path
        found_lib = False
        for root, dirs, files in os.walk(os.path.dirname(faiss.__file__)):
            for file in files:
                # Prioritize GPU library names
                if file.startswith("libfaiss_gpu") and (file.endswith(".so") or file.endswith(".dylib") or file.endswith(".dll")):
                    faiss_library_path = os.path.join(root, file)
                    is_gpu_enabled = "ON"
                    found_lib = True
                    break
                # Fallback to CPU library name if no specific GPU library is found yet
                elif not found_lib and file.startswith("libfaiss") and (file.endswith(".so") or file.endswith(".dylib") or file.endswith(".dll")):
                    faiss_library_path = os.path.join(root, file)
            if found_lib:
                break

        # Confirm GPU enablement based on package name or library path
        if "gpu" in faiss.__file__.lower() or (faiss_library_path and "gpu" in faiss_library_path.lower()):
            is_gpu_enabled = "ON"
        else:
            is_gpu_enabled = "OFF"

        print(f"DEBUG: Faiss discovered library path: {faiss_library_path}", file=sys.stderr)
        print(f"DEBUG: Faiss GPU enabled (detected): {is_gpu_enabled}", file=sys.stderr)

    except ImportError:
        print("WARNING: Faiss module not found in the build environment. Faiss-dependent features will be disabled.", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: Error getting Faiss paths: {e}", file=sys.stderr)

    if faiss_include_dir:
        cmake_args.append(f"-DPYTHON_FAISS_INCLUDE_DIR={faiss_include_dir}")
    if faiss_library_path:
        cmake_args.append(f"-DPYTHON_FAISS_LIBRARY_PATH={faiss_library_path}")
    cmake_args.append(f"-DPYTHON_FAISS_IS_GPU_ENABLED={is_gpu_enabled}")

    print(f"DEBUG: Generated CMake args: {cmake_args}", file=sys.stderr)
    return cmake_args
