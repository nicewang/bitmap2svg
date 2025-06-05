import faiss
import os
import sys

def get_faiss_cmake_args():
    """
    Dynamically finds Faiss installation paths and returns them as a list of CMake arguments.
    This function will be called by scikit-build-core as a cmake.args hook.
    """
    faiss_include_dir = ""
    faiss_library_path = ""
    is_gpu_enabled = "OFF"
    cmake_args = []

    print("DEBUG (build_hook_for_faiss): Executing get_faiss_cmake_args()", file=sys.stderr)

    try:
        # Get Faiss include directory
        faiss_include_dir = os.path.join(os.path.dirname(faiss.__file__), 'include')
        print(f"DEBUG (build_hook_for_faiss): Faiss discovered include dir: {faiss_include_dir}", file=sys.stderr)

        # Get Faiss library file path
        found_lib = False
        for root, dirs, files in os.walk(os.path.dirname(faiss.__file__)):
            for file in files:
                if file.startswith("libfaiss_gpu") and (file.endswith(".so") or file.endswith(".dylib") or file.endswith(".dll")):
                    faiss_library_path = os.path.join(root, file)
                    is_gpu_enabled = "ON"
                    found_lib = True
                    break
                elif not found_lib and file.startswith("libfaiss") and (file.endswith(".so") or file.endswith(".dylib") or file.endswith(".dll")):
                    faiss_library_path = os.path.join(root, file)
            if found_lib:
                break

        if "gpu" in faiss.__file__.lower() or (faiss_library_path and "gpu" in faiss_library_path.lower()):
            is_gpu_enabled = "ON"
        else:
            is_gpu_enabled = "OFF"

        print(f"DEBUG (build_hook_for_faiss): Faiss discovered library path: {faiss_library_path}", file=sys.stderr)
        print(f"DEBUG (build_hook_for_faiss): Faiss GPU enabled (detected): {is_gpu_enabled}", file=sys.stderr)

    except ImportError:
        print("WARNING (build_hook_for_faiss): Faiss module not found in the build environment. Faiss-dependent features will be disabled.", file=sys.stderr)
    except Exception as e:
        print(f"WARNING (build_hook_for_faiss): Error getting Faiss paths: {e}", file=sys.stderr)

    if faiss_include_dir:
        cmake_args.append(f"-DPYTHON_FAISS_INCLUDE_DIR={faiss_include_dir}")
    if faiss_library_path:
        cmake_args.append(f"-DPYTHON_FAISS_LIBRARY_PATH={faiss_library_path}")
    cmake_args.append(f"-DPYTHON_FAISS_IS_GPU_ENABLED={is_gpu_enabled}")

    print(f"DEBUG (build_hook_for_faiss): Generated CMake args: {cmake_args}", file=sys.stderr)
    return cmake_args

# If this file is run directly (e.g., for debugging), print the arguments
if __name__ == "__main__":
    args = get_faiss_cmake_args()
    print(" ".join(args))
