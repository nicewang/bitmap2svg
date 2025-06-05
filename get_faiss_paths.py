import faiss
import os
import sys

def get_faiss_paths():
    faiss_include_dir = ""
    faiss_library_path = ""
    is_gpu_enabled = "OFF"

    try:
        faiss_include_dir = os.path.join(os.path.dirname(faiss.__file__), 'include')
        
        for root, dirs, files in os.walk(os.path.dirname(faiss.__file__)):
            for file in files:
                if file.startswith("libfaiss_gpu") and (file.endswith(".so") or file.endswith(".dylib") or file.endswith(".dll")):
                    faiss_library_path = os.path.join(root, file)
                    is_gpu_enabled = "ON" 
                    break
                elif file.startswith("libfaiss") and (file.endswith(".so") or file.endswith(".dylib") or file.endswith(".dll")):
                    # If we find a non-GPU library, we still want to capture its pat
                    faiss_library_path = os.path.join(root, file)
            if faiss_library_path:
                break

        # Check if GPU is enabled by looking for 'gpu' in the faiss module path
        if "gpu" in faiss.__file__.lower():
            is_gpu_enabled = "ON"

    except ImportError:
        # Faiss not found, return empty paths
        pass
    except Exception as e:
        sys.stderr.write(f"Error getting Faiss paths: {e}\n")
        pass

    # CMake expects arguments like -DPYTHON_VAR=VALUE
    # So we'll print them in that format
    cmake_args = []
    if faiss_include_dir:
        cmake_args.append(f"-DPYTHON_FAISS_INCLUDE_DIR={faiss_include_dir}")
    if faiss_library_path:
        cmake_args.append(f"-DPYTHON_FAISS_LIBRARY_PATH={faiss_library_path}")
    cmake_args.append(f"-DPYTHON_FAISS_IS_GPU_ENABLED={is_gpu_enabled}")

    # Print a single string that contains all arguments, space-separated
    print(" ".join(cmake_args))

if __name__ == "__main__":
    get_faiss_paths()
