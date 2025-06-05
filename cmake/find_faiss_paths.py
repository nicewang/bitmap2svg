import os
import sys
import importlib.util

def get_faiss_paths_and_print():
    """
    Detects Faiss include directory, library path, and GPU status,
    then prints them as KEY=VALUE pairs for CMake to parse.
    Output is semicolon-separated KEY=VALUE pairs.
    """
    results = []
    try:
        # Ensure faiss is importable (it should be due to build-system.requires)
        import faiss
        # print(f"DEBUG: Successfully imported faiss. Version: {faiss.__version__}", file=sys.stderr) # For debugging

        # 1. Get Faiss include directory
        faiss_include_dir = faiss.get_include()
        if faiss_include_dir and os.path.exists(faiss_include_dir):
            results.append(f"PYTHON_FAISS_INCLUDE_DIR={faiss_include_dir}")
        else:
            print(f"DEBUG: Faiss include dir not found or invalid: {faiss_include_dir}", file=sys.stderr)


        # 2. Get Faiss library path
        faiss_module_dir = os.path.dirname(faiss.__file__)
        faiss_lib_path_candidate = None

        # Try strategy 1: swigfaiss_avx2.so or swigfaiss.so (often the linkable .so itself)
        try:
            import faiss.swigfaiss_avx2
            faiss_lib_path_candidate = faiss.swigfaiss_avx2.__file__
        except ImportError:
            try:
                import faiss.swigfaiss
                faiss_lib_path_candidate = faiss.swigfaiss.__file__
            except ImportError:
                pass # Will try libfaiss.so next

        # Try strategy 2: libfaiss.so in common locations within the package
        if not (faiss_lib_path_candidate and os.path.exists(faiss_lib_path_candidate)):
            search_paths = [
                faiss_module_dir,
                os.path.join(faiss_module_dir, '.libs') # For manylinux wheels
            ]
            for search_dir in search_paths:
                potential_path = os.path.join(search_dir, "libfaiss.so")
                if os.path.exists(potential_path):
                    faiss_lib_path_candidate = potential_path
                    break
        
        if faiss_lib_path_candidate and os.path.exists(faiss_lib_path_candidate):
            results.append(f"PYTHON_FAISS_LIBRARY_PATH={os.path.abspath(faiss_lib_path_candidate)}")
        else:
            print(f"DEBUG: Faiss library path not found. Last candidate: {faiss_lib_path_candidate}", file=sys.stderr)


        # 3. Determine if Faiss is GPU enabled
        is_gpu_enabled = False
        try:
            compile_options = faiss.get_compile_options()
            if "with GPU" in compile_options:
                is_gpu_enabled = True
        except AttributeError: # Fallback if get_compile_options not present
            if hasattr(faiss, 'StandardGpuResources'):
                 is_gpu_enabled = True # Strong hint
        except Exception as e:
            print(f"DEBUG: Error checking Faiss GPU compile_options/StandardGpuResources: {e}", file=sys.stderr)
        
        # Another fallback based on version string if primary checks are inconclusive
        if not is_gpu_enabled and 'gpu' in faiss.__version__.lower():
            is_gpu_enabled = True
            # print(f"DEBUG: GPU enabled due to 'gpu' in version string: {faiss.__version__}", file=sys.stderr)


        results.append(f"PYTHON_FAISS_IS_GPU_ENABLED={'ON' if is_gpu_enabled else 'OFF'}")

    except ImportError:
        print("ERROR: Python module 'faiss' not found. Faiss paths cannot be determined.", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in find_faiss_paths.py: {e}", file=sys.stderr)

    # Print as a single semicolon-separated string for CMake
    print(";".join(results))

if __name__ == "__main__":
    get_faiss_paths_and_print()
