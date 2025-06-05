import faiss
import os
import sys
import tomli
import importlib.util

def _get_faiss_cmake_args_internal():
    """
    Dynamically finds Faiss installation paths and returns them as a list of CMake arguments.
    This function will be called by the custom build backend.
    """
    faiss_include_dir = ""
    faiss_library_path = ""
    is_gpu_enabled = "OFF"
    cmake_args = []

    print("DEBUG (Custom Backend): Executing _get_faiss_cmake_args_internal()", file=sys.stderr)

    try:
        # Get Faiss include directory
        faiss_include_dir = os.path.join(os.path.dirname(faiss.__file__), 'include')
        print(f"DEBUG (Custom Backend): Faiss discovered include dir: {faiss_include_dir}", file=sys.stderr)

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

        print(f"DEBUG (Custom Backend): Faiss discovered library path: {faiss_library_path}", file=sys.stderr)
        print(f"DEBUG (Custom Backend): Faiss GPU enabled (detected): {is_gpu_enabled}", file=sys.stderr)

    except ImportError:
        print("WARNING (Custom Backend): Faiss module not found in the build environment. Faiss-dependent features will be disabled.", file=sys.stderr)
    except Exception as e:
        print(f"WARNING (Custom Backend): Error getting Faiss paths: {e}", file=sys.stderr)

    if faiss_include_dir:
        cmake_args.append(f"-DPYTHON_FAISS_INCLUDE_DIR={faiss_include_dir}")
    if faiss_library_path:
        cmake_args.append(f"-DPYTHON_FAISS_LIBRARY_PATH={faiss_library_path}")
    cmake_args.append(f"-DPYTHON_FAISS_IS_GPU_ENABLED={is_gpu_enabled}")

    print(f"DEBUG (Custom Backend): Generated CMake args: {cmake_args}", file=sys.stderr)
    return cmake_args

def get_requires_for_build_wheel(config_settings=None):
    """
    Called by pip to get build requirements.
    We'll add our dynamic CMake args here.
    """
    print("DEBUG (Custom Backend): get_requires_for_build_wheel called.", file=sys.stderr)

    try:
        spec = importlib.util.find_spec("scikit_build_core.build")
        if spec is None:
            raise ImportError("scikit_build_core.build not found. Please ensure it's in build-system.requires.")
        scikit_build_core_backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = scikit_build_core_backend
        spec.loader.exec_module(scikit_build_core_backend)
        _get_requires_for_build_wheel_orig = scikit_build_core_backend.get_requires_for_build_wheel
    except ImportError as e:
        print(f"ERROR (Custom Backend): Could not import scikit_build_core.build: {e}", file=sys.stderr)
        raise

    # Augment config_settings with our dynamic CMake args
    if config_settings is None:
        config_settings = {}
    
    # Get dynamic Faiss CMake arguments
    dynamic_cmake_args = _get_faiss_cmake_args_internal()

    # The key for extra CMake arguments is 'cmake.args'
    # scikit-build-core expects this in config_settings
    if 'cmake.args' in config_settings:
        if isinstance(config_settings['cmake.args'], list):
            config_settings['cmake.args'].extend(dynamic_cmake_args)
        else:
            # If it's not a list, it's a single string (unlikely but handle)
            config_settings['cmake.args'] = [config_settings['cmake.args']] + dynamic_cmake_args
    else:
        config_settings['cmake.args'] = dynamic_cmake_args

    print(f"DEBUG (Custom Backend): Final config_settings for scikit-build-core: {config_settings}", file=sys.stderr)

    # Call the original scikit-build-core's get_requires_for_build_wheel
    return _get_requires_for_build_wheel_orig(config_settings)

def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """
    Called by pip to prepare metadata.
    """
    print("DEBUG (Custom Backend): prepare_metadata_for_build_wheel called.", file=sys.stderr)
    try:
        spec = importlib.util.find_spec("scikit_build_core.build")
        scikit_build_core_backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = scikit_build_core_backend
        spec.loader.exec_module(scikit_build_core_backend)
        _prepare_metadata_for_build_wheel_orig = scikit_build_core_backend.prepare_metadata_for_build_wheel
    except ImportError as e:
        print(f"ERROR (Custom Backend): Could not import scikit_build_core.build: {e}", file=sys.stderr)
        raise

    return _prepare_metadata_for_build_wheel_orig(metadata_directory, config_settings)

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """
    Called by pip to build the wheel.
    """
    print("DEBUG (Custom Backend): build_wheel called.", file=sys.stderr)
    try:
        spec = importlib.util.find_spec("scikit_build_core.build")
        scikit_build_core_backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = scikit_build_core_backend
        spec.loader.exec_module(scikit_build_core_backend)
        _build_wheel_orig = scikit_build_core_backend.build_wheel
    except ImportError as e:
        print(f"ERROR (Custom Backend): Could not import scikit_build_core.build: {e}", file=sys.stderr)
        raise

    if config_settings is None:
        config_settings = {}
    
    dynamic_cmake_args = _get_faiss_cmake_args_internal()
    if 'cmake.args' in config_settings:
        if isinstance(config_settings['cmake.args'], list):
            config_settings['cmake.args'].extend(dynamic_cmake_args)
        else:
            config_settings['cmake.args'] = [config_settings['cmake.args']] + dynamic_cmake_args
    else:
        config_settings['cmake.args'] = dynamic_cmake_args

    return _build_wheel_orig(wheel_directory, config_settings, metadata_directory)

def build_sdist(sdist_directory, config_settings=None):
    """
    Called by pip to build the source distribution.
    """
    print("DEBUG (Custom Backend): build_sdist called.", file=sys.stderr)
    try:
        spec = importlib.util.find_spec("scikit_build_core.build")
        scikit_build_core_backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = scikit_build_core_backend
        spec.loader.exec_module(scikit_build_core_backend)
        _build_sdist_orig = scikit_build_core_backend.build_sdist
    except ImportError as e:
        print(f"ERROR (Custom Backend): Could not import scikit_build_core.build: {e}", file=sys.stderr)
        raise
    
    # Similar to build_wheel, re-inject dynamic args for sdist build
    if config_settings is None:
        config_settings = {}
    
    dynamic_cmake_args = _get_faiss_cmake_args_internal()
    if 'cmake.args' in config_settings:
        if isinstance(config_settings['cmake.args'], list):
            config_settings['cmake.args'].extend(dynamic_cmake_args)
        else:
            config_settings['cmake.args'] = [config_settings['cmake.args']] + dynamic_cmake_args
    else:
        config_settings['cmake.args'] = dynamic_cmake_args

    return _build_sdist_orig(sdist_directory, config_settings)
