import os
import sys
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as SetuptoolsBuildExt

# --- Custom build command to integrate CMake ---
# Inherit from setuptools' standard build_ext command
class CMakeBuild(SetuptoolsBuildExt):
    def run(self):
        # Ensure CMake is available
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions.")

        # setuptools' temporary build directory
        # We'll create a subdirectory inside it for the CMake build
        cmake_build_dir = os.path.join(self.build_temp, 'cmake_build')
        os.makedirs(cmake_build_dir, exist_ok=True)

        # Get the path to the directory containing setup.py and CMakeLists.txt
        project_source_dir = os.path.abspath(os.path.dirname(__file__))

        # --- CMake Configuration ---
        cmake_command = ['cmake', project_source_dir] # Path to CMakeLists.txt

        # Set the output directory for libraries within the CMake build directory
        # We will copy from here to the final package directory later
        # This path is relative to the CMake *build* directory (cmake_build_dir)
        cmake_library_output_dir = os.path.join(cmake_build_dir, 'lib')
        os.makedirs(cmake_library_output_dir, exist_ok=True) # Ensure output dir exists for CMake

        cmake_args = [
            # Set the output directory for libraries
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(cmake_library_output_dir)}',
            # Pass Python executable path to CMake (useful for finding Python libraries if needed)
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            # Set the build type (Release for optimization, Debug for debugging)
            f'-DCMAKE_BUILD_TYPE={"Debug" if self.debug else "Release"}',
        ]

        # Add generator and architecture for Windows builds using Visual Studio
        if sys.platform == 'win32':
            # Use a suitable Visual Studio generator for the runner environment
            # 'Visual Studio 17 2022' or 'Visual Studio 16 2019' are common
            cmake_args.extend(['-G', 'Visual Studio 17 2022', '-A', 'x64']) # Use x64 architecture
            # Vcpkg integration is handled by CMakeLists.txt if VCPKG_ROOT is set in env

        # Add any extra arguments passed to setup.py build_ext
        cmake_args.extend(self.define or []) # setuptools passes -D options here

        print(f"Running CMake configure in {cmake_build_dir}")
        print(f"Command: {' '.join(cmake_command + cmake_args)}")
        try:
            subprocess.check_call(cmake_command + cmake_args, cwd=cmake_build_dir)
        except subprocess.CalledProcessError as e:
            print(f"CMake configuration failed: {e}")
            raise

        # --- CMake Build ---
        build_command = ['cmake', '--build', '.']
        build_args = []

        # Add parallel jobs option
        if sys.platform != 'win32':
             build_args.extend(['--', '-j4']) # Build with 4 parallel jobs (adjust as needed)
        else:
             # MSVC uses /m for parallel builds
             build_args.extend(['--', '/m'])

        # For multi-configuration generators (like VS), specify the build config
        # For single-config generators (like Makefiles), this might be ignored but harmless
        build_args.extend(['--config', 'Debug' if self.debug else 'Release'])

        print(f"Running CMake build in {cmake_build_dir}")
        print(f"Command: {' '.join(build_command + build_args)}")
        try:
            subprocess.check_call(build_command + build_args, cwd=cmake_build_dir)
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed: {e}")
            raise


        # --- Copy the built library to the final package directory ---
        # self.build_lib is the root directory where setuptools is building the package structure
        # (e.g., build/lib/). The library needs to go inside the actual package directory
        # within this structure (e.g., build/lib/bitmap2svg/)
        python_package_dir = os.path.join(self.build_lib, 'bitmap2svg')
        os.makedirs(python_package_dir, exist_ok=True) # Ensure the target package directory exists

        # Determine the expected library name based on the platform
        lib_patterns = {
            'linux': 'libbitmap_to_svg.so',
            'darwin': 'libbitmap_to_svg.dylib',
            'win32': 'bitmap_to_svg.dll'
        }
        lib_name = lib_patterns.get(sys.platform)
        if not lib_name:
             raise RuntimeError(f"Unsupported platform: {sys.platform}")

        # --- Robust Search for the Built Library ---
        # CMake might place the built library directly in CMAKE_LIBRARY_OUTPUT_DIRECTORY,
        # or in a subdirectory like 'Release/' or 'Debug/', especially with multi-config generators.
        # We search in likely locations.
        potential_library_paths_relative_to_output = [
            lib_name,                 # Directly in the output directory
            os.path.join('Release', lib_name), # In Release subdirectory
            os.path.join('Debug', lib_name),   # In Debug subdirectory
            # Add other potential paths here if necessary based on build logs
        ]

        # Construct the full paths to check
        cmake_library_output_full_path = os.path.abspath(cmake_library_output_dir)
        full_paths_to_check = [
            os.path.join(cmake_library_output_full_path, p)
            for p in potential_library_paths_relative_to_output
        ]

        lib_path_in_cmake_output = None
        # Iterate through potential paths and find the first one that exists
        for p in full_paths_to_check:
            if os.path.exists(p):
                lib_path_in_cmake_output = p
                break # Found the library, stop searching

        # If the library was not found in any of the expected locations, raise an error
        if not lib_path_in_cmake_output:
            # Try to list files in the output directory to help diagnose why it wasn't found
            output_dir_contents = []
            if os.path.exists(cmake_library_output_full_path):
                # Walk through the output directory and list files relative to it
                for root, _, files in os.walk(cmake_library_output_full_path):
                     for f in files:
                          output_dir_contents.append(os.path.relpath(os.path.join(root, f), cmake_library_output_full_path))

            raise FileNotFoundError(
                f"Built library '{lib_name}' not found in expected paths under {cmake_library_output_full_path}. "
                f"Checked: {full_paths_to_check}. "
                f"Directory contents include: {output_dir_contents}"
            )

        # --- Perform the Copy ---
        # Destination path is inside the Python package directory within the build root
        dest_lib_path = os.path.join(python_package_dir, lib_name)
        print(f"Copying built library from {lib_path_in_cmake_output} to {dest_lib_path}")
        shutil.copy(lib_path_in_cmake_output, dest_lib_path)

        # Note: setuptools' bdist_wheel should automatically include files copied
        # into the package directory within self.build_lib.
        # package_data={'bitmap2svg': ['*.so', '*.dylib', '*.dll']} in setup()
        # helps bdist_wheel know *what patterns* of files to look for in that location.


# --- setuptools setup function ---
setup(
    # Package name (on PyPI and) used for import
    name='bitmap2svg',
    version='0.2.0', 

    # Automatically find package directories (the folder with __init__.py)
    packages=find_packages(),

    # Include non-code files (like our built shared library) in the package
    # The keys are package names, values are lists of glob patterns
    # These patterns are matched against files found *within* the package directory
    # after the build_ext step has potentially copied them there.
    package_data={'bitmap2svg': ['*.so', '*.dylib', '*.dll']},

    # Specify custom commands to override default setuptools commands
    # We use our CMakeBuild class instead of the default build_ext
    cmdclass={
        'build_ext': CMakeBuild,
    },

    # Project metadata
    author="Xiaonan (Nice) Wang",
    author_email="wangxiaonannice@gmail.com", 
    description="Convert bitmap images to SVG using Potrace via Python",
    long_description=open('README.md').read() if os.path.exists('README.md') else '', 
    long_description_content_type='text/markdown',
    url="https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg", 
    license="MIT", 

    # Classifiers help users find your package on PyPI
    # See https://pypi.org/classifiers/
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", 
        "Development Status :: 3 - Alpha", 
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics :: Converters",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    # Minimum Python version required
    python_requires='>=3.8',

    # Runtime dependencies
    install_requires=[
        'Pillow',
    ],

    # Build dependencies needed by setup.py itself (specifically our CMakeBuild command)
    # pip build backend will ensure these are available before running setup.py
    setup_requires=[
        'cmake', # Need cmake executable to run CMakeBuild
        # setuptools and wheel are implicitly required by 'python -m build'
    ],

    # Note about C++ dependencies:
    # The C++ code depends on the Potrace C library. This is a *system-level* dependency
    # needed during the C++ build process (handled by CMake and the workflow)
    # AND as a *runtime* dependency on the user's machine where the Python library is used.
    # Users installing the wheel must install Potrace separately on their OS (e.g., via apt, brew).
)
