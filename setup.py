import os
import sys
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob 


# Get the directory containing setup.py
here = os.path.dirname(__file__)


# A hack to get the pybind11 include directory.
# This class MUST be defined BEFORE it is used in the ext_modules list below.
class get_pybind_include(object):
    """Helper class to determine the pybind11 include directory.
    Postpones importing pybind11 until needed."""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        try:
            # Attempt to import pybind11 when the string representation is needed.
            # This might happen during setup dependency resolution or build.
            import pybind11
            return pybind11.get_include(self.user)
        except ImportError:
            # This warning is acceptable if pybind11 is listed in install_requires
            # and setuptools handles its installation before calling build_ext.
            # However, if it's not installed and build_ext runs, compilation will fail anyway.
            print("Warning: pybind11 not importable. Build may fail if it's truly missing.")
            return "" # Return an empty string if pybind11 is not found


# Helper function to find the site-packages directory in a given environment prefix.
# This is often where packages like opencv-python place their files.
def find_site_packages(prefix):
    """Finds the site-packages directory for a Python environment prefix."""
    # Common locations for site-packages based on Python version and OS.
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    potential_paths = [
        os.path.join(prefix, 'lib', f'python{python_version}', 'site-packages'), # Unix/Linux/macOS standard
        os.path.join(prefix, 'lib', 'site-packages'), # Some older or simpler venvs
        os.path.join(prefix, 'site-packages'), # Some distributions/venv structures
    ]
    if sys.platform == 'win32':
         # Windows paths are different
         potential_paths = [
             os.path.join(prefix, 'Lib', 'site-packages'),
         ]

    print(f"Attempting to find site-packages in prefix: {prefix}")
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found site-packages at: {path}")
            return path
    print(f"Could not find site-packages in potential locations under {prefix}.")
    return None


# Define the C++ extension module(s) to be built.
ext_modules = [
    Extension(
        # The name of the Python module that will be imported in Python (e.g., `import bitmap2svg_core`).
        'bitmap2svg_core', # Correct extension name
        # The source files for the C++ extension relative to the directory containing setup.py.
        # Use os.path.join(here, ...) for robustness against where setup.py is called from.
        sources=[
            os.path.join(here, 'bitmap2svg', 'cpp', 'bindings.cpp'),  
            os.path.join(here, 'bitmap2svg', 'cpp', 'cpp_svg_converter.cpp') 
        ],
        # Compile options
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            # Path to our own C++ headers (the bitmap2svg directory)
            os.path.join(here, 'bitmap2svg', 'cpp'),
        ],
        language='c++',
        # We don't specify library names (-l) or library search paths (-L) here directly,
        # as they depend on the specific dependency installation found at build time.
        # Compiler and linker arguments will be added dynamically by the BuildExt class.
    ),
]


# Custom build extension command to add compiler and linker-specific options,
# especially to find and link external dependencies like OpenCV.
class BuildExt(build_ext):
    """Custom build extension command that finds and adds build options for dependencies."""
    # Default compiler options for different platforms.
    c_opts = {
        'msvc': ['/EHsc', '/std:c++17', '/DNOMINMAX'], # /DNOMINMAX helps avoid conflicts with Windows.h and std::min/max
        'unix': ['-std=c++17', '-fPIC'], # -fPIC is crucial for building shared libraries on Unix/Linux/macOS
    }
    # Default linker options for different platforms.
    link_opts = {
        'msvc': [],
        'unix': [],
    }

    # Add macOS specific flags (for the C++ standard library).
    if sys.platform == 'darwin':
        # On macOS, explicitly linking against libc++ is often required for compatibility.
        c_opts['unix'] += ['-stdlib=libc++']
        link_opts['unix'] += ['-stdlib=libc++']
        # macOS also sometimes needs explicit rpath settings for dynamic libraries
        # to be found at runtime, but let's address linking first.

    # This method is called by setuptools to build each extension.
    def build_extensions(self):
        # Get the compiler type (e.g., 'msvc', 'unix').
        ct = self.compiler.compiler_type
        # Copy the default options/arguments to avoid modifying the class attributes directly.
        opts = self.c_opts.get(ct, []).copy()
        link_args = self.link_opts.get(ct, []).copy()

        # Add -DNDEBUG for release builds (disables assertions and debug checks).
        if sys.platform == 'win32':
             opts.append('/DNDEBUG')
        else:
             opts.append('-DNDEBUG') # Use -DNDEBUG on Unix-like systems


        # --- Dynamically find and add OpenCV linker and include flags ---
        # This logic searches for OpenCV native libraries and headers within the site-packages directory
        # where packages like opencv-python are typically installed.
        opencv_lib_dir = None
        opencv_include_dir = None

        print("Attempting to find OpenCV installation by searching site-packages...")
        site_packages_dir = find_site_packages(sys.prefix) # Use the helper function

        if site_packages_dir:
            # Search for the 'cv2' package directory within site-packages.
            cv2_package_dir = os.path.join(site_packages_dir, 'cv2')
            if os.path.exists(cv2_package_dir):
                 print(f"Found cv2 package directory: {cv2_package_dir}")
