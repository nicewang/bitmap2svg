import os
import sys
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob  # Required to find library files


# Get the directory containing setup.py
here = os.path.dirname(__file__)


# Define a placeholder marker for pybind11 include paths.
# We will resolve this to actual paths inside the build_extensions method.
PYBIND11_INCLUDE_PLACEHOLDER = "__PYBIND11_INCLUDE_DIR_PLACEHOLDER__"


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

    # print(f"Attempting to find site-packages in prefix: {prefix}") # Uncomment for debugging
    for path in potential_paths:
        if os.path.exists(path):
            # print(f"Found site-packages at: {path}") # Uncomment for debugging
            return path
    # print(f"Could not find site-packages in potential locations under {prefix}.") # Uncomment for debugging
    return None


# Define the C++ extension module(s) to be built.
# Use placeholders for dependency include paths that need runtime lookup during build.
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
        # Initial list of include directories for the C++ compiler.
        # Use placeholders for pybind11. Project's own headers here.
        # Dependency include paths will be resolved and added dynamically by the BuildExt class.
        include_dirs=[
            # Placeholders for pybind11 include paths. Will be replaced in BuildExt.build_extensions.
            PYBIND11_INCLUDE_PLACEHOLDER,
            # Path to our own project's C++ headers. Use 'here' for absolute path robustness.
            os.path.join(here, 'bitmap2svg', 'cpp'),
        ],
        # Specify the language of the source files.
        language='c++',
        # Linker flags and additional compiler flags will be added dynamically by BuildExt.
        # We don't specify library names (-l) or library search paths (-L) here directly.
    ),
]


# Custom build extension command to add compiler and linker-specific options,
# especially to find and link external dependencies like OpenCV and resolve placeholders.
class BuildExt(build_ext):
    """Custom build extension command that finds and adds build options for dependencies."""
    # Default compiler options for different platforms.
    c_opts = {
        'msvc': ['/EHsc', '/std:c++17', '/DNOMINMAX'], # /DNOMINMAX helps avoid conflicts with Windows.h and std::min/max
        'unix': ['-std=c++17', '-fPIC'], # -fPIC is crucial for building shared libraries on Unix/Linux/macOS
    }
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
    # Dependency lookups and placeholder replacements happen here, AFTER build dependencies are installed.
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


        # --- Resolve dependency paths and add build options ---

        # 1. Resolve Pybind11 include paths from the build environment.
        pybind11_include_dir = None
        try:
            # Import pybind11 now. It should be available as it's in build-system.requires or install_requires.
            import pybind11
            pybind11_include_dir = pybind11.get_include()
            # pybind11_user_include_dir = pybind11.get_include(user=True) # Uncomment if user includes needed
            print(f"Resolved pybind11 include path: {pybind11_include_dir}")
            # print(f"Resolved pybind11 user include path: {pybind11_user_include_dir}") # Uncomment if user includes needed
        except ImportError:
             print("CRITICAL ERROR: pybind11 not available during build_extensions!")
             # Raising an error here will stop the build process cleanly.
             raise RuntimeError("pybind11 not found. Ensure it's in build-system.requires or install_requires.")
        except Exception as e:
             print(f"CRITICAL ERROR: Failed to get pybind11 include paths: {e}")
             raise # Re-raise any other error


        # 2. Find and add OpenCV linker and include flags.
        # This logic searches for OpenCV native libraries and headers within the site-packages directory.
        opencv_lib_dir = None
        opencv_include_dir = None

        print("Attempting to find OpenCV installation by searching site-packages...")
        # Use sys.prefix which points to the root of the build environment (or target env in editable mode)
        site_packages_dir = find_site_packages(sys.prefix) # Use the helper function

        if site_packages_dir:
            # Search for the 'cv2' package directory within site-packages.
            cv2_package_dir = os.path.join(site_packages_dir, 'cv2')
            if os.path.exists(cv2_package_dir):
                 print(f"Found cv2 package directory: {cv2_package_dir}")

                 # Define potential directories where OpenCV native libraries might be located
                 potential_lib_dirs = [
                     cv2_package_dir, # Often libs are directly here or in a subdir
                     os.path.join(cv2_package_dir, '.dylibs') if sys.platform == 'darwin' else None, # Common on macOS wheels
                     os.path.join(cv2_package_dir, 'lib') if sys.platform != 'win32' else None, # Some package structures on Unix/Linux/macOS
                     os.path.join(cv2_package_dir, 'bin') if sys.platform == 'win32' else None, # Common for Windows DLLs
                     # Fallback: environment's standard lib directory (less common for opencv-python wheels, but possible)
                     os.path.join(sys.prefix, 'lib'),
                 ]
                 # Filter out None paths if platform check resulted in None.
                 potential_lib_dirs = [p for p in potential_lib_dirs if p is not None]
                 # print(f"Checking potential OpenCV library locations: {potential_lib_dirs}") # Uncomment for debugging

                 # Iterate through potential library directories to find the core OpenCV dynamic library file.
                 for lib_dir_candidate in potential_lib_dirs:
                     if os.path.exists(lib_dir_candidate):
                         core_lib_pattern = None
                         if sys.platform == 'darwin': # macOS (dynamic library suffix)
                             core_lib_pattern = os.path.join(lib_dir_candidate, 'libopencv_core*.dylib')
                         elif sys.platform.startswith('linux'): # Linux (shared object suffix)
                             core_lib_pattern = os.path.join(lib_dir_candidate, 'libopencv_core*.so')
                         elif sys.platform == 'win32': # Windows (DLL suffix)
                             core_lib_pattern = os.path.join(lib_dir_candidate, 'opencv_core*.dll') # Check for DLL

                         if core_lib_pattern:
                             found_libs = glob.glob(core_lib_pattern)
                             if found_libs:
                                 # Found matching library files, assume this is the correct lib directory
                                 opencv_lib_dir = lib_dir_candidate
                                 print(f"Found OpenCV library files matching '{core_lib_pattern}' in '{lib_dir_candidate}'")
                                 break # Found the directory, stop searching potential_lib_dirs
                             # else: print(f"No files matching '{core_lib_pattern}' found in '{lib_dir_candidate}'") # Uncomment for debugging
                         # else: print(f"No core library pattern defined for platform {sys.platform} or path {lib_dir_candidate}") # Uncomment for debugging


                 # Now try to find the OpenCV include directory.
                 # Headers are often in sys.prefix/include/opencv2 or sys.prefix/include/opencv4,
                 # but sometimes might be located within the package structure itself (less common for headers).
                 print("Attempting to find OpenCV include directory.")
                 potential_include_dirs = [
                      # Check standard environment include paths first (most common location for headers)
                      os.path.join(sys.prefix, 'include', 'opencv4'), # Common for OpenCV 4+
                      os.path.join(sys.prefix, 'include', 'opencv2'), # Common for OpenCV 2+
                      os.path.join(sys.prefix, 'include'), # Fallback (sometimes headers are directly here)
                      # Check within the cv2 package directory as a fallback (less common)
                      os.path.join(cv2_package_dir, 'include', 'opencv4') if os.path.exists(cv2_package_dir) else None,
                      os.path.join(cv2_package_dir, 'include', 'opencv2') if os.path.exists(cv2_package_dir) else None,
                      os.path.join(cv2_package_dir, 'include') if os.path.exists(cv2_package_dir) else None,
                 ]
                 # Filter out None paths
                 potential_include_dirs = [p for p in potential_include_dirs if p is not None]
                 # print(f"Checking potential OpenCV include locations: {potential_include_dirs}") # Uncomment for debugging

                 for inc_dir_candidate in potential_include_dirs:
                     # Check for a known header file that exists in the include dir to confirm it's likely OpenCV includes.
                     if os.path.exists(os.path.join(inc_dir_candidate, 'opencv2', 'opencv.hpp')):
                         opencv_include_dir = inc_dir_candidate
                         print(f"Found OpenCV include directory at: {opencv_include_dir}")
                         break # Found a good candidate

            # else: print(f"cv2 package directory not found in site-packages at {cv2_package_dir}") # Uncomment for debugging
        # else: print("Site-packages directory not found.") # Uncomment for debugging


        # --- Configure extension build based on found dependency paths ---

        # Iterate through each extension defined in ext_modules.
        for ext in self.extensions:
            # Ensure extra_compile_args and extra_link_args are lists before extending them.
            if not hasattr(ext, 'extra_compile_args'):
                ext.extra_compile_args = []
            if not hasattr(ext, 'extra_link_args'):
                ext.extra_link_args = []
                
            # --- Add Pybind11 include path ---
            resolved_include_dirs = []
            for inc_dir_placeholder in ext.include_dirs:
                 if inc_dir_placeholder == PYBIND11_INCLUDE_PLACEHOLDER:
                      if pybind11_include_dir: # Only add if pybind11 was resolved
                          if pybind11_include_dir not in resolved_include_dirs:
                               resolved_include_dirs.append(pybind11_include_dir)
                               # print(f"Added resolved pybind11 include: {pybind11_include_dir}") # Uncomment for debugging
                      # else: print(f"Warning: Pybind11 include placeholder found but pybind11 not resolved.") # Uncomment for debugging
                 # Add other non-placeholder include dirs (like project headers)
                 elif inc_dir_placeholder not in resolved_include_dirs:
                      resolved_include_dirs.append(inc_dir_placeholder)

            ext.include_dirs = resolved_include_dirs # Update include_dirs with resolved paths


            # --- Add found OpenCV include and library paths ---
            if opencv_lib_dir and opencv_include_dir:
                print("Configuring extension build with found OpenCV paths.")

                # Add library search path for the linker (-L).
                # This tells the linker where to look for the libraries we specify with -l.
                link_args.append(f'-L{opencv_lib_dir}')
                # print(f"Adding linker search path: -L{opencv_lib_dir}") # Uncomment for debugging


                # Add necessary OpenCV libraries to link against (-l).
                # Based on cpp_svg_converter.cpp code, we need core and imgproc.
                # Add other modules here if your C++ code uses them.
                # The linker finds the correct version/suffix (.dylib, .so, .dll) if the search path (-L)
                # is set correctly and the base library name (-lNAME) is provided.
                opencv_libs_needed = ['opencv_core', 'opencv_imgproc'] # Add other modules if needed
                # print(f"Attempting to link libraries: {opencv_libs_needed}") # Uncomment for debugging

                for lib_name in opencv_libs_needed:
                     # Handle potential platform-specific library naming conventions.
                     if sys.platform == 'win32' and ct == 'msvc':
                         # Windows MSVC linking requires .lib import libraries.
                         # Their names often include version suffix (e.g., opencv_core460.lib).
                         # Finding the exact .lib name robustly is tricky.
                         # A common heuristic is searching for the .lib file matching a pattern in the lib dir.
                         lib_file_pattern = os.path.join(opencv_lib_dir, f'{lib_name}*.lib')
                         found_libs = glob.glob(lib_file_pattern)
                         if found_libs:
                             # Add the exact file name of the found .lib file.
                             lib_filename = os.path.basename(found_libs[0])
                             link_args.append(lib_filename)
                             # print(f"Adding Windows import lib: {lib_filename}") # Uncomment for debugging
                         else:
                              # Fallback: add the standard .lib name. This might not work if the .lib file name includes versioned.
                              fallback_lib_name = f'{lib_name}.lib'
                              link_args.append(fallback_lib_name)
                              # print(f"Warning: Specific Windows import lib '{lib_name}*.lib' not found by glob in '{opencv_lib_dir}'. Adding '{fallback_lib_name}' as fallback.") # Uncomment for debugging
                     else: # Unix/macOS/other compilers (dynamic linking using -lNAME)
                         # Add the base library name (e.g., -lopencv_core).
                         # The linker finds the correct .dylib or .so file using the -L path.
                         link_args.append(f'-l{lib_name}')
                         # print(f"Adding library: -l{lib_name}") # Uncomment for debugging


                # Add found OpenCV include path(s) to the compiler includes (-I).
                # This is necessary for the C++ compiler to find OpenCV header files (#include <opencv2/...>).
                # Explicitly add found OpenCV include directory.
                # Add it after pybind11 includes and before project includes to prioritize it
                # over default system paths like /usr/local/include.
                # Find the position of the project's include dir or just append after resolved pybind11
                project_include_pos = -1
                for i, path in enumerate(ext.include_dirs):
                    if path == os.path.join(here, 'bitmap2svg', 'cpp'):
                        project_include_pos = i
                        break

                insert_pos = project_include_pos if project_include_pos != -1 else len(ext.include_dirs) # Insert before project include, or at the end

                # Add the base include dir (e.g., .../include or .../site-packages/cv2/include).
                if opencv_include_dir not in ext.include_dirs:
                     ext.include_dirs.insert(insert_pos, opencv_include_dir)
                     print(f"Adding include directory: {opencv_include_dir} for compiler.")


                # Add the specific opencv2 or opencv4 subdirectory if headers are organized that way
                # Check if headers are directly in opencv_include_dir/opencv2
                opencv2_subdir = os.path.join(opencv_include_dir, 'opencv2')
                if os.path.exists(os.path.join(opencv2_subdir, 'opencv.hpp')):
                     if opencv2_subdir not in ext.include_dirs:
                          ext.include_dirs.append(opencv2_subdir) # Append this subdirectory path
                          print(f"Adding include directory: {opencv2_subdir} for compiler.")
                # Check if headers are directly in opencv_include_dir/opencv4
                opencv4_subdir = os.path.join(opencv_include_dir, 'opencv4')
                if os.path.exists(os.path.join(opencv4_subdir, 'opencv.hpp')):
                     if opencv4_subdir not in ext.include_dirs:
                          ext.include_dirs.append(opencv4_subdir) # Append this subdirectory path
                          print(f"Adding include directory: {opencv4_subdir} for compiler.")


            else:
                # This block is hit if site-packages is not found, or cv2 package is not found,
                # or if the search for library/include directories within site-packages fails.
                print("\n" + "="*60)
                print("CRITICAL WARNING: Could not find OpenCV installation for C++ extension linking.")
                print(f"Searched within site-packages based on environment prefix '{sys.prefix}'.")
                print("Please ensure 'opencv-python' is installed in the active environment")
                print("and its files are located in a standard structure within site-packages.")
                print("The build may proceed, but linking will likely fail with 'symbol not found' errors.")
                print("="*60 + "\n")
                # Consider uncommenting the line below to force the build to fail loudly
                # if linking is critical.
                # raise RuntimeError("OpenCV installation not found or incomplete for C++ extension build.")


            # --- End OpenCV configuration ---

            # Apply the collected compiler options and linker arguments to each extension
            # Ensure extra_compile_args and extra_link_args are lists before extending them.
            for ext in self.extensions:
                if not hasattr(ext, 'extra_compile_args'):
                    ext.extra_compile_args = []
                if not hasattr(ext, 'extra_link_args'):
                    ext.extra_link_args = []

                # Extend with the dynamically determined options/args
                ext.extra_compile_args.extend(opts)
                ext.extra_link_args.extend(link_args)

                # Print the final arguments being used for each extension for debugging.
                print(f"\nFinal arguments for building extension '{ext.name}':")
                print(f"  Compile args: {ext.extra_compile_args}")
                print(f"  Link args: {ext.extra_link_args}")
                print(f"  Include dirs: {ext.include_dirs}\n")


            # Call the base class build_extensions method to perform the actual compilation and linking.
            # This is where the C++ compiler (e.g., clang++, x86_64-linux-gnu-g++) and linker (e.g., ld) are invoked
            # using the arguments gathered in the steps above.
            build_ext.build_extensions(self)


# The main setup function call that configures the package for setuptools.
setup(
    name='bitmap2svg', # The name of your package on PyPI.
    version='0.2.1', # Current version of your package.
    author='Xiaonan (Nice) Wang', # Package author.
    author_email='wangxiaonannice@gmail.com', # Author's email.
    description='A bitmap to SVG converter using C++ and pybind11', # Short description.
    long_description='', # Long description (Ideally, load from README.md: open('README.md').read()).
    # packages=setuptools.find_packages(), # Automatically find Python packages (directories containing __init__.py).
    packages=setuptools.find_packages(where='.'), # Specify where to look for packages if not root (useful for src layout).
    # Specify the C++ extension module(s) to build.
    ext_modules=ext_modules,
    # Tell setuptools to use our custom BuildExt command when building extensions.
    cmdclass={'build_ext': BuildExt},
    # Declare required Python dependencies. pip will ensure these are installed.
    # The BuildExt class finds the native parts installed by 'opencv-python'.
    install_requires=[
        'pybind11>=2.6', # Required for the C++ bindings.
        'Pillow', # Likely used for image loading/processing in the Python part.
        'numpy', # OpenCV Python interface often uses NumPy arrays.
        'opencv-python' # Provides the cv2 Python module and native libraries.
    ],
    zip_safe=False, # Required for packages with C++ extensions like this.
    python_requires='>=3.6', # Specify the minimum required Python version.
    # Add URLs, classifiers, keywords etc. as needed for PyPI
    # url='https://github.com/yourusername/bitmap2svg', # Example URL.
    # classifiers=[
    #     'Programming Language :: C++',
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent', # Specify specific OS if known
    # ],
    # keywords='svg bitmap vector graphics opencv conversion',
)
