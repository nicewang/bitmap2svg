import os
import sys
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as SetuptoolsBuildExt # Rename to avoid conflict

# Custom build command to run CMake
class CMakeBuild(SetuptoolsBuildExt):
    def run(self):
        # Ensure build directory exists
        # self.build_temp is the temporary build directory set by setuptools
        # We'll create a subdirectory for CMake builds
        cmake_build_dir = os.path.join(self.build_temp, 'cmake_build')
        os.makedirs(cmake_build_dir, exist_ok=True)

        # Configure command
        cmake_command = ['cmake', os.path.abspath(os.path.dirname(__file__))] # Path to CMakeLists.txt
        # Set the output directory for the shared library within the package directory
        # This path needs to be relative to the install root, which setuptools determines.
        # A common pattern is to install into build_lib/package_name/
        # We set CMake output to build_temp/cmake_build/lib/
        # Then in the 'copy_extensions' step, we copy it to build_lib/package_name/
        output_lib_dir = os.path.join(cmake_build_dir, 'lib')
        os.makedirs(output_lib_dir, exist_ok=True) # Ensure output dir exists

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(output_lib_dir)}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            # Set build type (Debug/Release)
            f'-DCMAKE_BUILD_TYPE={"Debug" if self.debug else "Release"}',
        ]

        # Add generator for Windows
        if sys.platform == 'win32':
            # Use Ninja or Visual Studio. Ninja is often easier in CI.
            # Ensure Ninja is installed if using -G Ninja.
            # Let's stick with VS for common Windows runners.
            # Need to specify architecture (x64 or Win32)
            cmake_args.extend(['-G', 'Visual Studio 17 2022', '-A', 'x64']) # Adjust VS version if needed
            # Vcpkg integration is handled by CMakeLists.txt using VCPKG_ROOT env var

        # Add any extra args specified by the user
        cmake_args.extend(self.define or []) # setuptools passes -D options here

        print(f"Running CMake configure in {cmake_build_dir}: {' '.join(cmake_command + cmake_args)}")
        subprocess.check_call(cmake_command + cmake_args, cwd=cmake_build_dir)

        # Build command
        build_command = ['cmake', '--build', '.']
        build_args = []

        # Add parallel jobs
        if sys.platform != 'win32':
             build_args.extend(['--', '-j4']) # Adjust job count as needed
        else:
             build_args.extend(['--', '/m']) # Parallel build for MSVC

        # Add build config for multi-config generators (like Visual Studio)
        build_args.extend(['--config', 'Debug' if self.debug else 'Release'])

        print(f"Running CMake build in {cmake_build_dir}: {' '.join(build_command + build_args)}")
        subprocess.check_call(build_command + build_args, cwd=cmake_build_dir)

        # --- Copy the built library to the final package directory ---
        # self.build_lib is the root of the installed package structure (e.g., build/lib/)
        # The library needs to go into build_lib/python_bitmap_to_svg/
        python_package_dir = os.path.join(self.build_lib, 'python_bitmap_to_svg')
        os.makedirs(python_package_dir, exist_ok=True)

        # Find the built library file(s) in the CMake output directory
        # The library name depends on platform and configuration
        lib_patterns = {
            'linux': 'libbitmap_to_svg.so',
            'darwin': 'libbitmap_to_svg.dylib',
            'win32': 'bitmap_to_svg.dll' # Assuming Release config filename
        }
        lib_name = lib_patterns.get(sys.platform)
        if not lib_name:
             raise RuntimeError(f"Unsupported platform: {sys.platform}")

        # Adjust expected filename for multi-config generators (e.g., VS)
        if sys.platform == 'win32':
             # VS might append _d for debug, but we build Release in workflow
             lib_path_in_cmake_output = os.path.join(output_lib_dir, 'Release', lib_name) # VS puts in Config subdir
             if not os.path.exists(lib_path_in_cmake_output):
                 lib_path_in_cmake_output = os.path.join(output_lib_dir, lib_name) # Or directly in output_lib_dir
        else:
             lib_path_in_cmake_output = os.path.join(output_lib_dir, lib_name)


        if not os.path.exists(lib_path_in_cmake_output):
             raise FileNotFoundError(f"Built library not found at expected path: {lib_path_in_cmake_output}")


        dest_lib_path = os.path.join(python_package_dir, lib_name)
        print(f"Copying library from {lib_path_in_cmake_output} to {dest_lib_path}")
        shutil.copy(lib_path_in_cmake_output, dest_lib_path)


setup(
    name='bitmap2svg',
    version='0.2.0',
    packages=['bitmap2svg'],
    package_data={'python_bitmap_to_svg': ['*.so', '*.dylib', '*.dll']},
    cmdclass={
        'build_ext': CMakeBuild,
    },

    author="Xiaonan (Nice) Wang", 
    author_email="wangxiaonannice@gmail.com", 
    description="Convert bitmap images to SVG using Potrace via Python",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url="https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg", 
    license="MIT", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics :: Converters",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8', 
    install_requires=[
        'Pillow', 
    ],
    setup_requires=['cmake'], 
)
