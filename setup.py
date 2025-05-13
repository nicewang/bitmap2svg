import os
import sys
import subprocess
import shutil

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as SetuptoolsBuildExt

# --- Custom build command to integrate CMake ---
class CMakeBuild(SetuptoolsBuildExt):
    def run(self):
        # Ensure CMake is available
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions.")

        # setuptools' temporary build directory
        cmake_build_dir = os.path.join(self.build_temp, 'cmake_build')
        os.makedirs(cmake_build_dir, exist_ok=True)

        # Get the path to the directory containing setup.py and CMakeLists.txt
        project_source_dir = os.path.abspath(os.path.dirname(__file__))

        # --- CMake Configuration ---
        cmake_command = ['cmake', project_source_dir]

        cmake_library_output_dir = os.path.join(cmake_build_dir, 'lib')
        os.makedirs(cmake_library_output_dir, exist_ok=True)

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(cmake_library_output_dir)}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={"Debug" if self.debug else "Release"}',
        ]

        if sys.platform == 'win32':
            cmake_args.extend(['-G', 'Visual Studio 17 2022', '-A', 'x64'])

        cmake_args.extend(self.define or [])

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

        if sys.platform != 'win32':
             build_args.extend(['--', '-j4'])
        else:
             build_args.extend(['--', '/m'])

        build_args.extend(['--config', 'Debug' if self.debug else 'Release'])

        print(f"Running CMake build in {cmake_build_dir}")
        print(f"Command: {' '.join(build_command + build_args)}")
        try:
            subprocess.check_call(build_command + build_args, cwd=cmake_build_dir)
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed: {e}")
            raise


        # --- Copy the built library to the final package directory ---
        python_package_dir = os.path.join(self.build_lib, 'bitmap2svg')
        os.makedirs(python_package_dir, exist_ok=True)

        lib_patterns = {
            'linux': 'libbitmap_to_svg.so',
            'darwin': 'libbitmap_to_svg.dylib',
            'win32': 'bitmap_to_svg.dll'
        }
        lib_name = lib_patterns.get(sys.platform)
        if not lib_name:
             raise RuntimeError(f"Unsupported platform: {sys.platform}")

        potential_library_paths_relative_to_output = [
            lib_name,
            os.path.join('Release', lib_name),
            os.path.join('Debug', lib_name),
        ]

        cmake_library_output_full_path = os.path.abspath(cmake_library_output_dir)
        full_paths_to_check = [
            os.path.join(cmake_library_output_full_path, p)
            for p in potential_library_paths_relative_to_output
        ]

        lib_path_in_cmake_output = None
        for p in full_paths_to_check:
            if os.path.exists(p):
                lib_path_in_cmake_output = p
                break

        if not lib_path_in_cmake_output:
             output_dir_contents = []
             if os.path.exists(cmake_library_output_full_path):
                 for root, _, files in os.walk(cmake_library_output_full_path):
                      for f in files:
                           output_dir_contents.append(os.path.relpath(os.path.join(root, f), cmake_library_output_full_path))

             raise FileNotFoundError(
                 f"Built library '{lib_name}' not found in expected paths under {cmake_library_output_full_path}. "
                 f"Checked: {full_paths_to_check}. "
                 f"Directory contents include: {output_dir_contents}"
             )

        dest_lib_path = os.path.join(python_package_dir, lib_name)
        print(f"Copying built library from {lib_path_in_cmake_output} to {dest_lib_path}")
        shutil.copy(lib_path_in_cmake_output, dest_lib_path)


# --- setuptools setup function ---
setup(
    name='bitmap2svg',
    version='0.2.0',

    packages=find_packages(),

    package_data={'bitmap2svg': ['*.so', '*.dylib', '*.dll']},

    cmdclass={
        'build_ext': CMakeBuild,
    },

    ext_modules=[Extension('_dummy_extension', sources=[])], # <-- Add this line

    # Project metadata
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

    setup_requires=[
        'cmake',
    ],
)
