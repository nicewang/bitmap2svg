import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os

# Get version from pyproject.toml or define here
# For simplicity, let's define it here and ensure pyproject.toml matches.
VERSION = "0.1.0"

class get_pybind_include(object):
    """Helper class to defer importing pybind11 until it is actually needed."""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# Potrace library paths (users can set these via environment variables if needed)
POTRACE_INCLUDE_DIR = os.environ.get('POTRACE_INCLUDE_DIR')
POTRACE_LIB_DIR = os.environ.get('POTRACE_LIB_DIR')

include_dirs = [
    get_pybind_include(),
    get_pybind_include(user=True), # In case pybind11 is installed user-locally
    'bitmap2svg/cpp' # Location of your cpp files and potentially potracelib.h if bundled
]
if POTRACE_INCLUDE_DIR:
    include_dirs.append(POTRACE_INCLUDE_DIR)

library_dirs = []
if POTRACE_LIB_DIR:
    library_dirs.append(POTRACE_LIB_DIR)

# Define compiler and linker args
extra_compile_args = []
extra_link_args = []
libraries = ['potrace'] # Default for Linux/macOS

if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17', '/EHsc', '/O2']
    libraries = ['potrace'] 

else: # macOS and Linux
    extra_compile_args = ['-std=c++17', '-O2', '-fvisibility=hidden']
    if sys.platform == 'darwin': # macOS specific flags
        extra_compile_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.14'])
        extra_link_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.14'])


ext_modules = [
    Extension(
        'bitmap2svg._bitmap2svg_cpp', # Output module name: bitmap2svg/_bitmap2svg_cpp.so or .pyd
        sources=['bitmap2svg/cpp/bindings.cpp'], # Add your bitmap_to_svg.cpp if it's separate
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[('VERSION_INFO', VERSION)], # Pass version to C++
    ),
]

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    def build_extensions(self):
        # Ensure pybind11 is present for building
        try:
            import pybind11
        except ImportError:
            raise RuntimeError("pybind11 is required to build this extension. "
                               "Please install it with: pip install pybind11")
        
        super().build_extensions()

setup(
    name='bitmap2svg',
    version=VERSION,
    author='Xiaonan (Nice) Wang',
    author_email='wangxiaonannice@gmail.com',
    description='Python wrapper for C++ bitmap to SVG conversion using Potrace',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg',
    packages=['bitmap2svg'],
    package_dir={'bitmap2svg': 'bitmap2svg'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.7', # pybind11 minimums, adjust as needed
    install_requires=[
        'numpy>=1.16',
    ],
    setup_requires=[ # Build-time dependencies
        'pybind11>=2.6.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: C++',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Multimedia :: Graphics :: Converters',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False, # C extensions usually aren't zip_safe
)
