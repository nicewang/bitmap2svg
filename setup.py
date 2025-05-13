import os
import sys
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Get the directory containing setup.py
here = os.path.dirname(__file__)

# A hack to get the pybind11 include directory
# See https://github.com/pybind/pybind11/blob/master/docs/compiling.rst#building-with-setuptools
class get_pybind_include(object):
    """Helper class to determine the pybind11 include directory
    The purpose of this class is to postpone importing pybind11
    until it is actually needed, as PyPI doesn't universally work."""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# Define the C++ extension module
ext_modules = [
    Extension(
        # The name of the Python module that will be imported
        'bitmap2svg_core', # Correct extension name
        # The source files for the C++ extension
        # Paths are relative to the directory containing setup.py
        sources=[
            'bitmap2svg/cpp/bindings.cpp',  
            'bitmap2svg/cpp/cpp_svg_converter.cpp' 
        ],
        # Compile options
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
            # Path to our own C++ headers (the bitmap2svg directory)
            'bitmap2svg/cpp', 
        ],
        language='c++',
        # We will add compiler-specific args in the BuildExt class
    ),
]

# Custom build extension command to add compiler-specific flags
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/std:c++17'],
        'unix': ['-std=c++17'],
    }
    link_opts = {
        'msvc': [],
        'unix': [],
    }

    # Add macOS specific flags
    if sys.platform == 'darwin':
        # Explicitly use libc++ standard library on macOS
        c_opts['unix'] += ['-stdlib=libc++']
        link_opts['unix'] += ['-stdlib=libc++']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_args = self.link_opts.get(ct, [])

        # Add -DNDEBUG for release builds
        if sys.platform == 'win32':
             opts.append('/DNDEBUG')
        else:
             opts.append('-DNDEBUG')


        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_args
        build_ext.build_extensions(self)

setup(
    name='bitmap2svg', 
    version='0.2.1',
    author='Xiaonan (Nice) Wang',
    author_email='wangxiaonannice@gmail.com',
    description='A bitmap to SVG converter using C++ and pybind11',
    long_description='',
    packages=setuptools.find_packages(), 
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=['pybind11>=2.6', 'Pillow', 'numpy', 'opencv-python'],
    zip_safe=False,
)
