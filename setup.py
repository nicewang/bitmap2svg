from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        'bitmap2svg_core',
        ['bitmap2svg/cpp/bitmap_to_svg.cpp', 'bitmap2svg/cpp/bindings.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
        ],
        libraries=["potrace"],
        language='c++',
        extra_compile_args=['-std=c++11', '-stdlib=libc++'],
    ),
]

setup(
    name='bitmap2svg',
    version='0.2.0',
    author='Xiaonan (Nice) Wang',
    description='Convert bitmap to SVG',
    packages=['bitmap2svg'],
    ext_modules=ext_modules,
    zip_safe=False,
)