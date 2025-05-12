from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys

class get_pybind_include(object):
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        "_bitmap2svg_core",
        sources=[
            "bitmap2svg/cpp/bindings.cpp",
            "bitmap2svg/cpp/bitmap_to_svg.cpp",
        ],
        include_dirs=[
            "bitmap2svg/cpp",
            get_pybind_include(),
        ],
        language="c++",
        extra_compile_args=["-std=c++17"]
    ),
]

setup(
    name="bitmap2svg_potrace",
    version="0.2.0",
    packages=["bitmap2svg"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
