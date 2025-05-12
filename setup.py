from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

cpp_sources = [
    'bitmap2svg/cpp/bitmap_to_svg.cpp',
    'bitmap2svg/cpp/bindings.cpp',
]

ext_modules = [
    Pybind11Extension(
        "_bitmap2svg_core",
        sources=cpp_sources,
        include_dirs=[
            'bitmap2svg/cpp',
        ],
        libraries=["potrace"],
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="bitmap2svg_potrace",
    version="0.2.0",
    author="Xiaonan (Nice) Wang",
    author_email="wangxiaonannice@gmail.com",
    description="A Python wrapper for bitmap to SVG conversion using Potrace and pybind11",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    url="https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg",
    packages=find_packages(where='.'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
