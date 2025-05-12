import setuptools
import os

try:
    import pybind11
    pybind11_include_dir = pybind11.get_include()
except ImportError:
    pybind11_include_dir = ""

setuptools.setup(
    name="bitmap2svg_potrace",
    version="0.2.0",
    author="Xiao (Nice) Wang",
    author_email="wangxiaonannice@gmail.com",
    description="A Python wrapper for bitmap to SVG conversion using Potrace",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=[
        setuptools.Extension(
            '_bitmap2svg_core',
            [
                'bitmap2svg/cpp/bitmap_to_svg.cpp',
                'bitmap2svg/cpp/bindings.cpp'
            ],
            include_dirs=[
                '/usr/local/include',
                '/usr/include',
                 pybind11_include_dir,
            ],
            library_dirs=[
                '/usr/local/lib',
                '/usr/lib',
            ],
            libraries=['potrace'],
            extra_compile_args=['-std=c++17'],
        )
    ],
)
