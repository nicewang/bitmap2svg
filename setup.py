from setuptools import setup, find_packages, Extension
import os
import sys

try:
    import pybind11
except ImportError:
    raise RuntimeError("Pybind11 is required. Please install it with 'pip install pybind11'")

tracer_module = Extension(
    'bitmap2svg._tracer',
    sources=[
        'bitmap2svg/cpp/bitmap_to_svg.cpp',
        'bitmap2svg/cpp/bindings.cpp',
    ],
    libraries=['potrace'],
    include_dirs=[
        'bitmap2svg/cpp',
        pybind11.get_include(),
        pybind11.get_include(user=True),
    ],
    extra_compile_args=['-std=c++11', '-fPIC', '-stdlib=libc++'],
    extra_link_args=[],
    language='c++'
)

setup(
    name='bitmap2svg',
    version='0.2.0',
    packages=find_packages(),
    ext_modules=[tracer_module],
    install_requires=[
        'Pillow',
        'numpy',
        'pybind11>=2.6',
    ],
    author='Xiaonan (Nice) Wang',
    description='Convert bitmap images to SVG',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Multimedia :: Graphics :: Convert',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
)
