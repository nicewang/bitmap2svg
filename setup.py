import setuptools
import os
import sys
import subprocess

def get_pybind11_include_dirs():
    try:
        import pybind11
        return [pybind11.get_include(), pybind11.get_include(user=True)]
    except ImportError:
        try:
            pip_show = subprocess.check_output([sys.executable, '-m', 'pip', 'show', 'pybind11']).decode('utf-8')
            location_line = [line for line in pip_show.splitlines() if line.startswith('Location:')]
            if location_line:
                pybind11_location = location_line[0].split(': ', 1)[1]
                include_path = os.path.join(pybind11_location, 'pybind11', 'include')
                if os.path.exists(include_path):
                     return [include_path]
                alternate_include_path = os.path.join(pybind11_location, 'include')
                if os.path.exists(alternate_include_path):
                      return [alternate_include_path]

        except Exception as e:
             print(f"Warning: Could not automatically find pybind11 include paths: {e}", file=sys.stderr)

        print("Warning: pybind11 not found or could not determine include paths. Extension will likely fail to build.", file=sys.stderr)
        return []

cpp_sources = [
    'bitmap2svg/cpp/bitmap_to_svg.cpp',
    'bitmap2svg/cpp/bindings.cpp',
]

_bitmap2svg_core_extension = setuptools.Extension(
    '_bitmap2svg_core',
    sources=cpp_sources,
    include_dirs=[
        '/usr/local/include',
        '/usr/include',
        'bitmap2svg/cpp',
        *get_pybind11_include_dirs(),
    ],
    library_dirs=[
        '/usr/local/lib',
        '/usr/lib',
    ],
    libraries=['potrace'],
    extra_compile_args=['-std=c++17'],
    language='c++',
)

setuptools.setup(
    name="bitmap2svg_potrace",
    version="0.2.0",
    author="Xiaonan (Nice) Wang",
    author_email="wangxiaonannice@gmail.com",
    description="A Python wrapper for bitmap to SVG conversion using Potrace and pybind11",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    url="https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg",
    packages=setuptools.find_packages(where='.'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Graphics :: Convert",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>=3.8',
    ext_modules=[_bitmap2svg_core_extension],
    zip_safe=False,
)
