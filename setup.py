from setuptools import setup

setup(
    name="bitmap2svg",
    version="0.2.2",
    description="A library to convert bitmaps to SVG, using C++ and OpenCV.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Xiaonan (Nice) Wang",
    author_email="wangxiaonannice@gmail.com",
    license="MIT",  
    url="https://github.com/nicewang/bitmap2svg",
    packages=["bitmap2svg"],
)
