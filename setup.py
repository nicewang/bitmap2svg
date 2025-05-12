from setuptools import setup

setup(
    name="bitmap2svg",
    version="0.2.0",
    packages=["bitmap2svg"],
    package_data={
        "bitmap2svg": ["CMakeLists.txt"],
    },
    include_package_data=True,
)
