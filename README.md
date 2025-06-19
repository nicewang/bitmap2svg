# bitmap2svg
## Version: 0.2.3.3
### 1. Introduction
A library to convert bitmaps to SVG, using C++, OpenCV, with potential CUDA/FAISS GPU acceleration.

* The `max_svg_size` of converted svg is `10000`

### 2. Package Structure
```
${project_workspace}/
├── external/
│   ├── faiss_sources/
├── bitmap2svg/
│   ├── __init__.py
│   └── cpp/
│       ├── bitmap_to_svg.cpp
│       ├── bitmap_to_svg.h
│       ├── bindings.cpp
├── pyproject.toml
├── setup.py
├── CMakeLists.txt
└── README.md               
```
### 3. Install
#### 3.1 Local Install
```bash
cd ${project_workspace}/
pip install --upgrade pybind11 setuptools wheel
pip install -e .
# or
pip install .
```
#### 3.2 Remote Install
```bash
# install prerequisites firstly
pip install -q opencv-python scikit-image pillow
pip install scikit-build-core cmake ninja pybind11
# then install this lib
pip install git+https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg-cu-v0.2.3.3

```