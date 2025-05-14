# bitmap2svg
## Version: 0.2.1
### 1. Introduction
A library to convert bitmaps to SVG, using C++ and OpenCV
### 2. Package Structure
```
${project_workspace}/
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
pip install git+https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs.git@v0.2.1-bitmap2svg

```