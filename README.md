# bitmap2svg
## Version: 0.2.3.2
### 1. Introduction
A library to convert bitmaps to SVG, using C++, OpenCV, with potential CUDA/FAISS GPU acceleration.

* The `max_svg_size` of converted svg is `10000`
* The *GPU Version* of v0.2.3

### 2. Package Structure
```
${project_workspace}/
├── external/
│   ├── faiss_sources/		<- ensembled faiss sources (v1.7.2)
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
Will build the faiss-dependency (ensembled in this project, at external/faiss_sources) while installing this lib.
#### 3.1 Enviroment Requirements
- Python Version <= 3.10
	- Since faiss doesn't support running in environments with python-version > 3.10

#### 3.2 Local Install
```bash
cd ${project_workspace}/
pip install --upgrade pybind11 setuptools wheel
pip install -e .
# or
pip install .
```
#### 3.3 Remote Install
```bash
# install prerequisites firstly
pip install -q opencv-python scikit-image pillow
pip install scikit-build-core cmake ninja pybind11
# then install this lib
pip install git+https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg-cu-v0.2.3.2

```
### 4. Dependencies and the Correspoding Patches
#### faiss
Version: [v1.7.2](https://github.com/facebookresearch/faiss/releases/tag/v1.7.2)

Patches:

- [Modification at external/fa
iss_sources/CMakeLists.txt](patches/external_faiss_sources_cmakelists.diff)
	- `VERSION 1.6.4` -> `VERSION 1.7.2`
- [Modification at external/fa
iss_sources/faiss/gpu/CMakeLists.txt](patches/external_faiss_sources_faiss_gpu_cmakelists.diff)
	- For solving issues of *cublas* while installing.
