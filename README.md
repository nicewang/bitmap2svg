# bitmap2svg
## Current Version: 0.2.2
### 1. Introduction
A library to convert bitmaps to SVG, using C++ and OpenCV

* The `max_svg_size` of converted svg is `10000`
* Compared with v0.2.1, the conversion effect is sacrificed, especially the depiction of details.

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
# install prerequisites firstly
pip install -q opencv-python scikit-image pillow
pip install scikit-build-core cmake ninja pybind11
# then install this lib
pip install git+https://github.com/nicewang/bitmap2svg.git

```
## Appendix
### A. Defects
* **🔷<span style="color:blue;">Color Distortion</spen>** of converted SVG, especially when `adaptive color quantization`
* **🔷<span style="color:blue;">Blurred Borders</span>** of converted SVG

### B. Design Draft
```
Color Quantization:
    Add Blobs
Contours Detection and Simplification:
    Add Shapes
?:
    Add Details
Features (Polygons) Importance Sorting and Filtering:
    Denoising
```

## Historical Versions
- CPU Version:
    * [v0.2.3](https://github.com/nicewang/bitmap2svg/tree/bitmap2svg-v0.2.3)
    * [v0.2.2](https://github.com/nicewang/bitmap2svg/tree/v0.2.2-bitmap2svg) (Tagged)
    * [v0.2.1](https://github.com/nicewang/bitmap2svg/tree/v0.2.1-bitmap2svg) (Tagged)
    * [v0.1.0](https://github.com/nicewang/bitmap2svg/tree/v0.1.0-bitmap2svg) (Tagged)
- GPU Verison:
    * [v0.2.3.2](https://github.com/nicewang/bitmap2svg/tree/bitmap2svg-cu-v0.2.3.2)
    * [v0.2.3.3](https://github.com/nicewang/bitmap2svg/tree/bitmap2svg-cu-v0.2.3.3) (Toy Version)

## Citation
```BibTex
@misc{bitmap2svg,
  author       = {Xiaonan (Nice) Wang},
  title        = {Bitmap2SVG: A library to convert bitmaps to SVG using C++ and OpenCV},
  howpublished = {\url{https://github.com/nicewang/bitmap2svg}},
  note         = {Accessed: 2026-02-27},
  year         = {2026},
  doi          = {10.5281/zenodo.18799889}
}
```
