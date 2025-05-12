# bitmap2svg
## Current Version: 0.2.0
### 1. Introduction
Convert a bitmap to SVG string
### 2. Package Structure
```
${project_workspace}/
├── bitmap2svg/
│   ├── __init__.py
│   └── cpp/
│       ├── bitmap_to_svg.cpp
│       ├── bindings.cpp
├── pyproject.toml
├── setup.py
└── README.md
```
### 3. Install
#### 3.1 Local Install
* Way 1:

```bash
cd ${project_workspace}/
python setup.py build
python setup.py install
```
* Way 2:

```bash
cd ${project_workspace}/
pip install --upgrade pybind11 setuptools wheel
pip install -e .
```
* You can also upload it (and then install):

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
``` 
#### 3.2 Remote Install
```bash
pip install git+https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs.git@bitmap2svg

```
## Historical Versions
* [v0.1.0](https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/v0.1.0-bitmap2svg)