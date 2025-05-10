## Version:0.1.0
### 1. Introduction

Convert a grayscale bitmap (2D numpy array) to SVG string
### 2. Package Structure
```
bitmap2svg/
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
cd ${project_workspace}/bitmap2svg
python setup.py build
python setup.py install
```
* Way 2:

```bash
cd ${project_workspace}/bitmap2svg
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
pip install "git+https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs.git@main#subdirectory=nice/libs/bitmap2svg"

```
### 4. Quick Start
* Demo Code:

```Python
import numpy as np
import bitmap2svg
from IPython.display import SVG, display
import matplotlib.pyplot as plt

# Create a larger canvas (50x50 for better resolution)
img = np.zeros((50, 50), dtype=np.uint8)

# Create a rose curve
def create_rose(img, n=5, d=4):
    center_x, center_y = 25, 25
    scale = 15
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # Convert to polar coordinates
            dx = (x - center_x) / scale
            dy = (y - center_y) / scale
            r = np.sqrt(dx*dx + dy*dy)
            theta = np.arctan2(dy, dx)
            
            # Rose curve equation: r = cos(n*theta/d)
            if abs(r - abs(np.cos(n*theta/d))) < 0.1:
                img[y, x] = 255

# Clear the image and create new shape
img.fill(0)
create_rose(img)

# Convert and display as SVG
svg_code = bitmap2svg.convert(img)
display(SVG(svg_code))

# Also show using matplotlib
plt.figure(figsize=(5,5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
```