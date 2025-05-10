## [bitmap2svg](https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/bitmap2svg?tab=readme-ov-file)
### [v0.1.0](https://github.com/Opensource-Nice-Arishi/kaggle_drawing_with_LLMs/tree/v0.1.0-bitmap2svg?tab=readme-ov-file)
* [Demo Code](v0.1.0/quick_start_demo.py):

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