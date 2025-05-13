import numpy as np
from PIL import Image
import cv2 # Using OpenCV for color quantization
import io
import sys 

# Import the compiled C++ module
try:
    print("Attempting to import bitmap2svg_core module...", file=sys.stderr)
    import bitmap2svg_core
    print("Successfully imported bitmap2svg_core module.", file=sys.stderr)

except ImportError as e:
    print(f"Failed to import bitmap2svg_core: {e}", file=sys.stderr)
    # Re-raise the exception so the original error is still visible
    raise

def compress_hex_color(hex_color):
    """Convert hex color to shortest possible representation"""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    return hex_color

def bitmap_to_svg(image: Image.Image, num_colors: int = 16) -> str:
    """
    Convert a PIL Image to SVG using the C++ backend.

    Args:
        image: Input PIL.Image.Image object.
        num_colors: Number of colors to quantize the image to.

    Returns:
        A string containing the SVG code.
    """
    print("Python bitmap_to_svg called.", file=sys.stderr)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_np = np.array(image)

    # Perform color quantization using OpenCV
    pixels = img_np.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    # Create the palette for the C++ code
    palette = []
    for color in centers:
        # Create C++ Color objects using the bound class
        palette.append(bitmap2svg_core.Color(color[0], color[1], color[2]))

    # Call the C++ convert function
    try:
        print("Calling C++ convert function...", file=sys.stderr)
        # Pass the NumPy array and the palette to the C++ function
        svg_code = bitmap2svg_core.convert(img_np, palette)
        print("C++ convert function returned.", file=sys.stderr)
    except Exception as e:
        print(f"Error calling C++ convert function: {e}", file=sys.stderr)
        raise # Re-raise the exception

    print("Python bitmap_to_svg finished.", file=sys.stderr)
    return svg_code
