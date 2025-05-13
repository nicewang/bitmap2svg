import ctypes
import os
import sys
from typing import Union, Optional

# Determine the shared library name based on the platform
if sys.platform == 'win32':
    _library_name = 'bitmap_to_svg.dll'
elif sys.platform == 'darwin':
    _library_name = 'libbitmap_to_svg.dylib'
else: # Linux and others
    _library_name = 'libbitmap_to_svg.so'

_library = None

def _load_library():
    """Loads the shared library, searching in the package directory."""
    global _library
    if _library is not None:
        return _library

    # Search path: first in the directory of this file (the installed package dir)
    package_dir = os.path.dirname(__file__)
    library_path = os.path.join(package_dir, _library_name)

    if not os.path.exists(library_path):
        # If not found, try loading from default system paths (LD_LIBRARY_PATH, PATH, etc.)
        # This might happen during development or if the library is installed globally.
        print(f"Warning: C++ library not found at {library_path}. Trying default system paths.")
        try:
            _library = ctypes.CDLL(_library_name)
        except OSError as e:
            raise FileNotFoundError(f"Could not find or load the C++ library: {_library_name}. Looked in {package_dir} and system paths.") from e
    else:
         try:
             _library = ctypes.CDLL(library_path)
         except OSError as e:
            raise OSError(f"Could not load the C++ library from {library_path}: {e}") from e


    # Define the function signature for ctypes
    # const char* convert_image_to_svg(int width, int height, const uint8_t* pixels, int channels);
    try:
        _library.convert_image_to_svg.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int
        ]
        _library.convert_image_to_image.restype = ctypes.c_char_p
    except AttributeError:
         raise AttributeError(f"Could not find the 'convert_image_to_svg' function in the loaded library '{_library_name}'. Ensure the function is exported correctly.")


    return _library

def convert_image_to_svg(width: int, height: int, pixels: bytes, channels: int) -> str:
    """
    Converts bitmap pixel data to SVG using the C++ library.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        pixels: Raw pixel data as a bytes object (e.g., from image.tobytes()).
                Format depends on channels (e.g., 'L', 'RGB', 'RGBA').
        channels: Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA).

    Returns:
        A string containing the SVG XML code.

    Raises:
        FileNotFoundError: If the C++ library cannot be found.
        OSError: If the C++ library fails to load or execute.
        ValueError: If input dimensions/channels or pixel data size are invalid.
        AttributeError: If the expected C++ function is not found in the library.
    """
    if width <= 0 or height <= 0 or channels <= 0:
         raise ValueError("Width, height, and channels must be positive integers.")
    expected_size = width * height * channels
    if len(pixels) != expected_size:
        raise ValueError(f"Expected {expected_size} bytes of pixel data, but got {len(pixels)}.")

    lib = _load_library()

    # Convert Python bytes to ctypes pointer
    # ctypes.c_uint8 is equivalent to uint8_t
    # pixels is a bytes object, which is contiguous in memory.
    # Use addressof to get the memory address, then cast to the desired pointer type.
    pixels_ptr = ctypes.cast(ctypes.addressof(ctypes.create_string_buffer(pixels)), ctypes.POINTER(ctypes.c_uint8))

    # Call the C++ function
    # The C++ function returns const char*, which ctypes maps to c_char_p
    # c_char_p is a pointer to a null-terminated byte string.
    # We need to decode it to a Python string.
    try:
         svg_cstr = lib.convert_image_to_svg(width, height, pixels_ptr, channels)
    except Exception as e:
         raise OSError(f"Error during C++ function call: {e}") from e


    # Decode the returned C-style string (bytes) to a Python string
    # The C++ code uses a static string, so we don't need to free memory from Python.
    if svg_cstr:
         return svg_cstr.decode('utf-8')
    else:
         # C++ returns "<svg></svg>" on error, or potentially NULL.
         # Handle NULL return explicitly if needed, but current C++ returns string.
         return ""

# Optional: Add docstrings and type hints for better usability
convert_image_to_svg.__doc__ = """
Converts bitmap pixel data (bytes) to an SVG string.

Args:
    width: Image width.
    height: Image height.
    pixels: Raw pixel data as a bytes object (e.g., from PIL Image.tobytes()).
    channels: Number of channels (1=L, 3=RGB, 4=RGBA).

Returns:
    An SVG XML string.
"""

# Example Usage (for testing or demonstration)
if __name__ == '__main__':
    # Create dummy pixel data (e.g., a 4x4 image with a black square on white)
    # Grayscale (L), 1 channel
    dummy_width, dummy_height, dummy_channels = 4, 4, 1
    dummy_pixels_list = [
        255, 255, 255, 255,
        255,   0,   0, 255,
        255,   0,   0, 255,
        255, 255, 255, 255,
    ]
    dummy_pixels_bytes = bytes(dummy_pixels_list)

    try:
        svg_output = convert_image_to_svg(dummy_width, dummy_height, dummy_pixels_bytes, dummy_channels)
        print("Generated SVG (Grayscale):")
        print(svg_output)

        # Example with color (RGB)
        color_width, color_height, color_channels = 3, 3, 3
        color_pixels_list = [
             255, 0, 0,   0, 255, 0,   0, 0, 255, # Red, Green, Blue
             255, 0, 0,   0, 255, 0,   0, 0, 255,
             255, 0, 0,   0, 255, 0,   0, 0, 255,
        ]
        color_pixels_bytes = bytes(color_pixels_list)
        svg_output_color = convert_image_to_svg(color_width, color_height, color_pixels_bytes, color_channels)
        print("\nGenerated SVG (Color - might trace individual colors):")
        print(svg_output_color)

    except Exception as e:
        print(f"Error during conversion: {e}")
