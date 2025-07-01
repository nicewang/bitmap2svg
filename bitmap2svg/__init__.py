import numpy as np
from PIL import Image
import sys

def bitmap_to_svg(
    image: Image.Image,
    num_colors: int | None = None, # User specifies desired colors, None for adaptive.
    resize: bool = True,           # Whether to resize the image before processing.
    target_size: tuple[int, int] = (384, 384), # Target (width, height) for resizing.
    simplification_epsilon_factor: float = 0.009, # Controls polygon simplification.
    min_contour_area: float = 10.0,              # Minimum area for a polygon to be included.
    max_features_to_render: int = 0,              # Max number of polygons (0 for unlimited).
    use_processed_mask = False,         # Whether to use processed mask before finding contours.
    adaptive_epsilon = False,           # Whether to use adaptive calculated epsilon in approxPolyDP.
    use_smooth_paths: bool = False      # Whether to use simple Bezier curve output instead of polygon output.
) -> str:
    """
    Converts a PIL Image object to an SVG string using the C++ backend.

    The C++ backend handles color quantization (including adaptive color count
    if num_colors is None or <=0), contour detection, polygon simplification,
    and SVG generation.

    Args:
        image: The input PIL.Image.Image object.
        num_colors: The desired number of dominant colors in the SVG.
                    If None or <=0, the C++ backend will adaptively choose
                    the number of colors.
        resize: If True, the image will be resized to `target_size` before
                conversion. This can significantly affect processing time and
                output complexity.
        target_size: A tuple (width, height) for resizing if `resize` is True.
        simplification_epsilon_factor: Controls polygon simplification.
                                       Higher values mean more simplification and
                                       fewer points in polygons. Corresponds to
                                       the epsilon factor used in Douglas-Peucker.
                                       Default: 0.009.
        min_contour_area: The minimum area (in pixels^2 of the processed image)
                          for a detected contour to be included in the SVG.
                          Helps filter out noise or very small details.
                          Default: 10.0.
        max_features_to_render: The maximum number of distinct colored polygons
                                (features) to render in the SVG. If 0, all
                                features passing other filters will be rendered.
                                Features are typically sorted by area descending,
                                so this effectively renders the largest N features.
                                Default: 0 (unlimited).
        use_processed_mask: Whether to use processed mask before finding contours.
        adaptive_epsilon: Whether to use adaptive calculated epsilon in approxPolyDP.
        use_smooth_paths: Whether to use simple Bezier curve output instead of polygon output.

    Returns:
        A string containing the SVG code.

    Raises:
        TypeError: If the input image is not a PIL.Image.Image instance.
        ValueError: If target_size is not valid.
        ImportError: If the C++ core module cannot be imported.
        RuntimeError: If an error occurs during C++ processing.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input 'image' must be a PIL.Image.Image instance.")

    if not (isinstance(target_size, tuple) and len(target_size) == 2 and
            all(isinstance(dim, int) and dim > 0 for dim in target_size)):
        raise ValueError("target_size must be a tuple of two positive integers (width, height).")

    original_pil_width, original_pil_height = image.size

    # Work on a copy of the image to avoid modifying the original PIL Image object.
    processed_image = image.copy()

    if resize:
        # print(f"Resizing image from {processed_image.size} to {target_size}", file=sys.stderr)
        processed_image = processed_image.resize(target_size, Image.LANCZOS) # LANCZOS for high quality resize

    # Ensure the image is in RGB format, as C++ expects 3 channels.
    if processed_image.mode != 'RGB':
        processed_image = processed_image.convert('RGB')

    # Convert the (potentially resized) PIL Image to a NumPy array.
    # The C++ backend expects raw pixel data.
    # Ensure it's C-contiguous for pybind11.
    img_np_raw = np.array(processed_image, dtype=np.uint8)
    if not img_np_raw.flags['C_CONTIGUOUS']:
        img_np_raw = np.ascontiguousarray(img_np_raw) # Make it C-contiguous if not already

    # Determine the num_colors_hint for the C++ function.
    # Pass 0 or a negative value to C++ to trigger its adaptive color selection logic.
    num_colors_for_cpp_hint = 0
    if num_colors is not None:
        if num_colors > 0:
            num_colors_for_cpp_hint = num_colors
        # If num_colors is 0 or negative, it will also trigger adaptive in C++
        # (or can explicitly set it to 0 if num_colors is None).

    # Call the C++ core function.
    try:

        from . import bitmap2svg_core 

        svg_code = bitmap2svg_core.convert_bitmap_to_svg_cpp(
            raw_bitmap_data_rgb=img_np_raw, # This numpy array contains the processed image data & dimensions
            # width=processed_width, # Pass if C++ binding needs explicit processed width
            # height=processed_height, # Pass if C++ binding needs explicit processed height
            num_colors_hint=num_colors_for_cpp_hint,
            simplification_epsilon_factor=simplification_epsilon_factor,
            min_contour_area=min_contour_area,
            max_features_to_render=max_features_to_render,
            use_processed_mask=use_processed_mask,
            adaptive_epsilon=adaptive_epsilon,
            use_smooth_paths=use_smooth_paths,
            original_width_py=original_pil_width, # For SVG's width attribute
            original_height_py=original_pil_height # For SVG's height attribute
        )
    except ImportError as e:
        print(
            f"Failed to import bitmap2svg_core: {e}\n"
            "Please ensure the C++ module is compiled correctly and is in the PYTHONPATH "
            "or the current working directory structure (e.g., part of the 'bitmap2svg' package).",
            file=sys.stderr
        )
        # Re-raise the exception so the user knows the import failed critically.
        raise
    except Exception as e:
        # Catch potential errors from the C++ extension (e.g., py::value_error)
        # or other runtime issues.
        print(f"Error during C++ SVG conversion: {e}", file=sys.stderr)
        # Re-raise to signal failure to the caller.
        raise

    return svg_code

# Example usage if this script is run directly.
if __name__ == '__main__':
    print("Running SVG conversion example...")
    try:
        # Attempt to load a test image. Create a dummy one if not found.
        test_image_path = "test.png" # Replace with a path to an actual image file for testing.
        try:
            img = Image.open(test_image_path)
            print(f"Loaded test image '{test_image_path}' with size: {img.size}")
        except FileNotFoundError:
            print(f"Test image '{test_image_path}' not found. Creating a dummy image for testing.", file=sys.stderr)
            # Create a simple 100x150 dummy image with a few colored rectangles.
            dummy_array = np.zeros((100, 150, 3), dtype=np.uint8)
            dummy_array[10:40, 10:50, 0] = 200  # Reddish rectangle
            dummy_array[10:40, 60:100, 1] = 200 # Greenish rectangle
            dummy_array[50:90, 10:50, 2] = 200  # Bluish rectangle
            dummy_array[50:90, 60:100, :] = 150 # Grayish rectangle
            dummy_array[0:5, :, 0] = 255 # Red border top
            dummy_array[:, 0:5, 1] = 255 # Green border left
            img = Image.fromarray(dummy_array, 'RGB')
            print(f"Created dummy image with size: {img.size}")

        # --- Test Case 1: Adaptive colors, with resizing, default optimization ---
        print("\nTest Case 1: Adaptive colors, resize to (200, 150), default opts")
        svg_adaptive_resized = bitmap_to_svg(
            img,
            num_colors=None,
            resize=True,
            target_size=(200, 150)
        )
        output_path_1 = "output_cpp_adaptive_resized_default_opts.svg"
        with open(output_path_1, "w", encoding="utf-8") as f:
            f.write(svg_adaptive_resized)
        print(f"SVG (adaptive, resized, default opts) saved to: {output_path_1}")

        # --- Test Case 2: Fixed number of colors (e.g., 5), no resizing, default optimization ---
        print("\nTest Case 2: 5 colors, no resize, default opts")
        svg_fixed_no_resize = bitmap_to_svg(
            img,
            num_colors=5,
            resize=False
        )
        output_path_2 = "output_cpp_5colors_no_resize_default_opts.svg"
        with open(output_path_2, "w", encoding="utf-8") as f:
            f.write(svg_fixed_no_resize)
        print(f"SVG (5 colors, no resize, default opts) saved to: {output_path_2}")

        # --- Test Case 3: More colors, with resizing, high simplification, specific min_area ---
        print("\nTest Case 3: 10 colors, resize (100,100), high simplification (0.03), min_area=5")
        svg_10_colors_custom_opts = bitmap_to_svg(
            img,
            num_colors=10,
            resize=True,
            target_size=(100,100),
            simplification_epsilon_factor=0.03, # Higher simplification
            min_contour_area=5.0,               # Explicitly set different from default
            max_features_to_render=50           # Limit to 50 features
        )
        output_path_3 = "output_cpp_10colors_custom_opts.svg"
        with open(output_path_3, "w", encoding="utf-8") as f:
            f.write(svg_10_colors_custom_opts)
        print(f"SVG (10 colors, custom opts) saved to: {output_path_3}")

        # --- Test Case 4: No resize, default colors, very low simplification (more detail) ---
        print("\nTest Case 4: No resize, default colors, very low simplification (0.005)")
        svg_low_simplification = bitmap_to_svg(
            img,
            resize=False,
            simplification_epsilon_factor=0.005, # Lower simplification (more detail)
            min_contour_area=5.0,                # Explicitly set, very small minimum area
        )
        output_path_4 = "output_cpp_low_simplification.svg"
        with open(output_path_4, "w", encoding="utf-8") as f:
            f.write(svg_low_simplification)
        print(f"SVG (low simplification) saved to: {output_path_4}")

        # --- Test Case 5: Limit features ---
        print("\nTest Case 5: Adaptive colors, no resize, limit to 3 features, default opts")
        svg_limited_features = bitmap_to_svg(
            img,
            num_colors=None, # Adaptive
            resize=False,
            max_features_to_render=3 # Only render the 3 "most important" features
        )
        output_path_5 = "output_cpp_limited_features.svg"
        with open(output_path_5, "w", encoding="utf-8") as f:
            f.write(svg_limited_features)
        print(f"SVG (limited to 3 features) saved to: {output_path_5}")


        print("\nAll examples finished. Check the .svg files.")

    except ImportError:
        # This was already caught above, but good to have a catch-all here too for the example.
        print("ImportError: bitmap2svg_core module not found. Ensure it's compiled and accessible.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred in the example usage: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
