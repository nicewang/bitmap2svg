import numpy as np  # Import the NumPy library for efficient numerical computations

from PIL import Image  # Import the PIL (Pillow) library for image processing

import cv2  # Import the OpenCV library for computer vision tasks

def compress_hex_color(hex_color):
    """
    Convert a hexadecimal color code to its shortest possible representation.

    For example, convert #FFCC00 to #FC0.

    Args:
        hex_color (str): The hexadecimal color code, e.g., "#RRGGBB".

    Returns:
        str: The compressed hexadecimal color code if possible, otherwise the original color code.
    """
    # Extract the red, green, and blue components from the hexadecimal color code.
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    # Check if each color component is a multiple of 17.
    if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
        # If so, divide each component by 17 and convert the result to a single-digit hexadecimal character.
        # For example, if r = 255 (FF), r // 17 = 15 (F).
        return f'#{r//17:x}{g//17:x}{b//17:x}'  # Use an f-string to format the string
    return hex_color  # Otherwise, return the original hexadecimal color code.

def extract_features_by_scale(img_np, num_colors=16):
    """
    Extract image features hierarchically by scale.

    This function segments the image into representative color regions and extracts contours
    and other features for each region. It also sorts the features by the importance of the
    colors in the image.

    Args:
        img_np (np.ndarray): The input image, represented as a NumPy array.
        num_colors (int, optional): The number of colors to quantize to. Defaults to 16.

    Returns:
        list: A list of hierarchical features, sorted by importance. Each feature is a dictionary
              containing 'points' (contour point string), 'color' (hex color), 'area' (region area),
              'importance' (importance score), 'point_count' (number of points in contour),
              and 'original_contour' (original contour, for adaptive simplification).
    """
    # Ensure the input image is in RGB format.
    if len(img_np.shape) == 3 and img_np.shape[2] > 1:
        img_rgb = img_np  # If it's already RGB, use it directly.
    else:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)  # Otherwise, convert it from grayscale to RGB.

    # Convert the RGB image to grayscale for certain processing steps.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape  # Get the height and width of the image.

    # Color quantization: Reduce the number of colors in the image to the specified number.
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)  # Reshape the RGB image into a list of pixels.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # Set the termination criteria for the K-means algorithm.
    # Use OpenCV's K-means algorithm for color quantization.
    try:
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    except Exception as e:
        print(f"Error in cv2.kmeans: {e}")
        print("OpenCV may not be correctly installed.  Please ensure that OpenCV is correctly installed and configured for your environment.  A common issue on Linux is missing the libGL.so.1 dependency.  This can be resolved by installing the libgl1-mesa-glx package (e.g., sudo apt-get install libgl1-mesa-glx).  If you are on a different operating system, please install the equivalent package.  If you are running in a headless environment, you may need to install a virtual display driver such as Xvfb.")
        raise

    # Convert the quantized color centers to unsigned 8-bit integers.
    palette = centers.astype(np.uint8)
    # Create the quantized image using the quantized color centers and labels.
    quantized = palette[labels.flatten()].reshape(img_rgb.shape)

    hierarchical_features = []  # Initialize an empty list to store the hierarchical features.

    # Sort colors by frequency.
    unique_labels, counts = np.unique(labels, return_counts=True)  # Count the occurrences of each color label.
    sorted_indices = np.argsort(-counts)  # Get the indices of the colors sorted in descending order of frequency.
    sorted_colors = [palette[i] for i in sorted_indices]  # Get the sorted list of colors.

    # Calculate the center point of the image, used for importance calculations later.
    center_x, center_y = width / 2, height / 2

    # Iterate over the sorted colors.
    for color in sorted_colors:
        # Create a color mask for the current color.
        color_mask = cv2.inRange(quantized, color, color)

        # Find the contours in the mask.
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by area (largest first).
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Convert the RGB color to a compressed hexadecimal format.
        hex_color = compress_hex_color(f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')

        color_features = []  # Store the features for the current color

        # Iterate over the contours of the current color.
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:  # Skip very small contours.
                continue

            # Calculate the center of the contour.
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue  # Avoid division by zero.
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            # Calculate the distance from the contour center to the image center, and normalize it.
            dist_from_center = np.sqrt(((cx - center_x) / width) ** 2 + ((cy - center_y) / height) ** 2)

            # Simplify the contour, reducing the number of points.
            epsilon = 0.02 * cv2.arcLength(contour, True)  # A smaller epsilon value preserves more detail
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Convert the contour points to a string format.
            points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])

            # Calculate the importance of the contour.
            importance = (
                area * # Contour area
                (1 - dist_from_center) * # Distance from the center (closer is more important)
                (1 / (len(approx) + 1))  # Contour complexity (fewer points is more important)
            )

            color_features.append({
                'points': points,
                'color': hex_color,
                'area': area,
                'importance': importance,
                'point_count': len(approx),
                'original_contour': approx  # Save the original contour for later adaptive simplification
            })

        # Sort the features of the current color by importance.
        color_features.sort(key=lambda x: x['importance'], reverse=True)
        hierarchical_features.extend(color_features)  # Add the current color's features to the total list.

    # Sort all the features by overall importance.
    hierarchical_features.sort(key=lambda x: x['importance'], reverse=True)

    return hierarchical_features  # Return the sorted list of features.

def simplify_polygon(points_str, simplification_level):
    """
    Simplify a polygon by reducing coordinate precision or the number of points.

    Args:
        points_str (str): A space-separated string of "x,y" coordinates.
        simplification_level (int): The simplification level (0-3).

    Returns:
        str: The simplified point string.
    """
    if simplification_level == 0:
        return points_str  # Do not simplify.

    points = points_str.split()  # Split the string into a list of points.

    if simplification_level == 1:
        # Round the coordinates to 1 decimal place.
        return " ".join([f"{float(p.split(',')[0]):.1f},{float(p.split(',')[1]):.1f}" for p in points])

    elif simplification_level == 2:
        # Round the coordinates to the nearest integer.
        return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in points])

    elif simplification_level == 3:
        # Reduce the number of points, but ensure at least 3 points are kept.
        if len(points) <= 4:
            # If there are 4 or fewer points, just round to integer
            return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in points])
        else:
            # Keep approximately half the points, but maintain at least 3
            step = min(2, len(points) // 3)  # Calculate a suitable step size
            reduced_points = [points[i] for i in range(0, len(points), step)]
            # Ensure we keep at least 3 points and the last point
            if len(reduced_points) < 3:
                reduced_points = points[:3]
            if points[-1] not in reduced_points:
                reduced_points.append(points[-1])
            return " ".join([f"{float(p.split(',')[0]):.0f},{float(p.split(',')[1]):.0f}" for p in reduced_points])

    return points_str  # Return the original string if the simplification_level is invalid.

def bitmap_to_svg_layered(image, max_size_bytes=10000, resize=True, target_size=(384, 384),
                         adaptive_fill=True, num_colors=None):
    """
    Convert a bitmap to SVG using layered feature extraction, optimizing space usage.

    This function converts a raster image (bitmap) into a scalable vector graphic (SVG) format.
    It extracts the main features of the image and represents them as polygons in the SVG.
    The function aims to generate the smallest possible SVG file while maintaining visual quality.

    Args:
        image (PIL.Image): The input image (a PIL.Image object).
        max_size_bytes (int, optional): The maximum size of the SVG file in bytes. Defaults to 10000.
        resize (bool, optional): Whether to resize the image before processing. Defaults to True.
        target_size (tuple, optional): The target size for resizing (width, height). Defaults to (384, 384).
        adaptive_fill (bool, optional): Whether to adaptively fill the available space. Defaults to True.
        num_colors (int, optional): The number of colors to quantize to. If None, uses adaptive selection.

    Returns:
        str: The SVG string representation.
    """
    # Adaptive color selection: Choose the number of colors based on image complexity.
    if num_colors is None:
        if resize:
            pixel_count = target_size[0] * target_size[1]
        else:
            pixel_count = image.size[0] * image.size[1]

        if pixel_count < 65536:  # 256x256
            num_colors = 8
        elif pixel_count < 262144:  # 512x512
            num_colors = 12
        else:
            num_colors = 16

    # Resize the image if needed.
    if resize:
        original_size = image.size  # Save the original dimensions
        image = image.resize(target_size, Image.LANCZOS)  # Use a high-quality LANCZOS filter.
    else:
        original_size = image.size
    # Convert the PIL Image to a NumPy array.
    img_np = np.array(image)

    # Get the image dimensions.
    height, width = img_np.shape[:2]

    # Calculate the average background color.
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        avg_bg_color = np.mean(img_np, axis=(0, 1)).astype(int)
        bg_hex_color = compress_hex_color(f'#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}')
    else:
        bg_hex_color = '#fff'  # Default background color is white

    # Start building the SVG string.
    # Use the original dimensions in the viewBox for proper scaling when displayed.
    orig_width, orig_height = original_size
    svg_header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}">\n'
    svg_bg = f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>\n'  # Background rectangle
    svg_base = svg_header + svg_bg
    svg_footer = '</svg>'

    # Calculate the base SVG size.
    base_size = len((svg_base + svg_footer).encode('utf-8'))
    available_bytes = max_size_bytes - base_size  # Calculate the bytes available for adding features.

    # Extract the image features.
    features = extract_features_by_scale(img_np, num_colors=num_colors)

    # If not using adaptive fill, add features until the size limit is reached.
    if not adaptive_fill:
        svg = svg_base
        for feature in features:
            feature_svg = f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
            if len((svg + feature_svg + svg_footer).encode('utf-8')) > max_size_bytes:
                break  # Stop adding features if the limit is exceeded.
            svg += feature_svg
        svg += svg_footer
        return svg

    # Use adaptive fill: Try to add more features by simplifying less important ones.
    feature_sizes = []
    for feature in features:
        feature_sizes.append({
            'original': len(f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level1': len(f'<polygon points="{simplify_polygon(feature["points"], 1)}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level2': len(f'<polygon points="{simplify_polygon(feature["points"], 2)}" fill="{feature["color"]}" />\n'.encode('utf-8')),
            'level3': len(f'<polygon points="{simplify_polygon(feature["points"], 3)}" fill="{feature["color"]}" />\n'.encode('utf-8'))
        })

    svg = svg_base
    bytes_used = base_size
    added_features = set()

    # Add the most important features first
    for i, feature in enumerate(features):
        feature_svg = f'<polygon points="{feature["points"]}" fill="{feature["color"]}" />\n'
        feature_size = feature_sizes[i]['original']

        if bytes_used + feature_size <= max_size_bytes:
            svg += feature_svg
            bytes_used += feature_size
            added_features.add(i)

    # Try to add the remaining features, simplifying them progressively
    for level in range(1, 4):
        for i, feature in enumerate(features):
            if i in added_features:
                continue

            feature_size = feature_sizes[i][f'level{level}']
            if bytes_used + feature_size <= max_size_bytes:
                feature_svg = f'<polygon points="{simplify_polygon(feature["points"], level)}" fill="{feature["color"]}" />\n'
                svg += feature_svg
                bytes_used += feature_size
                added_features.add(i)

    svg += svg_footer

    final_size = len(svg.encode('utf-8'))
    if final_size > max_size_bytes:
        # If the limit is exceeded, return a basic SVG
        return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="{bg_hex_color}"/></svg>'

    utilization = (final_size / max_size_bytes) * 100  # Calculate the space utilization

    return svg  # Return the generated SVG
