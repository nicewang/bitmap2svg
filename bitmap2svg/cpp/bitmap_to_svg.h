#ifndef BITMAP_TO_SVG_H
#define BITMAP_TO_SVG_H

#include <string>
#include <vector>
#include <opencv2/core/types.hpp> // For cv::Point

// Define a simple Color structure for palette colors (R, G, B)
// This structure is used to represent colors in the generated palette.
struct Color {
    unsigned char r; // Red channel
    unsigned char g; // Green channel
    unsigned char b; // Blue channel
};

// Structure to hold extracted SVG feature data before sorting and rendering.
// Each SvgFeature typically corresponds to a colored polygon in the final SVG.
struct SvgFeature {
    std::vector<cv::Point> points; // Simplified contour points (from cv::approxPolyDP) defining the polygon.
    std::string color_hex;         // Compressed hex color string (e.g., "#RRGGBB" or "#RGB") for the fill.
    double area;                   // Area of the original contour, used for importance calculation.
    double importance;             // Calculated importance score for sorting features (larger is more important).

    // Custom comparator to sort features by importance in descending order.
    // This allows rendering more important features first or selectively.
    bool operator<(const SvgFeature& other) const {
        return importance > other.importance;
    }
};

/**
 * @brief Converts a raw bitmap image to an SVG string with internal color quantization.
 *
 * This function handles the entire process:
 * 1. Adaptive color quantization using k-means. This step can be GPU-accelerated
 * (e.g., using FAISS) if the library is compiled with appropriate support and
 * a compatible GPU is available at runtime; otherwise, it defaults to a CPU implementation.
 * 2. Contour detection for each dominant color in the quantized image.
 * 3. Polygon simplification of detected contours using cv::approxPolyDP.
 * 4. Calculation of a heuristic "importance" score for each feature (based on area, centrality, complexity).
 * 5. Sorting features by their calculated importance.
 * 6. Generation of an SVG string, potentially limiting the number of features or total SVG size.
 *
 * @param raw_bitmap_data_rgb_ptr Pointer to the raw pixel data of the input image.
 * The data is expected to be in RGB order, tightly packed (i.e., an array of
 * height * width * 3 bytes).
 * @param width Width of the input image in pixels. Must be greater than 0.
 * @param height Height of the input image in pixels. Must be greater than 0.
 * @param num_colors_hint Desired number of dominant colors for the quantization step.
 * If this value is less than or equal to 0, an adaptive number of colors will be
 * chosen automatically based on the image size (typically between 6 and 12).
 * Otherwise, the specified number of colors (up to a reasonable limit like 256)
 * will be targeted. The actual number of colors in the output SVG may be less
 * if k-means converges to fewer distinct centers or if some colors result in
 * no significant contours.
 * @param simplification_epsilon_factor Factor to determine the epsilon for cv::approxPolyDP,
 * relative to the contour's arc length. A smaller value (e.g., 0.001) means
 * less simplification (more points, more detail), while a larger value (e.g., 0.02)
 * means more simplification (fewer points, less detail). Default: 0.009.
 * @param min_contour_area Minimum area (in pixels squared) for a contour to be considered
 * significant enough to be rendered as an SVG feature. Contours smaller than this
 * will be ignored. Default: 10.0.
 * @param max_features_to_render Maximum number of polygon features to render in the SVG.
 * Features are rendered in order of decreasing importance. If this value is
 * less than or equal to 0, all features that meet other criteria (like min_contour_area
 * and SVG size limits) will be attempted to be rendered. Default: 0 (no explicit limit).
 * @param original_svg_width Desired width to be set in the SVG's root `width` attribute.
 * If less than or equal to 0, the processed (quantized) image width is used.
 * This affects the display size of the SVG but not its internal coordinate system
 * defined by the `viewBox`. Default: -1.
 * @param original_svg_height Desired height to be set in the SVG's root `height` attribute.
 * If less than or equal to 0, the processed (quantized) image height is used.
 * This affects the display size of the SVG but not its internal coordinate system
 * defined by the `viewBox`. Default: -1.
 * @return A string containing the generated SVG code. If an error occurs (e.g., invalid
 * dimensions, quantization failure), an SVG string containing an error message
 * may be returned.
 */
std::string bitmapToSvg_with_internal_quantization(
    const unsigned char* raw_bitmap_data_rgb_ptr,
    int width,
    int height,
    int num_colors_hint,
    double simplification_epsilon_factor = 0.009,
    double min_contour_area = 10.0,
    int max_features_to_render = 0,
    int original_svg_width = -1,
    int original_svg_height = -1
);

#endif // BITMAP_TO_SVG_H
