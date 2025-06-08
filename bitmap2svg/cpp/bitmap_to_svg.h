#ifndef BITMAP_TO_SVG_H
#define BITMAP_TO_SVG_H

#include <string>
#include <vector>
#include <opencv2/core/types.hpp> // For cv::Point

// Optimization 2: Add error code enumeration
enum class ConversionResult {
    SUCCESS,
    INVALID_INPUT,
    QUANTIZATION_FAILED,
    SIZE_LIMIT_EXCEEDED
};

// Define a simple Color structure for palette colors (R, G, B).
// This structure is used to represent colors in the generated palette.
// Note: Internally, OpenCV might use BGR, but this struct and the
// final SVG colors are consistently RGB.
struct Color {
    unsigned char r; // Red channel
    unsigned char g; // Green channel
    unsigned char b; // Blue channel

    Color() : r(0), g(0), b(0) {}
    Color(unsigned char red, unsigned char green, unsigned char blue) 
        : r(red), g(green), b(blue) {}

};

// Optimization 10: Use more efficient contour importance calculation
struct ContourFeature {
    std::vector<cv::Point> points;
    std::string color_hex;
    double area;
    double importance;

    ContourFeature() : area(0.0), importance(0.0) {}
    
    // Add move constructor
    ContourFeature(std::vector<cv::Point>&& pts, std::string&& color, double a, double imp)
        : points(std::move(pts)), color_hex(std::move(color)), area(a), importance(imp) {}
    
    bool operator<(const ContourFeature& other) const noexcept {
        return importance > other.importance; // Descending order
    }
};

// Structure to hold extracted SVG feature data before sorting and rendering.
// Each SvgFeature typically corresponds to a colored polygon in the final SVG.
struct SvgFeature {
    std::vector<cv::Point> points; // Simplified contour points (from cv::approxPolyDP) defining the polygon.
    std::string color_hex;         // Compressed hex color string (e.g., "#RRGGBB" or "#RGB") for the fill (RGB format).
    double area;                   // Area of the original contour, used for importance calculation.
    double importance;             // Calculated importance score for sorting features (larger is more important).

    // Custom comparator to sort features by importance in descending order.
    // This allows rendering more important features first or selectively.
    bool operator<(const SvgFeature& other) const {
        return importance > other.importance; // Sorts in descending order of importance
    }
};

/**
 * @brief Converts a raw bitmap image to an SVG string with internal color quantization.
 *
 * This function handles the entire process:
 * 1. Adaptive color quantization using k-means. This step can be GPU-accelerated
 * (e.g., using FAISS GPU) if the library is compiled with appropriate support (`WITH_FAISS_GPU` defined)
 * and a compatible GPU is available at runtime; otherwise, it defaults to a CPU implementation (OpenCV k-means).
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
 * Otherwise, the specified number of colors (clamped between 1 and 256)
 * will be targeted. The actual number of colors in the output SVG may be less.
 * @param simplification_epsilon_factor Factor to determine the epsilon for cv::approxPolyDP,
 * relative to the contour's arc length. Default: 0.009.
 * @param min_contour_area Minimum area (in pixels squared) for a contour to be considered
 * significant enough to be rendered. Default: 10.0.
 * @param max_features_to_render Maximum number of polygon features to render in the SVG.
 * Rendered in order of decreasing importance. If <= 0, no explicit limit. Default: 0.
 * @param original_svg_width Desired width for the SVG's root `width` attribute.
 * If <= 0, processed image width is used. Affects display size, not viewBox. Default: -1.
 * @param original_svg_height Desired height for the SVG's root `height` attribute.
 * If <= 0, processed image height is used. Affects display size, not viewBox. Default: -1.
 * @return A string containing the generated SVG code. Returns an error SVG if issues occur.
 */
std::string bitmapToSvg_with_internal_quantization(
    const unsigned char* raw_bitmap_data_rgb_ptr,
    int width,
    int height,
    int num_colors_hint,
    double simplification_epsilon_factor = 0.02,
    double min_contour_area = 10.0,
    int max_features_to_render = 1000,
    int original_svg_width = 0,
    int original_svg_height = 0
);

#endif // BITMAP_TO_SVG_H
