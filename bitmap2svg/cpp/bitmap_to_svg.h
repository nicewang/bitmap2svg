#ifndef BITMAP_TO_SVG_H
#define BITMAP_TO_SVG_H

#include <string>
#include <vector>
#include <opencv2/core/types.hpp> // For cv::Point

// Define a simple Color structure for palette colors (R, G, B)
struct Color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// Structure to hold extracted SVG feature data before sorting and rendering
struct SvgFeature {
    std::vector<cv::Point> points; // Simplified contour points (from cv::approxPolyDP)
    std::string color_hex;         // Compressed hex color string (e.g., "#RRGGBB" or "#RGB")
    double area;                   // Area of the contour
    double importance;             // Calculated importance score for sorting
    // int original_contour_index; // Optional: for debugging or more complex logic

    // Custom comparator to sort features by importance in descending order
    bool operator<(const SvgFeature& other) const {
        return importance > other.importance;
    }
};

/**
 * @brief Converts a raw bitmap image to an SVG string with internal color quantization.
 *
 * This function handles the entire process:
 * 1. Optional adaptive color quantization using k-means.
 * 2. Contour detection for each dominant color.
 * 3. Polygon simplification of contours.
 * 4. Calculation of feature importance (area, centrality, complexity).
 * 5. Sorting features by importance.
 * 6. Generation of an SVG string.
 *
 * @param raw_bitmap_data_rgb_ptr Pointer to the raw pixel data of the input image.
 * Expected to be in RGB order, tightly packed (height x width x 3 bytes).
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param num_colors_hint Desired number of colors for quantization.
 * If <= 0, an adaptive number of colors will be chosen based on image size.
 * @param simplification_epsilon_factor Factor to determine the epsilon for cv::approxPolyDP.
 * A smaller value means less simplification. E.g., 0.005.
 * @param original_svg_width Width to be set in the SVG's `width` attribute.
 * If <= 0, the processed image width (after potential resizing by Python) is used.
 * @param original_svg_height Height to be set in the SVG's `height` attribute.
 * If <= 0, the processed image height is used.
 * @return A string containing the generated SVG code.
 */
std::string bitmapToSvg_with_internal_quantization(
    const unsigned char* raw_bitmap_data_rgb_ptr,
    int width,
    int height,
    int num_colors_hint,
    double simplification_epsilon_factor = 0.005,
    int original_svg_width = -1,
    int original_svg_height = -1
);

#endif // BITMAP_TO_SVG_H
