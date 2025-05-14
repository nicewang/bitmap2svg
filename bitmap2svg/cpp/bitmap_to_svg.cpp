#include "bitmap_to_svg.h" // Contains Color, SvgFeature, and function declaration

#include <opencv2/opencv.hpp>     // Main OpenCV header
#include <opencv2/imgproc.hpp>    // For cv::cvtColor, cv::kmeans, cv::inRange, cv::findContours, cv::approxPolyDP, etc.
#include <iostream>
#include <sstream>  // Required for std::stringstream
#include <vector>
#include <algorithm> // For std::sort, std::max, std::min
#include <iomanip>   // For std::setw, std::setfill for hex formatting

// Helper function to convert RGB to compressed hex (e.g., #RRGGBB to #RGB if possible)
std::string compress_hex_color_cpp(unsigned char r, unsigned char g, unsigned char b) {
    std::stringstream ss;
    ss << "#";
    // Check if shorthand hex can be used (e.g., #112233 -> #123)
    // This occurs if R, G, and B components are all multiples of 17 (0x11)
    // which means their two hex digits are identical (e.g., 0xRR where R is the same digit)
    if (r % 17 == 0 && g % 17 == 0 && b % 17 == 0) {
        ss << std::hex << (r / 17) << (g / 17) << (b / 17);
    } else {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(r)
           << std::setw(2) << std::setfill('0') << static_cast<int>(g)
           << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return ss.str();
}

// Helper function to perform color quantization using OpenCV's k-means
// Input image `input_img_rgb` is expected to be in RGB format (CV_8UC3).
// Output `quantized_image_rgb` will also be in RGB format.
// Output `palette_vector_Color_struct` will contain `Color` structs {r,g,b}.
std::pair<cv::Mat, std::vector<Color>> perform_color_quantization_cpp(
    const cv::Mat& input_img_rgb, // Input raw RGB image (CV_8UC3)
    int& num_colors_actual        // In: desired num_colors or <=0 for adaptive. Out: actual num_colors used.
) {
    if (input_img_rgb.empty() || input_img_rgb.channels() != 3) {
        std::cerr << "Error: Input image for quantization is invalid. Must be 3 channels." << std::endl;
        return {cv::Mat(), {}};
    }

    // Adaptive color selection logic (same as Python version's heuristic)
    if (num_colors_actual <= 0) {
        long pixel_count = static_cast<long>(input_img_rgb.rows) * input_img_rgb.cols;
        if (pixel_count == 0) { // Handle empty image case
             num_colors_actual = 1;
        } else if (pixel_count < 65536) { // e.g., < 256x256
            num_colors_actual = 8;
        } else if (pixel_count < 262144) { // e.g., < 512x512
            num_colors_actual = 12;
        } else {
            num_colors_actual = 16;
        }
    }
    if (num_colors_actual <= 0) num_colors_actual = 1; // Ensure at least one color

    // Prepare data for k-means:
    // Reshape image to N x 3 matrix (N = rows*cols), each row is a pixel.
    // Convert to 32-bit float as k-means expects float samples.
    cv::Mat samples_rgb(input_img_rgb.total(), 3, CV_32F);
    // Assuming input_img_rgb is CV_8UC3 (uchar values from 0-255)
    // We need to convert it to CV_32FC3 first, then reshape, or iterate.
    // A more direct way:
    cv::Mat input_float_rgb;
    input_img_rgb.convertTo(input_float_rgb, CV_32F); // Convert uchar image to float image
    samples_rgb = input_float_rgb.reshape(1, input_img_rgb.total()); // Reshape to N rows, 3 cols (1 channel for reshape)

    cv::Mat labels;      // Output: cluster_idx for each sample (pixel)
    cv::Mat centers_rgb; // Output: cluster centers (palette colors), float, K x 3 (RGB order)
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 100, 0.2);
    
    cv::kmeans(samples_rgb, num_colors_actual, labels, criteria, 10, cv::KMEANS_RANDOM_CENTERS, centers_rgb);

    // Create the quantized image (RGB, CV_8UC3)
    cv::Mat quantized_image_rgb(input_img_rgb.size(), input_img_rgb.type());
    std::vector<Color> palette_vector_Color_struct;
    palette_vector_Color_struct.reserve(centers_rgb.rows);

    for (int i = 0; i < centers_rgb.rows; ++i) {
        palette_vector_Color_struct.push_back({
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(i, 0)))), // R
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(i, 1)))), // G
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(i, 2))))  // B
        });
    }
    
    int* plabels = labels.ptr<int>(0);
    for (int r = 0; r < quantized_image_rgb.rows; ++r) {
        for (int c = 0; c < quantized_image_rgb.cols; ++c) {
            int cluster_idx = plabels[r * quantized_image_rgb.cols + c];
            cv::Vec3b& pixel = quantized_image_rgb.at<cv::Vec3b>(r, c);
            pixel[0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(cluster_idx, 0)))); // R
            pixel[1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(cluster_idx, 1)))); // G
            pixel[2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(cluster_idx, 2)))); // B
        }
    }
    return {quantized_image_rgb, palette_vector_Color_struct};
}

// Main conversion function
std::string bitmapToSvg_with_internal_quantization(
    const unsigned char* raw_bitmap_data_rgb_ptr,
    int width,
    int height,
    int num_colors_hint,
    double simplification_epsilon_factor,
    int original_svg_width,
    int original_svg_height
) {
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Invalid image dimensions (width=" << width << ", height=" << height << ")." << std::endl;
        std::stringstream error_svg;
        error_svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << std::max(0,width) 
                  << "\" height=\"" << std::max(0,height) 
                  << "\"><text x=\"10\" y=\"20\" fill=\"red\">Error: Invalid image dimensions</text></svg>";
        return error_svg.str();
    }
    // Create an OpenCV Mat from the raw RGB data pointer.
    // CV_8UC3 means 8-bit unsigned char, 3 channels.
    // The data is COPIED here if raw_img_rgb is modified, or if the Mat header is just a wrapper,
    // ensure raw_bitmap_data_rgb_ptr lifetime is sufficient.
    // For safety, if modifications happen to raw_img_rgb, it's better to clone or ensure data ownership.
    // cv::Mat raw_img_rgb(height, width, CV_8UC3, const_cast<unsigned char*>(raw_bitmap_data_rgb_ptr));
    // To ensure data is copied and owned by raw_img_rgb if raw_bitmap_data_rgb_ptr might become invalid:
    cv::Mat temp_mat(height, width, CV_8UC3, const_cast<unsigned char*>(raw_bitmap_data_rgb_ptr));
    cv::Mat raw_img_rgb = temp_mat.clone(); // Ensure data is copied and owned by raw_img_rgb


    if (raw_img_rgb.empty()) {
        std::cerr << "Error: Raw image data is empty or Mat construction failed." << std::endl;
        std::stringstream error_svg;
        error_svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width 
                  << "\" height=\"" << height 
                  << "\"><text x=\"10\" y=\"20\" fill=\"red\">Error: Could not load image data</text></svg>";
        return error_svg.str();
    }
    // IMPORTANT: We assume raw_bitmap_data_rgb_ptr points to data in RGB order.
    // cv::Mat constructed this way will have its channels interpreted according to this assumption
    // when accessed directly (e.g., .at<Vec3b>(y,x)[0] for R).

    int actual_num_colors = num_colors_hint;
    // Pass raw_img_rgb directly, as perform_color_quantization_cpp expects it.
    // If perform_color_quantization_cpp modifies its input, and raw_img_rgb is needed later in its original state,
    // then a clone should be passed: quantization_result = perform_color_quantization_cpp(raw_img_rgb.clone(), actual_num_colors);
    auto quantization_result = perform_color_quantization_cpp(raw_img_rgb, actual_num_colors);
    
    cv::Mat quantized_img_rgb = quantization_result.first;
    std::vector<Color> palette = quantization_result.second;

    if (quantized_img_rgb.empty() || palette.empty()) {
        std::cerr << "Error: Color quantization failed." << std::endl;
        // FIX: Use std::stringstream to build the error SVG string
        std::stringstream error_svg;
        error_svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width 
                  << "\" height=\"" << height 
                  << "\"><text x=\"10\" y=\"20\" fill=\"red\">Error: Color quantization failed</text></svg>";
        return error_svg.str();
    }

    std::vector<SvgFeature> all_features;
    // Image center for importance calculation (using processed dimensions from quantized_img_rgb)
    cv::Point2f image_center(static_cast<float>(quantized_img_rgb.cols) / 2.0f, static_cast<float>(quantized_img_rgb.rows) / 2.0f);
    double max_dist_from_center = cv::norm(cv::Point2f(0,0) - image_center); // Max distance from center to a corner

    for (const auto& pal_color_struct : palette) { // pal_color_struct is {r,g,b}
        // Create a mask for the current palette color.
        // quantized_img_rgb is RGB, so target color for inRange should be RGB.
        cv::Vec3b target_cv_color_rgb(pal_color_struct.r, pal_color_struct.g, pal_color_struct.b);
        cv::Mat mask;
        cv::inRange(quantized_img_rgb, target_cv_color_rgb, target_cv_color_rgb, mask);

        if (cv::countNonZero(mask) == 0) {
            continue; // Skip if this color is not present
        }

        std::vector<std::vector<cv::Point>> contours;
        // cv::findContours modifies the input mask, so pass a clone if mask is needed later.
        cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::string hex_color_str = compress_hex_color_cpp(pal_color_struct.r, pal_color_struct.g, pal_color_struct.b);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < 20) continue; // Skip tiny contours (consistent with original Python)

            std::vector<cv::Point> simplified_contour;
            double epsilon = simplification_epsilon_factor * cv::arcLength(contour, true);
            cv::approxPolyDP(contour, simplified_contour, epsilon, true);
            if (simplified_contour.size() < 3) continue; // Need at least 3 points for a polygon

            cv::Moments M = cv::moments(simplified_contour);
            cv::Point2f contour_center(0.0f, 0.0f);
            if (M.m00 != 0) { // m00 is the area, avoid division by zero
                contour_center.x = static_cast<float>(M.m10 / M.m00);
                contour_center.y = static_cast<float>(M.m01 / M.m00);
            }
            
            double dist_from_img_center = cv::norm(contour_center - image_center);
            double normalized_dist = (max_dist_from_center > 0.001) ? (dist_from_img_center / max_dist_from_center) : 0.0;
            
            // Importance calculation (area * centrality * simplicity_bonus)
            // Centrality: (1 - normalized_dist) -> closer to center is more important
            // Simplicity bonus: (1 / (num_points + 1)) -> fewer points (simpler) is slightly better
            double importance = area * (1.0 - normalized_dist) * (1.0 / (simplified_contour.size() + 1.0));

            all_features.push_back({simplified_contour, hex_color_str, area, importance});
        }
    }

    std::sort(all_features.begin(), all_features.end()); // Sorts by importance (descending)

    // --- SVG Generation ---
    std::stringstream svg_ss;
    int svg_attr_width = (original_svg_width > 0) ? original_svg_width : quantized_img_rgb.cols;
    int svg_attr_height = (original_svg_height > 0) ? original_svg_height : quantized_img_rgb.rows;

    svg_ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_attr_width
           << "\" height=\"" << svg_attr_height
           << "\" viewBox=\"0 0 " << quantized_img_rgb.cols << " " << quantized_img_rgb.rows << "\">\n";
    
    // Background: Use average color of the original (pre-quantization) image.
    // raw_img_rgb is the image passed to quantization (RGB).
    // cv::mean on an RGB Mat will result in a cv::Scalar where [0]=R, [1]=G, [2]=B.
    cv::Scalar avg_color_rgb_scalar = cv::mean(raw_img_rgb);
    std::string bg_hex_color = compress_hex_color_cpp(
        static_cast<unsigned char>(avg_color_rgb_scalar[0]), // R
        static_cast<unsigned char>(avg_color_rgb_scalar[1]), // G
        static_cast<unsigned char>(avg_color_rgb_scalar[2])  // B
    );
    svg_ss << "<rect width=\"" << quantized_img_rgb.cols << "\" height=\"" << quantized_img_rgb.rows << "\" fill=\"" << bg_hex_color << "\"/>\n";

    for (const auto& feature : all_features) {
        if (feature.points.empty()) continue;
        svg_ss << "<polygon points=\"";
        for (size_t i = 0; i < feature.points.size(); ++i) {
            svg_ss << feature.points[i].x << "," << feature.points[i].y;
            if (i < feature.points.size() - 1) {
                svg_ss << " ";
            }
        }
        svg_ss << "\" fill=\"" << feature.color_hex << "\"/>\n";
    }

    svg_ss << "</svg>";
    return svg_ss.str();
}
