#include "bitmap_to_svg.h" // Contains Color, SvgFeature, and function declaration

#include <opencv2/opencv.hpp>    // Main OpenCV header
#include <opencv2/imgproc.hpp>   // For cv::cvtColor, cv::kmeans, cv::inRange, cv::findContours, cv::approxPolyDP, etc.
#include <iostream>
#include <sstream>  // Required for std::stringstream
#include <vector>
#include <algorithm> // For std::sort, std::max, std::min
#include <iomanip>   // For std::setw, std::setfill for hex formatting

// Define the SVG size constraint
const long MAX_SVG_SIZE_BYTES = 10000;
const long SVG_SIZE_SAFETY_MARGIN = 1000; // Buffer to stop before hitting the absolute max

// Helper function to convert RGB to compressed hex (e.g., #RRGGBB to #RGB if possible)
std::string compress_hex_color_cpp(unsigned char r, unsigned char g, unsigned char b) {
    std::stringstream ss;
    ss << "#";
    if (r % 17 == 0 && g % 17 == 0 && b % 17 == 0 &&
        (r / 17) < 16 && (g / 17) < 16 && (b / 17) < 16) { // Ensure single digit hex
        ss << std::hex << (r / 17) << (g / 17) << (b / 17);
    } else {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(r)
           << std::setw(2) << std::setfill('0') << static_cast<int>(g)
           << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return ss.str();
}

// Helper function to perform color quantization using OpenCV's k-means
std::pair<cv::Mat, std::vector<Color>> perform_color_quantization_cpp(
    const cv::Mat& input_img_rgb, // Input raw RGB image (CV_8UC3)
    int& num_colors_target        // In: desired num_colors. Out: actual num_colors used by kmeans.
) {
    if (input_img_rgb.empty() || input_img_rgb.channels() != 3) {
        std::cerr << "Error: Input image for quantization is invalid. Must be 3 channels." << std::endl;
        return {cv::Mat(), {}};
    }

    int k = num_colors_target; // Use the target directly

    if (k <= 0) {
        long pixel_count = static_cast<long>(input_img_rgb.rows) * input_img_rgb.cols;
        if (pixel_count == 0) { // Should be caught by input_img_rgb.empty() earlier
            k = 1;
        } else if (pixel_count < 16384) { // e.g., < 128x128
            k = 4;
        } else if (pixel_count < 65536) { // e.g., < 256x256
            k = 6;
        } else if (pixel_count < 262144) { // e.g., < 512x512
            k = 8;
        } else {
            k = 10;
        }
    }
    k = std::max(1, k); // Ensure at least one color
    num_colors_target = k; // Update the output parameter with the k we are actually using

    cv::Mat samples_rgb(input_img_rgb.total(), 3, CV_32F);
    cv::Mat input_float_rgb;
    input_img_rgb.convertTo(input_float_rgb, CV_32F);
    samples_rgb = input_float_rgb.reshape(1, input_img_rgb.total());

    if (samples_rgb.rows == 0) { // No samples to process
        std::cerr << "Error: No samples to process for k-means (image might be empty or too small after reshape)." << std::endl;
        return {cv::Mat(), {}};
    }
    if (samples_rgb.rows < k) { // k-means requires samples >= k
        // std::cout << "Warning: Number of samples (" << samples_rgb.rows << ") is less than k (" << k << "). Adjusting k." << std::endl;
        k = samples_rgb.rows;
        if (k == 0) return {cv::Mat(), {}}; // No samples, return empty
        num_colors_target = k; // Update with the adjusted k
    }


    cv::Mat labels;
    cv::Mat centers_rgb;
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, 0.1);
    int attempts = (k <=8) ? 3 : 5;

    if (k > 0) { // Ensure k is positive before calling kmeans
        cv::kmeans(samples_rgb, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers_rgb);
    } else { // Should ideally be caught by k = std::max(1,k), but as a safeguard
         std::cerr << "Error: k is zero or negative before calling kmeans (k=" << k << ")." << std::endl;
        return {cv::Mat(), {}}; // Or handle as single color image
    }

    if (centers_rgb.rows == 0) { // kmeans might return 0 centers for very small k or problematic data
        std::cerr << "Warning: kmeans returned 0 centers. Attempting fallback to average color." << std::endl;
         if (input_img_rgb.total() > 0 && !input_img_rgb.empty()) {
            cv::Scalar avg_color_scalar = cv::mean(input_img_rgb); // input_img_rgb is RGB
            cv::Mat single_color_img(input_img_rgb.size(), input_img_rgb.type());
            single_color_img.setTo(cv::Vec3b(static_cast<unsigned char>(avg_color_scalar[0]), // R
                                             static_cast<unsigned char>(avg_color_scalar[1]), // G
                                             static_cast<unsigned char>(avg_color_scalar[2])) // B
                                  );
            std::vector<Color> single_palette = {{
                static_cast<unsigned char>(avg_color_scalar[0]),
                static_cast<unsigned char>(avg_color_scalar[1]),
                static_cast<unsigned char>(avg_color_scalar[2])
            }};
            num_colors_target = 1;
            return {single_color_img, single_palette};
        }
        return {cv::Mat(), {}}; // Still failed
    }

    num_colors_target = centers_rgb.rows; // Actual number of centers found

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

    if (labels.empty() || labels.rows != samples_rgb.rows ) { // labels might be empty or incorrect size if kmeans failed badly
        std::cerr << "Error: kmeans failed to produce valid labels. Using palette average for image." << std::endl;
        // Fallback: create a single-color image using average of the palette (or first color)
        if (!palette_vector_Color_struct.empty()) {
            quantized_image_rgb.setTo(cv::Vec3b(palette_vector_Color_struct[0].r, palette_vector_Color_struct[0].g, palette_vector_Color_struct[0].b));
        } else { // No palette, no image
             return {cv::Mat(), {}};
        }
    } else {
        int* plabels = labels.ptr<int>(0);
        for (int r_idx = 0; r_idx < quantized_image_rgb.rows; ++r_idx) {
            for (int c_idx = 0; c_idx < quantized_image_rgb.cols; ++c_idx) {
                int sample_idx = r_idx * quantized_image_rgb.cols + c_idx;
                // Ensure sample_idx is within bounds of labels array
                if (sample_idx < labels.rows && plabels) {
                     int cluster_idx = plabels[sample_idx];
                     // Ensure cluster_idx is within bounds of centers_rgb matrix
                     if (cluster_idx >=0 && cluster_idx < centers_rgb.rows) {
                        cv::Vec3b& pixel = quantized_image_rgb.at<cv::Vec3b>(r_idx, c_idx);
                        pixel[0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(cluster_idx, 0)))); // R
                        pixel[1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(cluster_idx, 1)))); // G
                        pixel[2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb.at<float>(cluster_idx, 2)))); // B
                     } else {
                        // std::cerr << "Warning: cluster_idx " << cluster_idx << " out of bounds for centers (0-" << centers_rgb.rows-1 << ")" << std::endl;
                        // Optionally set to a default color or skip
                     }
                } else {
                    // std::cerr << "Warning: sample_idx " << sample_idx << " out of bounds for labels (0-" << labels.rows-1 << ")" << std::endl;
                    // Optionally set to a default color or skip
                }
            }
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
    double min_contour_area,
    int max_features_to_render,
    int original_svg_width,
    int original_svg_height
) {
    if (width <= 0 || height <= 0) {
        std::cerr << "Error: Invalid image dimensions (width=" << width << ", height=" << height << ")." << std::endl;
        std::stringstream error_svg;
        error_svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << std::max(1,width)
                  << "\" height=\"" << std::max(1,height)
                  << "\"><text x=\"10\" y=\"20\" fill=\"red\">Error: Invalid image dimensions</text></svg>";
        return error_svg.str();
    }
    cv::Mat temp_mat(height, width, CV_8UC3, const_cast<unsigned char*>(raw_bitmap_data_rgb_ptr));
    cv::Mat raw_img_rgb = temp_mat.clone();

    if (raw_img_rgb.empty()) {
        std::cerr << "Error: Raw image data is empty or Mat construction failed." << std::endl;
        std::stringstream error_svg;
        error_svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
                  << "\" height=\"" << height
                  << "\"><text x=\"10\" y=\"20\" fill=\"red\">Error: Could not load image data</text></svg>";
        return error_svg.str();
    }

    int actual_num_colors = num_colors_hint;
    auto quantization_result = perform_color_quantization_cpp(raw_img_rgb, actual_num_colors);

    cv::Mat quantized_img_rgb = quantization_result.first;
    std::vector<Color> palette = quantization_result.second;

    if (quantized_img_rgb.empty() || palette.empty()) {
        std::cerr << "Error: Color quantization failed or returned empty results." << std::endl;
        std::stringstream error_svg;
        error_svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
                  << "\" height=\"" << height
                  << "\"><text x=\"10\" y=\"20\" fill=\"red\">Error: Color quantization failed</text></svg>";
        return error_svg.str();
    }

    std::vector<SvgFeature> all_features;
    cv::Point2f image_center(static_cast<float>(quantized_img_rgb.cols) / 2.0f, static_cast<float>(quantized_img_rgb.rows) / 2.0f);
    double max_dist_from_center = cv::norm(cv::Point2f(0,0) - image_center);

    for (const auto& pal_color_struct : palette) {
        cv::Vec3b target_cv_color_rgb(pal_color_struct.r, pal_color_struct.g, pal_color_struct.b);
        cv::Mat mask;
        cv::inRange(quantized_img_rgb, target_cv_color_rgb, target_cv_color_rgb, mask);

        if (cv::countNonZero(mask) == 0) {
            continue;
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::string hex_color_str = compress_hex_color_cpp(pal_color_struct.r, pal_color_struct.g, pal_color_struct.b);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < min_contour_area) continue;

            std::vector<cv::Point> simplified_contour;
            double epsilon = std::max(0.5, simplification_epsilon_factor * cv::arcLength(contour, true));
            cv::approxPolyDP(contour, simplified_contour, epsilon, true);

            if (simplified_contour.size() < 3) continue;

            cv::Moments M = cv::moments(simplified_contour);
            cv::Point2f contour_center(0.0f, 0.0f);
            if (M.m00 > 1e-5) { // Avoid division by zero for very small areas
                contour_center.x = static_cast<float>(M.m10 / M.m00);
                contour_center.y = static_cast<float>(M.m01 / M.m00);
            }

            double dist_from_img_center = cv::norm(contour_center - image_center);
            double normalized_dist = (max_dist_from_center > 1e-5) ? (dist_from_img_center / max_dist_from_center) : 0.0;

            double importance = area * (1.0 - normalized_dist) * (1.0 / (simplified_contour.size() + 1.0));

            all_features.push_back({simplified_contour, hex_color_str, area, importance});
        }
    }

    std::sort(all_features.begin(), all_features.end()); // Sorts by importance (descending)

    std::stringstream svg_ss;
    int svg_attr_width = (original_svg_width > 0) ? original_svg_width : quantized_img_rgb.cols;
    int svg_attr_height = (original_svg_height > 0) ? original_svg_height : quantized_img_rgb.rows;

    svg_ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_attr_width
           << "\" height=\"" << svg_attr_height
           << "\" viewBox=\"0 0 " << quantized_img_rgb.cols << " " << quantized_img_rgb.rows << "\">";

    cv::Scalar avg_color_rgb_scalar = cv::mean(raw_img_rgb);
    std::string bg_hex_color = compress_hex_color_cpp(
        static_cast<unsigned char>(avg_color_rgb_scalar[0]), // R
        static_cast<unsigned char>(avg_color_rgb_scalar[1]), // G
        static_cast<unsigned char>(avg_color_rgb_scalar[2])  // B
    );
    svg_ss << "<rect width=\"" << quantized_img_rgb.cols
           << "\" height=\"" << quantized_img_rgb.rows
           << "\" fill=\"" << bg_hex_color << "\"/>";

    size_t features_rendered = 0;
    for (const auto& feature : all_features) {
        if (max_features_to_render > 0 && features_rendered >= static_cast<size_t>(max_features_to_render)) {
            break;
        }
        if (feature.points.empty()) continue;

        // Corrected check using the C++ constant
        if (static_cast<long>(svg_ss.tellp()) > (MAX_SVG_SIZE_BYTES - SVG_SIZE_SAFETY_MARGIN) ) {
             std::cerr << "Warning: Approaching max SVG size (" << MAX_SVG_SIZE_BYTES
                       << " bytes), truncating output. Current estimated size: " << svg_ss.tellp() << std::endl;
             break;
        }

        svg_ss << "<polygon points=\"";
        for (size_t i = 0; i < feature.points.size(); ++i) {
            svg_ss << feature.points[i].x << "," << feature.points[i].y;
            if (i < feature.points.size() - 1) {
                svg_ss << " ";
            }
        }
        svg_ss << "\" fill=\"" << feature.color_hex << "\"/>";
        features_rendered++;
    }

    svg_ss << "</svg>";
    return svg_ss.str();
}
