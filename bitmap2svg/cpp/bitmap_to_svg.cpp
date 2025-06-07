#include "bitmap_to_svg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <memory>
#include <unordered_set>

#ifdef WITH_CUDA
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudaimgproc.hpp>
#endif

#ifdef WITH_FAISS_GPU
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/Clustering.h>
#include <faiss/Index.h>
#endif

// Optimization 1: Use constexpr and more reasonable constants
constexpr long MAX_SVG_SIZE_BYTES = 10000;
constexpr long SVG_SIZE_SAFETY_MARGIN = 1000;
constexpr int MIN_CLUSTER_SIZE = 1;
constexpr int MAX_CLUSTER_SIZE = 256;
constexpr double MIN_EPSILON = 0.5;

// Optimization 3: Use more efficient color compression function
std::string compress_hex_color_optimized(unsigned char r, unsigned char g, unsigned char b) noexcept {
    // Pre-allocate string capacity
    std::string result;
    result.reserve(7); // "#RRGGBB"
    
    result += '#';
    
    // Check if it can be compressed to #RGB format
    if ((r & 0x0F) == (r >> 4) && (g & 0x0F) == (g >> 4) && (b & 0x0F) == (b >> 4)) {
        result += "0123456789abcdef"[r >> 4];
        result += "0123456789abcdef"[g >> 4];
        result += "0123456789abcdef"[b >> 4];
    } else {
        // Use lookup table instead of formatted output
        static const char hex_chars[] = "0123456789abcdef";
        result += hex_chars[r >> 4];
        result += hex_chars[r & 0x0F];
        result += hex_chars[g >> 4];
        result += hex_chars[g & 0x0F];
        result += hex_chars[b >> 4];
        result += hex_chars[b & 0x0F];
    }
    
    return result;
}

// Optimization 4: Add intelligent K value selection
int calculate_optimal_k(long pixel_count, int user_hint) noexcept {
    if (user_hint > 0) {
        return std::clamp(user_hint, MIN_CLUSTER_SIZE, MAX_CLUSTER_SIZE);
    }
    
    // Use more refined K value selection strategy
    if (pixel_count < 1024) return 4;
    if (pixel_count < 4096) return 6;
    if (pixel_count < 16384) return 8;
    if (pixel_count < 65536) return 10;
    if (pixel_count < 262144) return 12;
    return 16;
}

// Optimization 5: Use RAII and smart pointers for resource management
class FaissGpuManager {
#ifdef WITH_FAISS_GPU
private:
    std::unique_ptr<faiss::gpu::StandardGpuResources> resources_;
    
public:
    FaissGpuManager() {
        try {
            resources_ = std::make_unique<faiss::gpu::StandardGpuResources>();
        } catch (...) {
            resources_.reset();
        }
    }
    
    bool is_available() const noexcept { return resources_ != nullptr; }
    faiss::gpu::StandardGpuResources* get() const noexcept { return resources_.get(); }
#else
public:
    bool is_available() const noexcept { return false; }
    void* get() const noexcept { return nullptr; }
#endif
};

// Optimization 11: Pre-computation and caching
class SvgGenerator {
private:
    cv::Point2f image_center_;
    double max_dist_from_center_;
    std::ostringstream svg_stream_;
    size_t estimated_size_;
    
public:
    SvgGenerator(int width, int height) 
        : image_center_(width / 2.0f, height / 2.0f)
        , max_dist_from_center_(cv::norm(cv::Point2f(0, 0) - image_center_))
        , estimated_size_(0) {
        svg_stream_.str().reserve(MAX_SVG_SIZE_BYTES);
    }
    
    double calculate_importance(const std::vector<cv::Point>& contour, double area) const noexcept {
        cv::Moments M = cv::moments(contour);
        cv::Point2f centroid(0.0f, 0.0f);
        
        if (M.m00 > 1e-5) {
            centroid.x = static_cast<float>(M.m10 / M.m00);
            centroid.y = static_cast<float>(M.m01 / M.m00);
        }
        
        const double dist_from_center = cv::norm(centroid - image_center_);
        const double normalized_dist = (max_dist_from_center_ > 1e-5) ? 
            (dist_from_center / max_dist_from_center_) : 0.0;
        
        return area * (1.0 - normalized_dist * 0.5) * (100.0 / (contour.size() + 10.0));
    }
    
    bool can_add_feature() const noexcept {
        return estimated_size_ < (MAX_SVG_SIZE_BYTES - SVG_SIZE_SAFETY_MARGIN);
    }
    
    void add_polygon(const std::vector<cv::Point>& points, const std::string& color) {
        if (!can_add_feature()) return;
        
        svg_stream_ << "<polygon points=\"";
        for (size_t i = 0; i < points.size(); ++i) {
            svg_stream_ << points[i].x << "," << points[i].y;
            if (i < points.size() - 1) svg_stream_ << " ";
        }
        svg_stream_ << "\" fill=\"" << color << "\"/>";
        
        estimated_size_ = svg_stream_.tellp();
    }
    
    std::string get_svg() const { return svg_stream_.str(); }
};

// Error handling function for empty clustering centers
std::pair<cv::Mat, std::vector<Color>> handle_empty_centers(const cv::Mat& input_img_bgr) {
    std::cerr << "Warning: K-means clustering produced empty centers, using fallback method\n";
    
    // Create a simple palette using the most common colors in the image
    std::vector<Color> fallback_palette;
    cv::Mat quantized_img = input_img_bgr.clone();
    
    // Calculate the average color of the image as a single color
    cv::Scalar mean_color = cv::mean(input_img_bgr);
    Color avg_color;
    avg_color.b = static_cast<unsigned char>(mean_color[0]);
    avg_color.g = static_cast<unsigned char>(mean_color[1]);
    avg_color.r = static_cast<unsigned char>(mean_color[2]);
    
    fallback_palette.push_back(avg_color);
    
    // Set the entire image to the average color
    quantized_img.setTo(cv::Scalar(avg_color.b, avg_color.g, avg_color.r));
    
    return {quantized_img, fallback_palette};
}

// Generate OpenCV quantization result
std::pair<cv::Mat, std::vector<Color>> generate_quantized_result_opencv(
    const cv::Mat& input_img_bgr, 
    const cv::Mat& labels, 
    const cv::Mat& centers
) {
    // Create color palette
    std::vector<Color> palette_rgb;
    palette_rgb.reserve(centers.rows);
    
    for (int i = 0; i < centers.rows; ++i) {
        Color color;
        color.b = static_cast<unsigned char>(centers.at<float>(i, 0));
        color.g = static_cast<unsigned char>(centers.at<float>(i, 1));
        color.r = static_cast<unsigned char>(centers.at<float>(i, 2));
        palette_rgb.push_back(color);
    }
    
    // Create quantized image
    cv::Mat quantized_img(input_img_bgr.size(), input_img_bgr.type());
    
    int pixel_idx = 0;
    for (int i = 0; i < input_img_bgr.rows; ++i) {
        for (int j = 0; j < input_img_bgr.cols; ++j) {
            int cluster_idx = labels.at<int>(pixel_idx);
            if (cluster_idx >= 0 && cluster_idx < centers.rows) {
                cv::Vec3f center = centers.at<cv::Vec3f>(cluster_idx, 0);
                quantized_img.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    static_cast<unsigned char>(center[0]),
                    static_cast<unsigned char>(center[1]),
                    static_cast<unsigned char>(center[2])
                );
            } else {
                // Handle invalid cluster index
                quantized_img.at<cv::Vec3b>(i, j) = input_img_bgr.at<cv::Vec3b>(i, j);
            }
            pixel_idx++;
        }
    }
    
    return {quantized_img, palette_rgb};
}

#ifdef WITH_FAISS_GPU
// Generate FAISS quantization result
std::pair<cv::Mat, std::vector<Color>> generate_quantized_result_faiss(
    const cv::Mat& input_img_bgr,
    const faiss::Clustering& clustering,
    faiss::gpu::GpuIndexFlatL2& gpu_index,
    const float* data_ptr,
    long n_pixels
) {
    // Create color palette
    std::vector<Color> palette_rgb;
    const int k = clustering.k;
    palette_rgb.reserve(k);
    
    for (int i = 0; i < k; ++i) {
        Color color;
        color.b = static_cast<unsigned char>(std::clamp(clustering.centroids[i * 3 + 0], 0.0f, 255.0f));
        color.g = static_cast<unsigned char>(std::clamp(clustering.centroids[i * 3 + 1], 0.0f, 255.0f));
        color.r = static_cast<unsigned char>(std::clamp(clustering.centroids[i * 3 + 2], 0.0f, 255.0f));
        palette_rgb.push_back(color);
    }
    
    // Find the nearest cluster center for each pixel
    std::vector<float> distances(n_pixels);
    std::vector<faiss::Index::idx_t> labels(n_pixels);
    
    gpu_index.search(n_pixels, data_ptr, 1, distances.data(), labels.data());
    
    // Create quantized image
    cv::Mat quantized_img(input_img_bgr.size(), input_img_bgr.type());
    
    int pixel_idx = 0;
    for (int i = 0; i < input_img_bgr.rows; ++i) {
        for (int j = 0; j < input_img_bgr.cols; ++j) {
            int cluster_idx = static_cast<int>(labels[pixel_idx]);
            if (cluster_idx >= 0 && cluster_idx < k) {
                const Color& color = palette_rgb[cluster_idx];
                quantized_img.at<cv::Vec3b>(i, j) = cv::Vec3b(color.b, color.g, color.r);
            } else {
                // Handle invalid cluster index
                quantized_img.at<cv::Vec3b>(i, j) = input_img_bgr.at<cv::Vec3b>(i, j);
            }
            pixel_idx++;
        }
    }
    
    return {quantized_img, palette_rgb};
}

// Optimization 8: Separate FAISS and OpenCV implementations
std::pair<cv::Mat, std::vector<Color>> perform_faiss_quantization(
    const cv::Mat& input_img_bgr, 
    int k, 
    faiss::gpu::StandardGpuResources* resources
) {
    cv::Mat input_float;
    input_img_bgr.convertTo(input_float, CV_32FC3);
    
    const long n_pixels = input_float.rows * input_float.cols;
    const int dim = 3;
    
    if (n_pixels < k) {
        throw std::runtime_error("Not enough pixels for clustering");
    }
    
    // Ensure data continuity
    if (!input_float.isContinuous()) {
        input_float = input_float.clone();
    }
    
    cv::Mat samples = input_float.reshape(1, n_pixels);
    const float* data_ptr = samples.ptr<float>();
    
    // Configure clustering parameters
    faiss::ClusteringParameters cp;
    cp.niter = std::min(50, std::max(10, static_cast<int>(n_pixels / k / 100)));
    cp.nredo = 3;
    cp.verbose = false;
    cp.spherical = false;
    cp.update_index = true;
    cp.min_points_per_centroid = std::max(1, static_cast<int>(n_pixels / (k * 50)));
    cp.max_points_per_centroid = static_cast<int>(n_pixels / k * 5);
    
    faiss::Clustering clustering(dim, k, cp);
    faiss::gpu::GpuIndexFlatL2 gpu_index(resources, dim);
    
    clustering.train(n_pixels, data_ptr, gpu_index);
    
    if (clustering.centroids.empty()) {
        throw std::runtime_error("FAISS clustering produced no centroids");
    }
    
    // Generate quantized image and palette
    return generate_quantized_result_faiss(input_img_bgr, clustering, gpu_index, data_ptr, n_pixels);
}
#endif

// Optimization 9: Improved OpenCV quantization implementation
std::pair<cv::Mat, std::vector<Color>> perform_opencv_quantization(const cv::Mat& input_img_bgr, int k) {
    cv::Mat samples;
    input_img_bgr.convertTo(samples, CV_32F);
    samples = samples.reshape(1, input_img_bgr.total());
    
    if (samples.rows < k) {
        k = samples.rows;
        if (k == 0) return {cv::Mat(), {}};
    }
    
    cv::Mat labels, centers;
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, 0.1);
    const int attempts = (k <= 8) ? 5 : 3; // Reduce attempts to improve performance
    
    double compactness = cv::kmeans(samples, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers);
    
    if (centers.empty()) {
        return handle_empty_centers(input_img_bgr);
    }
    
    return generate_quantized_result_opencv(input_img_bgr, labels, centers);
}

// Optimization 6: Improved color quantization function
std::pair<cv::Mat, std::vector<Color>> perform_color_quantization_optimized(
    const cv::Mat& input_img_bgr,
    int& num_colors_target
) {
    // Input validation
    if (input_img_bgr.empty() || input_img_bgr.channels() != 3) {
        std::cerr << "Error: Invalid input image for quantization\n";
        return {cv::Mat(), {}};
    }

    const long pixel_count = static_cast<long>(input_img_bgr.rows) * input_img_bgr.cols;
    const int k = calculate_optimal_k(pixel_count, num_colors_target);
    num_colors_target = k;

    // Optimization 7: Use static GPU manager to avoid repeated initialization
    static FaissGpuManager gpu_manager;
    
#ifdef WITH_FAISS_GPU
    if (gpu_manager.is_available()) {
        try {
            return perform_faiss_quantization(input_img_bgr, k, gpu_manager.get());
        } catch (const std::exception& ex) {
            std::cerr << "FAISS GPU quantization failed: " << ex.what() 
                      << "\nFalling back to OpenCV K-means\n";
        }
    }
#endif

    return perform_opencv_quantization(input_img_bgr, k);
}

// Generate SVG from quantized image
std::string generate_svg_from_quantized_image(
    const cv::Mat& quantized_img_bgr,
    const std::vector<Color>& palette_rgb,
    const cv::Mat& original_img_bgr,
    double simplification_epsilon_factor,
    double min_contour_area,
    int max_features_to_render,
    int original_svg_width,
    int original_svg_height
) {
    try {
        // Set SVG dimensions
        int svg_width = (original_svg_width > 0) ? original_svg_width : quantized_img_bgr.cols;
        int svg_height = (original_svg_height > 0) ? original_svg_height : quantized_img_bgr.rows;
        
        // Create SVG generator
        SvgGenerator svg_gen(svg_width, svg_height);
        
        // SVG header
        std::ostringstream svg_stream;
        svg_stream << "<svg width=\"" << svg_width << "\" height=\"" << svg_height 
                   << "\" xmlns=\"http://www.w3.org/2000/svg\">";
        
        // Create contours for each color
        std::vector<ContourFeature> all_features;
        
        for (const auto& color : palette_rgb) {
            // Create mask for this color
            cv::Mat mask;
            cv::inRange(quantized_img_bgr, 
                       cv::Scalar(color.b - 1, color.g - 1, color.r - 1),
                       cv::Scalar(color.b + 1, color.g + 1, color.r + 1), 
                       mask);
            
            if (cv::countNonZero(mask) == 0) continue;
            
            // Find contours
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            // Process each contour
            for (auto& contour : contours) {
                double area = cv::contourArea(contour);
                if (area < min_contour_area) continue;
                
                // Simplify contour
                std::vector<cv::Point> simplified_contour;
                double epsilon = simplification_epsilon_factor * cv::arcLength(contour, true);
                cv::approxPolyDP(contour, simplified_contour, std::max(epsilon, MIN_EPSILON), true);
                
                if (simplified_contour.size() < 3) continue;
                
                // Calculate importance
                double importance = svg_gen.calculate_importance(simplified_contour, area);
                
                // Create color string
                std::string color_hex = compress_hex_color_optimized(color.r, color.g, color.b);
                
                // Add feature
                all_features.emplace_back(std::move(simplified_contour), std::move(color_hex), area, importance);
            }
        }
        
        // Sort by importance
        std::sort(all_features.begin(), all_features.end());
        
        // Limit feature count
        if (all_features.size() > static_cast<size_t>(max_features_to_render)) {
            all_features.resize(max_features_to_render);
        }
        
        // Generate SVG polygons
        for (const auto& feature : all_features) {
            if (!svg_gen.can_add_feature()) break;
            svg_gen.add_polygon(feature.points, feature.color_hex);
        }
        
        // Complete SVG
        svg_stream << svg_gen.get_svg() << "</svg>";
        
        return svg_stream.str();
        
    } catch (const std::exception& ex) {
        std::cerr << "Error generating SVG: " << ex.what() << std::endl;
        return "<svg><text fill='red'>Error: SVG generation failed.</text></svg>";
    }
}

// Optimization 12: Main function refactoring
std::string bitmapToSvg_with_internal_quantization(
    const unsigned char* raw_bitmap_data_rgb_ptr,
    int width,
    int height,
    int num_colors_hint,
    double simplification_epsilon_factor = 0.009,
    double min_contour_area = 10.0,
    int max_features_to_render = 0,
    int original_svg_width = 0,
    int original_svg_height = 0
) {
    try {
        // Input validation
        if (!raw_bitmap_data_rgb_ptr || width <= 0 || height <= 0) {
            return "<svg><text fill='red'>Error: Invalid input parameters.</text></svg>";
        }
        
        // Image conversion
        cv::Mat raw_img_rgb(height, width, CV_8UC3, const_cast<unsigned char*>(raw_bitmap_data_rgb_ptr));
        cv::Mat raw_img_bgr;
        cv::cvtColor(raw_img_rgb, raw_img_bgr, cv::COLOR_RGB2BGR);
        
        // Color quantization
        int actual_num_colors = num_colors_hint;
        auto [quantized_img_bgr, palette_rgb] = perform_color_quantization_optimized(raw_img_bgr, actual_num_colors);
        
        if (quantized_img_bgr.empty() || palette_rgb.empty()) {
            return "<svg><text fill='red'>Error: Color quantization failed.</text></svg>";
        }
        
        // Generate SVG
        return generate_svg_from_quantized_image(
            quantized_img_bgr, palette_rgb, raw_img_bgr,
            simplification_epsilon_factor, min_contour_area, max_features_to_render,
            original_svg_width, original_svg_height
        );
        
    } catch (const std::exception& ex) {
        std::cerr << "Error in bitmap to SVG conversion: " << ex.what() << std::endl;
        return "<svg><text fill='red'>Error: Conversion failed.</text></svg>";
    }
}
