#include "bitmap_to_svg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#ifdef WITH_CUDA // For OpenCV CUDA modules
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include "opencv2/cudev/common.hpp" // For cv::cuda::Stream
#endif

#ifdef WITH_FAISS_GPU // For FAISS GPU K-Means
#include <faiss/gpu/GpuClustering.h>
#include <faiss/gpu/GpuIndexFlat.h> // For GpuIndexFlatL2
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/Clustering.h> // For faiss::ClusteringParameters and faiss::Clustering
#include <faiss/FaissException.h> // For faiss::FaissException
#endif

// Define the SVG size constraint
const long MAX_SVG_SIZE_BYTES = 10000;
const long SVG_SIZE_SAFETY_MARGIN = 1000; // Buffer to stop before hitting the absolute max

// Helper function to convert RGB to compressed hex (e.g., #RRGGBB to #RGB if possible)
std::string compress_hex_color_cpp(unsigned char r, unsigned char g, unsigned char b) {
    std::stringstream ss;
    ss << "#";
    // Check if color can be compressed to #RGB format
    // This happens if R, G, and B are all multiples of 17 (0x11)
    // e.g., 0xAA -> 0xA, 0x33 -> 0x3
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

std::pair<cv::Mat, std::vector<Color>> perform_color_quantization_cpp(
    const cv::Mat& input_img_bgr, // Input BGR image (CV_8UC3), OpenCV standard
    int& num_colors_target        // In: desired num_colors. Out: actual num_colors used.
) {
    if (input_img_bgr.empty() || input_img_bgr.channels() != 3) {
        std::cerr << "Error: Input image for quantization is invalid. Must be 3 channels." << std::endl;
        return {cv::Mat(), {}};
    }

    int k = num_colors_target;
    if (k <= 0) { // Automatic k selection based on pixel count
        long pixel_count = static_cast<long>(input_img_bgr.rows) * input_img_bgr.cols;
        if (pixel_count == 0) k = 1;
        else if (pixel_count < 16384) k = 6;   // Small image
        else if (pixel_count < 65536) k = 8;   // Medium-small
        else if (pixel_count < 262144) k = 10; // Medium
        else k = 12;                           // Large
    }
    k = std::max(1, std::min(k, 256)); // Clamp k to a reasonable range

#ifdef WITH_FAISS_GPU
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "Attempting FAISS GPU K-Means path." << std::endl;
        try {
            faiss::gpu::StandardGpuResources res; // GPU resources for FAISS

            cv::Mat input_float_bgr; // Faiss operates on float data
            input_img_bgr.convertTo(input_float_bgr, CV_32FC3);

            long n_pixels = input_float_bgr.rows * input_float_bgr.cols;
            int dim = 3; // BGR or RGB dimensions

            if (n_pixels == 0) {
                std::cerr << "Error: No samples to process for FAISS k-means." << std::endl;
                return {cv::Mat(), {}};
            }
            if (n_pixels < k) { // Cannot have more clusters than samples
                k = n_pixels;
                if (k == 0) return {cv::Mat(), {}}; // Should be caught by n_pixels == 0
            }
            num_colors_target = k; // Update output parameter with potentially adjusted k

            // Upload image data to GPU for FAISS processing
            cv::cuda::GpuMat gpu_input_img_8u, gpu_input_img_32f;
            gpu_input_img_8u.upload(input_img_bgr); // Upload original 8-bit BGR
            gpu_input_img_8u.convertTo(gpu_input_img_32f, CV_32F); // Convert to 32-bit float on GPU

            if (!gpu_input_img_32f.isContinuous()) {
                 // This should ideally not happen after upload and convertTo
                std::cerr << "Error: GPU Mat for FAISS is not continuous. Cloning to make it continuous." << std::endl;
                gpu_input_img_32f = gpu_input_img_32f.clone(); // Force continuous memory
            }
            // Reshape to N x dim for FAISS (N samples, 3 dimensions)
            cv::cuda::GpuMat gpu_samples_bgr = gpu_input_img_32f.reshape(1, n_pixels);
            const float* d_samples = gpu_samples_bgr.ptr<float>(); // Pointer to GPU data

            faiss::ClusteringParameters cp;
            cp.niter = 50;  // Number of k-means iterations
            cp.nredo = 5;   // Number of times to redo k-means with different random seeds
            cp.verbose = false; // Suppress FAISS console output
            cp.spherical = false; // Standard Euclidean k-means
            cp.update_index = true;
            // Heuristics for min/max points per centroid, can be adjusted
            cp.min_points_per_centroid = std::max(1, static_cast<int>(n_pixels / (k * 20.0f)));
            cp.max_points_per_centroid = static_cast<int>((n_pixels / static_cast<float>(k)) * 10.0f);
            // cp.seed = 1234; // For reproducible k-means, if desired

            faiss::Clustering clustering(dim, k, cp);
            clustering.train(n_pixels, d_samples, res); // Train k-means model

            if (clustering.centroids.empty()) {
                std::cerr << "Error: FAISS k-means returned 0 centroids." << std::endl;
                // Fallback: could use average color or OpenCV path, here returning error
                return {cv::Mat(), {}};
            }
            
            num_colors_target = clustering.centroids.size() / dim; // Actual number of clusters found
            if (num_colors_target == 0) {
                 std::cerr << "Error: FAISS k-means resulted in 0 valid centroids after division by dim." << std::endl;
                 return {cv::Mat(), {}};
            }

            std::vector<float> h_centroids_bgr = clustering.centroids; // Download centroids (BGR order)

            // Assign labels to original samples using the found centroids
            faiss::gpu::GpuIndexFlatL2 centroid_index(&res, dim); // GPU index for searching
            centroid_index.add(num_colors_target, h_centroids_bgr.data()); // Add centroids to index

            std::vector<faiss::idx_t> h_labels(n_pixels); // Labels for each pixel
            std::vector<float> h_distances(n_pixels);    // Distances to closest centroid (optional)
            centroid_index.search(n_pixels, d_samples, 1, h_distances.data(), h_labels.data()); // Assign labels

            // Prepare palette (RGB) and quantized image (BGR)
            std::vector<Color> palette_vector_rgb; // Palette stores colors as RGB
            palette_vector_rgb.reserve(num_colors_target);
            for (int i = 0; i < num_colors_target; ++i) {
                palette_vector_rgb.push_back({
                    // Centroids from FAISS are BGR (same as input), convert to RGB for palette struct
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[i * dim + 2]))), // R
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[i * dim + 1]))), // G
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[i * dim + 0])))  // B
                });
            }

            cv::Mat quantized_image_bgr_cpu(input_img_bgr.size(), input_img_bgr.type()); // Output is BGR
            for (long r = 0; r < input_img_bgr.rows; ++r) {
                for (long c = 0; c < input_img_bgr.cols; ++c) {
                    long sample_idx = r * input_img_bgr.cols + c;
                    faiss::idx_t cluster_idx = h_labels[sample_idx];
                    if (cluster_idx >= 0 && cluster_idx < num_colors_target) {
                        cv::Vec3b& pixel = quantized_image_bgr_cpu.at<cv::Vec3b>(r, c); // OpenCV pixel is BGR
                        // Palette is RGB, centroids h_centroids_bgr are BGR.
                        // Assign BGR values from h_centroids_bgr directly.
                        pixel[0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[cluster_idx * dim + 0]))); // B
                        pixel[1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[cluster_idx * dim + 1]))); // G
                        pixel[2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[cluster_idx * dim + 2]))); // R
                    }
                }
            }
            std::cout << "FAISS GPU K-Means successful." << std::endl;
            return {quantized_image_bgr_cpu, palette_vector_rgb};

        } catch (const faiss::FaissException& ex) {
            std::cerr << "FAISS Exception in GPU quantization: " << ex.what() << std::endl;
            // Fallback to CPU path below
        } catch (const cv::Exception& ex) { // Catch OpenCV specific exceptions (e.g., CUDA errors)
            std::cerr << "OpenCV CUDA/core Exception in FAISS GPU path: " << ex.what() << std::endl;
            // Fallback to CPU path below
        }
    }
#endif // WITH_FAISS_GPU

    // Fallback to Original CPU Path (OpenCV k-means) or if no GPU/FAISS_GPU
    std::cout << "Using CPU (OpenCV) K-Means path for color quantization." << std::endl;
    cv::Mat samples_bgr_cpu_fallback(input_img_bgr.total(), 3, CV_32F);
    cv::Mat input_float_bgr_cpu_fallback;
    input_img_bgr.convertTo(input_float_bgr_cpu_fallback, CV_32F); // Convert BGR to float
    // Reshape image to a list of pixels (N samples, 3 channels)
    samples_bgr_cpu_fallback = input_float_bgr_cpu_fallback.reshape(1, input_img_bgr.total());

    if (samples_bgr_cpu_fallback.rows == 0) {
        std::cerr << "Error: No samples for OpenCV k-means." << std::endl;
        return {cv::Mat(), {}};
    }
    if (samples_bgr_cpu_fallback.rows < k) { k = samples_bgr_cpu_fallback.rows; if (k == 0) return {cv::Mat(), {}}; }
    num_colors_target = k; // Update output based on potentially adjusted k

    cv::Mat labels_cpu_fallback;
    cv::Mat centers_bgr_cpu_fallback; // OpenCV kmeans centers are BGR
    // K-means criteria: (type, max_iter, epsilon)
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, 0.1);
    int attempts = (k <= 8) ? 5 : 7; // More attempts for more clusters

    if (k > 0) {
        cv::kmeans(samples_bgr_cpu_fallback, k, labels_cpu_fallback, criteria, attempts, cv::KMEANS_PP_CENTERS, centers_bgr_cpu_fallback);
    } else {
         std::cerr << "Error: k=0 for OpenCV kmeans." << std::endl; return {cv::Mat(), {}};
    }

    if (centers_bgr_cpu_fallback.rows == 0) { // If k-means fails to find centers
        std::cerr << "Warning: OpenCV k-means returned 0 centers. Using average image color." << std::endl;
        if (input_img_bgr.total() > 0 && !input_img_bgr.empty()) {
            cv::Scalar avg_color_scalar_bgr = cv::mean(input_img_bgr); // BGR
            cv::Mat single_color_img(input_img_bgr.size(), input_img_bgr.type());
            single_color_img.setTo(cv::Vec3b(static_cast<unsigned char>(avg_color_scalar_bgr[0]), // B
                                             static_cast<unsigned char>(avg_color_scalar_bgr[1]), // G
                                             static_cast<unsigned char>(avg_color_scalar_bgr[2])) // R
                                  );
            std::vector<Color> single_palette_rgb = {{ // Palette is RGB
                static_cast<unsigned char>(avg_color_scalar_bgr[2]), // R
                static_cast<unsigned char>(avg_color_scalar_bgr[1]), // G
                static_cast<unsigned char>(avg_color_scalar_bgr[0])  // B
            }};
            num_colors_target = 1;
            return {single_color_img, single_palette_rgb};
        }
        return {cv::Mat(), {}}; // Should not happen if input_img_bgr was valid
    }

    num_colors_target = centers_bgr_cpu_fallback.rows; // Actual k used by OpenCV

    cv::Mat quantized_image_bgr_cpu_fallback(input_img_bgr.size(), input_img_bgr.type()); // BGR
    std::vector<Color> palette_vector_rgb_fallback; // Palette is RGB
    palette_vector_rgb_fallback.reserve(centers_bgr_cpu_fallback.rows);

    for (int i = 0; i < centers_bgr_cpu_fallback.rows; ++i) {
        // centers_bgr_cpu_fallback are float BGR (OpenCV convention)
        palette_vector_rgb_fallback.push_back({
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(i, 2)))), // R
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(i, 1)))), // G
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(i, 0))))  // B
        });
    }

    if (labels_cpu_fallback.empty() || labels_cpu_fallback.rows != samples_bgr_cpu_fallback.rows ) {
        std::cerr << "Error: OpenCV kmeans failed to produce valid labels. Resulting image might be incorrect." << std::endl;
        // Potentially fill with average color or first palette color as a simple recovery
    } else {
        int* plabels = labels_cpu_fallback.ptr<int>(0);
        for (int r_idx = 0; r_idx < quantized_image_bgr_cpu_fallback.rows; ++r_idx) {
            for (int c_idx = 0; c_idx < quantized_image_bgr_cpu_fallback.cols; ++c_idx) {
                int sample_idx = r_idx * quantized_image_bgr_cpu_fallback.cols + c_idx;
                if (sample_idx < labels_cpu_fallback.rows && plabels) {
                    int cluster_idx = plabels[sample_idx];
                    if (cluster_idx >=0 && cluster_idx < centers_bgr_cpu_fallback.rows) {
                        cv::Vec3b& pixel = quantized_image_bgr_cpu_fallback.at<cv::Vec3b>(r_idx, c_idx); // BGR
                        // centers_bgr_cpu_fallback is BGR, float
                        pixel[0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(cluster_idx, 0)))); // B
                        pixel[1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(cluster_idx, 1)))); // G
                        pixel[2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(cluster_idx, 2)))); // R
                    }
                }
            }
        }
    }
    return {quantized_image_bgr_cpu_fallback, palette_vector_rgb_fallback};
}

// Main conversion function
std::string bitmapToSvg_with_internal_quantization(
    const unsigned char* raw_bitmap_data_rgb_ptr, // Input raw pixel data (RGB order)
    int width,
    int height,
    int num_colors_hint,
    double simplification_epsilon_factor, // = 0.009,
    double min_contour_area,             // = 10.0,
    int max_features_to_render,         // = 0 (no limit),
    int original_svg_width,             // = -1 (use image width),
    int original_svg_height             // = -1 (use image height)
) {
    // Wrap raw RGB data into an OpenCV Mat
    cv::Mat raw_img_rgb_data_wrapper(height, width, CV_8UC3, const_cast<unsigned char*>(raw_bitmap_data_rgb_ptr));
    cv::Mat raw_img_bgr; // OpenCV typically works with BGR, so convert
    cv::cvtColor(raw_img_rgb_data_wrapper, raw_img_bgr, cv::COLOR_RGB2BGR);

    if (raw_img_bgr.empty()) {
        std::cerr << "Error: Input image data is empty or invalid after BGR conversion." << std::endl;
        return "<svg><text fill='red'>Error: Invalid input image.</text></svg>";
    }

    int actual_num_colors = num_colors_hint; // Will be updated by quantization function
    // Perform color quantization (either Faiss GPU or OpenCV CPU)
    // perform_color_quantization_cpp expects BGR input and returns BGR image + RGB palette
    auto quantization_result = perform_color_quantization_cpp(raw_img_bgr, actual_num_colors);

    cv::Mat quantized_img_bgr = quantization_result.first;    // This is on CPU, in BGR order
    std::vector<Color> palette_rgb = quantization_result.second; // This palette is RGB

    if (quantized_img_bgr.empty() || palette_rgb.empty()) {
        std::cerr << "Error: Color quantization failed." << std::endl;
        return "<svg><text fill='red'>Error: Color quantization failed.</text></svg>";
    }

    std::vector<SvgFeature> all_features;
    cv::Point2f image_center(static_cast<float>(quantized_img_bgr.cols) / 2.0f, static_cast<float>(quantized_img_bgr.rows) / 2.0f);
    double max_dist_from_center = cv::norm(cv::Point2f(0,0) - image_center); // Max possible distance

    for (const auto& pal_color_rgb : palette_rgb) { // Palette is RGB
        // Target color for cv::inRange needs to be BGR because quantized_img_bgr is BGR
        cv::Vec3b target_cv_color_bgr(pal_color_rgb.b, pal_color_rgb.g, pal_color_rgb.r);
        cv::Mat mask;
        cv::inRange(quantized_img_bgr, target_cv_color_bgr, target_cv_color_bgr, mask);

        if (cv::countNonZero(mask) == 0) continue; // Skip if color not present

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Hex color for SVG should be from the RGB palette_rgb
        std::string hex_color_str = compress_hex_color_cpp(pal_color_rgb.r, pal_color_rgb.g, pal_color_rgb.b);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < min_contour_area) continue; // Skip small contours

            std::vector<cv::Point> simplified_contour;
            double epsilon = std::max(0.5, simplification_epsilon_factor * cv::arcLength(contour, true));
            cv::approxPolyDP(contour, simplified_contour, epsilon, true);

            if (simplified_contour.size() < 3) continue; // Need at least a triangle

            cv::Moments M = cv::moments(simplified_contour);
            cv::Point2f contour_center(0.0f, 0.0f);
            if (M.m00 > 1e-5) { // Avoid division by zero
                contour_center.x = static_cast<float>(M.m10 / M.m00);
                contour_center.y = static_cast<float>(M.m01 / M.m00);
            }
            double dist_from_img_center = cv::norm(contour_center - image_center);
            // Normalize distance: 0 for center, 1 for corners/edges
            double normalized_dist = (max_dist_from_center > 1e-5) ? (dist_from_img_center / max_dist_from_center) : 0.0;
            // Importance heuristic: favors larger, more central, simpler shapes
            double importance = area * (1.0 - normalized_dist) * (1.0 / (simplified_contour.size() + 1.0));
            all_features.push_back({simplified_contour, hex_color_str, area, importance});
        }
    }

    std::sort(all_features.begin(), all_features.end()); // Sort by importance (descending)

    // SVG Generation
    std::stringstream svg_ss;
    // Use original image dimensions for viewBox, but SVG width/height can be different
    int viewbox_w = raw_img_bgr.cols; // Viewbox from actual processed image dimensions
    int viewbox_h = raw_img_bgr.rows;
    int svg_attr_width = (original_svg_width > 0) ? original_svg_width : viewbox_w;
    int svg_attr_height = (original_svg_height > 0) ? original_svg_height : viewbox_h;

    svg_ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_attr_width
           << "\" height=\"" << svg_attr_height
           << "\" viewBox=\"0 0 " << viewbox_w << " " << viewbox_h << "\">";

    // Background: use average color of the original (input) image
    // raw_img_bgr is BGR, so its mean is BGR
    cv::Scalar avg_color_bgr_scalar = cv::mean(raw_img_bgr);
    std::string bg_hex_color = compress_hex_color_cpp(
        static_cast<unsigned char>(avg_color_bgr_scalar[2]), // R from BGR mean
        static_cast<unsigned char>(avg_color_bgr_scalar[1]), // G from BGR mean
        static_cast<unsigned char>(avg_color_bgr_scalar[0])  // B from BGR mean
    );
    svg_ss << "<rect width=\"" << viewbox_w
           << "\" height=\"" << viewbox_h
           << "\" fill=\"" << bg_hex_color << "\"/>";
    
    size_t features_rendered = 0;
    for (const auto& feature : all_features) {
        if (max_features_to_render > 0 && features_rendered >= static_cast<size_t>(max_features_to_render)) break;
        if (feature.points.empty()) continue;
        
        // Check estimated SVG size before adding more features
        // tellp() gives current stream position, roughly current size
        if (static_cast<long>(svg_ss.tellp()) > (MAX_SVG_SIZE_BYTES - SVG_SIZE_SAFETY_MARGIN) ) {
            std::cerr << "Warning: Approaching max SVG size (" << MAX_SVG_SIZE_BYTES 
                      << " bytes), truncating output. Current estimated size: " << svg_ss.tellp() << std::endl;
            break;
        }
        svg_ss << "<polygon points=\"";
        for (size_t i = 0; i < feature.points.size(); ++i) {
            svg_ss << feature.points[i].x << "," << feature.points[i].y;
            if (i < feature.points.size() - 1) svg_ss << " ";
        }
        svg_ss << "\" fill=\"" << feature.color_hex << "\"/>"; // feature.color_hex is already RGB hex
        features_rendered++;
    }

    svg_ss << "</svg>";
    return svg_ss.str();
}
