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
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/Clustering.h> // For faiss::Clustering
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
    const cv::Mat& input_img_rgb, // Input raw RGB image (CV_8UC3)
    int& num_colors_target       // In: desired num_colors. Out: actual num_colors used.
) {
    if (input_img_rgb.empty() || input_img_rgb.channels() != 3) {
        std::cerr << "Error: Input image for quantization is invalid. Must be 3 channels." << std::endl;
        return {cv::Mat(), {}};
    }

    int k = num_colors_target;
    if (k <= 0) {
        long pixel_count = static_cast<long>(input_img_rgb.rows) * input_img_rgb.cols;
        if (pixel_count == 0) k = 1;
        else if (pixel_count < 16384) k = 6;
        else if (pixel_count < 65536) k = 8;
        else if (pixel_count < 262144) k = 10;
        else k = 12;
    }
    k = std::max(1, std::min(k, 256)); // FAISS can handle more clusters, but keep it reasonable. Max 16 was for auto.
    // User can still request more via num_colors_target if k was initially > 0.

#ifdef WITH_FAISS_GPU
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "Attempting FAISS GPU K-Means path." << std::endl;
        try {
            faiss::gpu::StandardGpuResources res; // GPU resources
            // res.setTempMemory(desired_temp_memory_bytes); // Optional: for large datasets

            cv::Mat input_float_rgb;
            input_img_rgb.convertTo(input_float_rgb, CV_32FC3); // FAISS expects float data

            long n_pixels = input_float_rgb.rows * input_float_rgb.cols;
            int dim = 3; // RGB

            if (n_pixels == 0) {
                std::cerr << "Error: No samples to process for FAISS k-means." << std::endl;
                return {cv::Mat(), {}};
            }
            if (n_pixels < k) {
                k = n_pixels;
                if (k == 0) return {cv::Mat(), {}};
            }
            num_colors_target = k; // Update output parameter

            // FAISS expects a flat array of floats (n_pixels * dim)
            // We can upload to GpuMat first, then get its pointer if it's continuous.
            // Or, directly use input_float_rgb.ptr<float>(0) if it's already on CPU
            // For GPU path, ensure data is on GPU for FAISS.

            cv::cuda::GpuMat gpu_input_img_8u, gpu_input_img_32f;
            gpu_input_img_8u.upload(input_img_rgb);
            gpu_input_img_8u.convertTo(gpu_input_img_32f, CV_32F); // Scale to 0-1 or use as 0-255? FAISS doesn't care but affects distances. Let's keep 0-255.

            if (!gpu_input_img_32f.isContinuous()) {
                 std::cerr << "Error: GPU Mat for FAISS is not continuous. This should not happen with reshape/create." << std::endl;
                 // If it happened, gpu_input_img_32f = gpu_input_img_32f.clone(); might fix.
                 // But convertTo from upload should be continuous.
                 // Fallback or error out. For now, error out.
                 return {cv::Mat(), {}};
            }
            // FAISS expects data in shape (n_pixels, dim)
            cv::cuda::GpuMat gpu_samples_rgb = gpu_input_img_32f.reshape(1, n_pixels); // N x 3, still on GPU

            const float* d_samples = gpu_samples_rgb.ptr<float>();

            faiss::ClusteringParameters cp;
            cp.niter = 50; // Number of iterations
            cp.nredo = 5;  // Number of times to redo k-means with different random seeds
            cp.verbose = false;
            cp.spherical = false; // Standard Euclidean k-means
            cp.update_index = true;
            cp.min_points_per_centroid = std::max(1, (int)(n_pixels / (k * 20))); // Heuristic
            cp.max_points_per_centroid = (int)(n_pixels / k) * 10; // Heuristic
            // cp.seed = time(nullptr); // Or a fixed seed for reproducibility

            faiss::Clustering clustering(dim, k, cp);
            clustering.train(n_pixels, d_samples, res);

            if (clustering.centroids.empty()) {
                std::cerr << "Error: FAISS k-means returned 0 centroids." << std::endl;
                // Fallback to average color or error out
                // (Simplified fallback: use first pixel color, or average)
                if (input_img_rgb.total() > 0 && !input_img_rgb.empty()) {
                    cv::Scalar avg_color_scalar = cv::mean(input_img_rgb);
                }
                return {cv::Mat(), {}};
            }
            
            num_colors_target = clustering.centroids.size() / dim; // Actual k used
            if (num_colors_target == 0) { /* ... handle error ...*/ return {cv::Mat(), {}}; }


            std::vector<float> h_centroids = clustering.centroids; // Download centroids

            // --- Assign labels using FAISS GPU index ---
            faiss::gpu::GpuIndexFlatL2 index(&res, dim);
            index.add(n_pixels, d_samples); // Add original samples to index
            std::vector<faiss::idx_t> h_labels(n_pixels); // For CPU labels
            std::vector<float> h_distances(n_pixels);    // For distances (optional)
            
            // To assign labels to the original samples based on the found centroids:
            // We need to create an index with the centroids and search the original samples in it.
            faiss::gpu::GpuIndexFlatL2 centroid_index(&res, dim);
            centroid_index.add(num_colors_target, h_centroids.data()); // Add centroids to index
            centroid_index.search(n_pixels, d_samples, 1, h_distances.data(), h_labels.data()); // Find closest centroid for each sample


            // --- Prepare palette and quantized image ---
            std::vector<Color> palette_vector_Color_struct;
            palette_vector_Color_struct.reserve(num_colors_target);
            for (int i = 0; i < num_colors_target; ++i) {
                palette_vector_Color_struct.push_back({
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids[i * dim + 0]))),
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids[i * dim + 1]))),
                    static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids[i * dim + 2])))
                });
            }

            cv::Mat quantized_image_rgb_cpu(input_img_rgb.size(), input_img_rgb.type());
            for (long r = 0; r < input_img_rgb.rows; ++r) {
                for (long c = 0; c < input_img_rgb.cols; ++c) {
                    long sample_idx = r * input_img_rgb.cols + c;
                    faiss::idx_t cluster_idx = h_labels[sample_idx];
                    if (cluster_idx >= 0 && cluster_idx < num_colors_target) {
                        cv::Vec3b& pixel = quantized_image_rgb_cpu.at<cv::Vec3b>(r, c);
                        const Color& new_color = palette_vector_Color_struct[cluster_idx];
                        pixel[0] = new_color.r; // Assuming BGR order for OpenCV Vec3b
                        pixel[1] = new_color.g;
                        pixel[2] = new_color.b;
                        pixel[0] = palette_vector_Color_struct[cluster_idx].b;
                        pixel[1] = palette_vector_Color_struct[cluster_idx].g;
                        pixel[2] = palette_vector_Color_struct[cluster_idx].r;
                    }
                }
            }
            std::cout << "FAISS GPU K-Means successful." << std::endl;
            return {quantized_image_rgb_cpu, palette_vector_Color_struct};

        } catch (const faiss::FaissException& ex) {
            std::cerr << "FAISS Exception in quantization: " << ex.what() << std::endl;
            // Fallback to CPU path
        } catch (const cv::Exception& ex) {
            std::cerr << "OpenCV CUDA/core Exception in FAISS path: " << ex.what() << std::endl;
            // Fallback to CPU path
        }
    }
#endif // WITH_FAISS_GPU

    // Fallback to Original CPU Path (OpenCV k-means)
    std::cout << "Using CPU (OpenCV) K-Means path." << std::endl;
    cv::Mat samples_rgb_cpu_fallback(input_img_rgb.total(), 3, CV_32F);
    cv::Mat input_float_rgb_cpu_fallback;
    input_img_rgb.convertTo(input_float_rgb_cpu_fallback, CV_32F);
    samples_rgb_cpu_fallback = input_float_rgb_cpu_fallback.reshape(1, input_img_rgb.total());

    if (samples_rgb_cpu_fallback.rows == 0) { /* ... */ return {cv::Mat(), {}}; }
    if (samples_rgb_cpu_fallback.rows < k) { k = samples_rgb_cpu_fallback.rows; if (k == 0) return {cv::Mat(), {}}; }
    num_colors_target = k; // Update output based on potentially adjusted k

    cv::Mat labels_cpu_fallback;
    cv::Mat centers_rgb_cpu_fallback;
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, 0.1); // OpenCV Kmeans criteria
    int attempts = (k <= 8) ? 5 : 7;

    if (k > 0) {
        cv::kmeans(samples_rgb_cpu_fallback, k, labels_cpu_fallback, criteria, attempts, cv::KMEANS_PP_CENTERS, centers_rgb_cpu_fallback);
    } else { /* ... error handling ... */ return {cv::Mat(), {}}; }

    if (centers_rgb_cpu_fallback.rows == 0) { /* ... fallback to average color logic ... */
        if (input_img_rgb.total() > 0 && !input_img_rgb.empty()) {
             cv::Scalar avg_color_scalar = cv::mean(input_img_rgb);
             cv::Mat single_color_img(input_img_rgb.size(), input_img_rgb.type());
             single_color_img.setTo(cv::Vec3b(static_cast<unsigned char>(avg_color_scalar[0]),
                                             static_cast<unsigned char>(avg_color_scalar[1]),
                                             static_cast<unsigned char>(avg_color_scalar[2])));
             std::vector<Color> single_palette = {{
                 static_cast<unsigned char>(avg_color_scalar[2]), // R from BGR
                 static_cast<unsigned char>(avg_color_scalar[1]), // G from BGR
                 static_cast<unsigned char>(avg_color_scalar[0])  // B from BGR
             }};
             num_colors_target = 1;
             return {single_color_img, single_palette};
        }
        return {cv::Mat(), {}};
    }

    num_colors_target = centers_rgb_cpu_fallback.rows; // Actual k used

    cv::Mat quantized_image_rgb_cpu_fallback(input_img_rgb.size(), input_img_rgb.type());
    std::vector<Color> palette_vector_Color_struct_fallback;
    palette_vector_Color_struct_fallback.reserve(centers_rgb_cpu_fallback.rows);

    for (int i = 0; i < centers_rgb_cpu_fallback.rows; ++i) {
        // centers_rgb are float BGR (OpenCV convention)
        palette_vector_Color_struct_fallback.push_back({
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb_cpu_fallback.at<float>(i, 2)))), // R
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb_cpu_fallback.at<float>(i, 1)))), // G
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb_cpu_fallback.at<float>(i, 0))))  // B
        });
    }

    if (labels_cpu_fallback.empty() || labels_cpu_fallback.rows != samples_rgb_cpu_fallback.rows ) {
        std::cerr << "Error: OpenCV kmeans failed to produce valid labels. Using palette average for image." << std::endl;
    } else {
        int* plabels = labels_cpu_fallback.ptr<int>(0);
        for (int r_idx = 0; r_idx < quantized_image_rgb_cpu_fallback.rows; ++r_idx) {
            for (int c_idx = 0; c_idx < quantized_image_rgb_cpu_fallback.cols; ++c_idx) {
                int sample_idx = r_idx * quantized_image_rgb_cpu_fallback.cols + c_idx;
                if (sample_idx < labels_cpu_fallback.rows && plabels) {
                    int cluster_idx = plabels[sample_idx];
                    if (cluster_idx >=0 && cluster_idx < centers_rgb_cpu_fallback.rows) {
                        cv::Vec3b& pixel = quantized_image_rgb_cpu_fallback.at<cv::Vec3b>(r_idx, c_idx);
                        // centers_rgb_cpu_fallback is BGR
                        pixel[0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb_cpu_fallback.at<float>(cluster_idx, 0)))); // B
                        pixel[1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb_cpu_fallback.at<float>(cluster_idx, 1)))); // G
                        pixel[2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_rgb_cpu_fallback.at<float>(cluster_idx, 2)))); // R
                    }
                }
            }
        }
    }
    return {quantized_image_rgb_cpu_fallback, palette_vector_Color_struct_fallback};
}

// Main conversion function
std::string bitmapToSvg_with_internal_quantization(
    const unsigned char* raw_bitmap_data_rgb_ptr,
    int width,
    int height,
    int num_colors_hint,
    double simplification_epsilon_factor, // = 0.009,
    double min_contour_area, // = 10.0,
    int max_features_to_render, // = 0,
    int original_svg_width, // = -1,
    int original_svg_height // = -1
) {
    cv::Mat raw_img_rgb_data_wrapper(height, width, CV_8UC3, const_cast<unsigned char*>(raw_bitmap_data_rgb_ptr));
    cv::Mat raw_img_bgr; // Let's work with BGR internally for OpenCV consistency
    cv::cvtColor(raw_img_rgb_data_wrapper, raw_img_bgr, cv::COLOR_RGB2BGR);


    if (raw_img_bgr.empty()) {
        return "...Error SVG..."; 
    }

    int actual_num_colors = num_colors_hint;
    auto quantization_result = perform_color_quantization_cpp(raw_img_bgr, actual_num_colors);

    cv::Mat quantized_img_bgr = quantization_result.first; // This is on CPU, in BGR order
    std::vector<Color> palette_rgb = quantization_result.second; // This palette is RGB

    if (quantized_img_bgr.empty() || palette_rgb.empty()) { /* ... error SVG ... */ return "...Error SVG..."; }

    std::vector<SvgFeature> all_features;
    cv::Point2f image_center(static_cast<float>(quantized_img_bgr.cols) / 2.0f, static_cast<float>(quantized_img_bgr.rows) / 2.0f);
    double max_dist_from_center = cv::norm(cv::Point2f(0,0) - image_center);

    for (const auto& pal_color_struct_rgb : palette_rgb) { // Palette is RGB
        // Target color for inRange needs to be BGR because quantized_img_bgr is BGR
        cv::Vec3b target_cv_color_bgr(pal_color_struct_rgb.b, pal_color_struct_rgb.g, pal_color_struct_rgb.r);
        cv::Mat mask;
        cv::inRange(quantized_img_bgr, target_cv_color_bgr, target_cv_color_bgr, mask);

        if (cv::countNonZero(mask) == 0) continue;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Hex color for SVG should be from the RGB palette_rgb
        std::string hex_color_str = compress_hex_color_cpp(pal_color_struct_rgb.r, pal_color_struct_rgb.g, pal_color_struct_rgb.b);

        for (const auto& contour : contours) {
            double area = cv::contourArea(contour);
            if (area < min_contour_area) continue;

            std::vector<cv::Point> simplified_contour;
            double epsilon = std::max(0.5, simplification_epsilon_factor * cv::arcLength(contour, true));
            cv::approxPolyDP(contour, simplified_contour, epsilon, true);

            if (simplified_contour.size() < 3) continue;

            cv::Moments M = cv::moments(simplified_contour);
            cv::Point2f contour_center(0.0f, 0.0f);
            if (M.m00 > 1e-5) {
                contour_center.x = static_cast<float>(M.m10 / M.m00);
                contour_center.y = static_cast<float>(M.m01 / M.m00);
            }
            double dist_from_img_center = cv::norm(contour_center - image_center);
            double normalized_dist = (max_dist_from_center > 1e-5) ? (dist_from_img_center / max_dist_from_center) : 0.0;
            double importance = area * (1.0 - normalized_dist) * (1.0 / (simplified_contour.size() + 1.0));
            all_features.push_back({simplified_contour, hex_color_str, area, importance});
        }
    }

    std::sort(all_features.begin(), all_features.end());

    // SVG Generation
    std::stringstream svg_ss;
    // Use original image dimensions for viewBox, but SVG width/height can be different
    int viewbox_w = raw_img_bgr.cols;
    int viewbox_h = raw_img_bgr.rows;
    int svg_attr_width = (original_svg_width > 0) ? original_svg_width : viewbox_w;
    int svg_attr_height = (original_svg_height > 0) ? original_svg_height : viewbox_h;

    svg_ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_attr_width
           << "\" height=\"" << svg_attr_height
           << "\" viewBox=\"0 0 " << viewbox_w << " " << viewbox_h << "\">";

    // Background: use average color of the original RGB image
    cv::Scalar avg_color_bgr_scalar = cv::mean(raw_img_bgr); // This is BGR average
    std::string bg_hex_color = compress_hex_color_cpp(
        static_cast<unsigned char>(avg_color_bgr_scalar[2]), // R from BGR
        static_cast<unsigned char>(avg_color_bgr_scalar[1]), // G from BGR
        static_cast<unsigned char>(avg_color_bgr_scalar[0])  // B from BGR
    );
    svg_ss << "<rect width=\"" << viewbox_w
           << "\" height=\"" << viewbox_h
           << "\" fill=\"" << bg_hex_color << "\"/>";
    
    size_t features_rendered = 0;
    for (const auto& feature : all_features) {
        if (max_features_to_render > 0 && features_rendered >= static_cast<size_t>(max_features_to_render)) break;
        if (feature.points.empty()) continue;
        if (static_cast<long>(svg_ss.tellp()) > (MAX_SVG_SIZE_BYTES - SVG_SIZE_SAFETY_MARGIN) ) {
             std::cerr << "Warning: Approaching max SVG size, truncating output." << std::endl;
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
