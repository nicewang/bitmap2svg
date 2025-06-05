#include "bitmap_to_svg.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <stdexcept> // For std::runtime_error

#ifdef WITH_CUDA // For OpenCV CUDA modules
// #include <opencv2/cudaarithm.hpp>
// #include <opencv2/cudaimgproc.hpp>
// #include "opencv2/cudev/common.hpp" // For cv::cuda::Stream
#endif

#ifdef WITH_FAISS_GPU // For FAISS GPU K-Means
#include "faiss/gpu/GpuIndexFlat.h" // For GpuIndexFlatL2
#include "faiss/gpu/StandardGpuResources.h"
#include "faiss/Clustering.h"    // For faiss::ClusteringParameters and faiss::Clustering
#include "faiss/Index.h"         // To get faiss::Index and faiss::idx_t (using 'Index.h' as per your last attempt)
#endif

// --- TEMPORARY DEBUGGING CODE ---
#ifndef FAISS_INDEX_H
#pragma message("WARNING: faiss/Index.h (or index.h) does NOT define FAISS_INDEX_H. This might indicate an incomplete or incorrect header.")
#endif
// --- END TEMPORARY DEBUGGING CODE ---

// Define the SVG size constraint
const long MAX_SVG_SIZE_BYTES = 10000;
const long SVG_SIZE_SAFETY_MARGIN = 1000; // Buffer to stop before hitting the absolute max

// Helper function to convert RGB to compressed hex (e.g., #RRGGBB to #RGB if possible)
std::string compress_hex_color_cpp(unsigned char r, unsigned char g, unsigned char b) {
    std::stringstream ss;
    ss << "#";
    // Check if color can be compressed to #RGB format
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
    // Attempt FAISS GPU path first.
    try {
        faiss::gpu::StandardGpuResources res; // Manages GPU resources
        std::cout << "FAISS GPU resources initialized. Attempting FAISS K-Means." << std::endl;

        cv::Mat input_float_bgr_cpu;
        input_img_bgr.convertTo(input_float_bgr_cpu, CV_32FC3);

        long n_pixels = input_float_bgr_cpu.rows * input_float_bgr_cpu.cols;
        int dim = 3;

        if (n_pixels == 0) {
            std::cerr << "Error: No samples to process for FAISS k-means (data prepared on CPU)." << std::endl;
            throw std::runtime_error("No samples to process for FAISS k-means.");
        }
        if (n_pixels < k) {
            k = n_pixels;
            if (k == 0) throw std::runtime_error("k became 0 after n_pixels check.");
        }
        num_colors_target = k;

        if (!input_float_bgr_cpu.isContinuous()) {
            input_float_bgr_cpu = input_float_bgr_cpu.clone();
        }
        cv::Mat samples_bgr_cpu_for_faiss = input_float_bgr_cpu.reshape(1, n_pixels);
        const float* h_samples_cpu_ptr = samples_bgr_cpu_for_faiss.ptr<float>();

        faiss::ClusteringParameters cp;
        cp.niter = 50;
        cp.nredo = 5;
        cp.verbose = false;
        cp.spherical = false;
        cp.update_index = true;
        cp.min_points_per_centroid = std::max(1, static_cast<int>(n_pixels / (k * 20.0f)));
        cp.max_points_per_centroid = static_cast<int>((n_pixels / static_cast<float>(k)) * 10.0f);

        faiss::Clustering clustering(dim, k, cp);

        // --- Crucial change for FAISS GPU Clustering ---
        // For GPU clustering, the train method typically expects a GPU index
        // to manage the data and perform the clustering on the GPU.
        // We create a dummy GpuIndexFlatL2 for this purpose.
        faiss::gpu::GpuIndexFlatL2 gpu_index(&res, dim);
        clustering.train(n_pixels, h_samples_cpu_ptr, gpu_index); // Pass the GPU index

        // Centroids are now in clustering.centroids (still on CPU, transferred back by FAISS)
        if (clustering.centroids.empty()) {
            std::cerr << "Error: FAISS k-means (GPU path) returned 0 centroids." << std::endl;
            throw std::runtime_error("FAISS k-means (GPU path) returned 0 centroids.");
        }
        
        num_colors_target = clustering.centroids.size() / dim;
        if (num_colors_target == 0) {
            std::cerr << "Error: FAISS k-means (GPU path) resulted in 0 valid centroids after division by dim." << std::endl;
            throw std::runtime_error("FAISS k-means (GPU path) 0 valid centroids after dim division.");
        }

        std::vector<float> h_centroids_bgr = clustering.centroids;

        // Assign labels using the found centroids on GPU
        // The GpuIndexFlatL2 created earlier is now used for search.
        // Or, if we want to be explicit, create a new one for searching centroids
        // (though 'gpu_index' from clustering.train should already hold them if update_index was true)
        // Let's create a new one to be safe if the previous one was modified by train implicitly.
        faiss::gpu::GpuIndexFlatL2 search_centroid_index(&res, dim); 
        search_centroid_index.add(num_colors_target, h_centroids_bgr.data()); 

        std::vector<faiss::idx_t> h_labels(n_pixels);
        std::vector<float> h_distances(n_pixels);
        search_centroid_index.search(n_pixels, h_samples_cpu_ptr, 1, h_distances.data(), h_labels.data());

        std::vector<Color> palette_vector_rgb;
        palette_vector_rgb.reserve(num_colors_target);
        for (int i = 0; i < num_colors_target; ++i) {
            palette_vector_rgb.push_back({
                static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[i * dim + 2]))), // R
                static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[i * dim + 1]))), // G
                static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[i * dim + 0])))  // B
            });
        }

        cv::Mat quantized_image_bgr_cpu(input_img_bgr.size(), input_img_bgr.type());
        for (long r_idx = 0; r_idx < input_img_bgr.rows; ++r_idx) {
            for (long c_idx = 0; c_idx < input_img_bgr.cols; ++c_idx) {
                long sample_idx = r_idx * input_img_bgr.cols + c_idx;
                faiss::idx_t cluster_idx = h_labels[sample_idx];
                if (cluster_idx >= 0 && cluster_idx < num_colors_target) {
                    cv::Vec3b& pixel = quantized_image_bgr_cpu.at<cv::Vec3b>(r_idx, c_idx);
                    pixel[0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[cluster_idx * dim + 0]))); // B
                    pixel[1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[cluster_idx * dim + 1]))); // G
                    pixel[2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[cluster_idx * dim + 2]))); // R
                } else {
                    if (!palette_vector_rgb.empty() && h_centroids_bgr.size() >= dim) {
                         cv::Vec3b& pixel = quantized_image_bgr_cpu.at<cv::Vec3b>(r_idx, c_idx);
                         pixel[0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[0]))); // B
                         pixel[1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[1]))); // G
                         pixel[2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, h_centroids_bgr[2]))); // R
                    } else {
                         quantized_image_bgr_cpu.at<cv::Vec3b>(r_idx, c_idx) = cv::Vec3b(0,0,0); // Default to black
                    }
                    if (cluster_idx < 0) {
                        std::cerr << "Warning: FAISS returned invalid cluster_idx " << cluster_idx << " for pixel (" << r_idx << "," << c_idx << ")" << std::endl;
                    }
                }
            }
        }
        std::cout << "FAISS GPU K-Means (using CPU data source) successful." << std::endl;
        return {quantized_image_bgr_cpu, palette_vector_rgb};

    } catch (const std::runtime_error& ex) { // Catch our custom runtime errors
        std::cerr << "Runtime Error in FAISS GPU quantization path: " << ex.what() << std::endl;
        std::cerr << "Falling back to CPU (OpenCV) K-Means." << std::endl;
    } catch (const std::exception& ex) { // Catch all other standard exceptions, including FAISS internal ones if they derive from std::exception
        std::cerr << "Standard Exception in FAISS GPU path: " << ex.what() << std::endl;
        std::cerr << "Falling back to CPU (OpenCV) K-Means." << std::endl;
    }
#endif // WITH_FAISS_GPU

    // Fallback to Original CPU Path (OpenCV k-means) or if FAISS_GPU path failed or was not compiled.
    std::cout << "Using CPU (OpenCV) K-Means path for color quantization." << std::endl;
    cv::Mat samples_bgr_cpu_fallback(input_img_bgr.total(), 3, CV_32F);
    cv::Mat input_float_bgr_cpu_fallback;
    input_img_bgr.convertTo(input_float_bgr_cpu_fallback, CV_32F);
    samples_bgr_cpu_fallback = input_float_bgr_cpu_fallback.reshape(1, input_img_bgr.total());

    if (samples_bgr_cpu_fallback.rows == 0) {
        std::cerr << "Error: No samples for OpenCV k-means." << std::endl;
        return {cv::Mat(), {}};
    }
    if (samples_bgr_cpu_fallback.rows < k) { k = samples_bgr_cpu_fallback.rows; if (k == 0) return {cv::Mat(), {}}; }
    num_colors_target = k;

    cv::Mat labels_cpu_fallback;
    cv::Mat centers_bgr_cpu_fallback;
    cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 50, 0.1);
    int attempts = (k <= 8) ? 5 : 7;

    if (k > 0) {
        cv::kmeans(samples_bgr_cpu_fallback, k, labels_cpu_fallback, criteria, attempts, cv::KMEANS_PP_CENTERS, centers_bgr_cpu_fallback);
    } else {
         std::cerr << "Error: k=0 for OpenCV kmeans." << std::endl; return {cv::Mat(), {}};
    }

    if (centers_bgr_cpu_fallback.rows == 0) {
        std::cerr << "Warning: OpenCV k-means returned 0 centers. Using average image color." << std::endl;
        if (input_img_bgr.total() > 0 && !input_img_bgr.empty()) {
            cv::Scalar avg_color_scalar_bgr = cv::mean(input_img_bgr);
            cv::Mat single_color_img(input_img_bgr.size(), input_img_bgr.type());
            single_color_img.setTo(cv::Vec3b(static_cast<unsigned char>(avg_color_scalar_bgr[0]),
                                             static_cast<unsigned char>(avg_color_scalar_bgr[1]),
                                             static_cast<unsigned char>(avg_color_scalar_bgr[2]))
                                  );
            std::vector<Color> single_palette_rgb = {{
                static_cast<unsigned char>(avg_color_scalar_bgr[2]),
                static_cast<unsigned char>(avg_color_scalar_bgr[1]),
                static_cast<unsigned char>(avg_color_scalar_bgr[0])
            }};
            num_colors_target = 1;
            return {single_color_img, single_palette_rgb};
        }
        return {cv::Mat(), {}};
    }

    num_colors_target = centers_bgr_cpu_fallback.rows;

    cv::Mat quantized_image_bgr_cpu_fallback(input_img_bgr.size(), input_img_bgr.type());
    std::vector<Color> palette_vector_rgb_fallback;
    palette_vector_rgb_fallback.reserve(centers_bgr_cpu_fallback.rows);

    for (int i = 0; i < centers_bgr_cpu_fallback.rows; ++i) {
        palette_vector_rgb_fallback.push_back({
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(i, 2)))),
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(i, 1)))),
            static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(i, 0))))
        });
    }

    if (labels_cpu_fallback.empty() || labels_cpu_fallback.rows != samples_bgr_cpu_fallback.rows ) {
        std::cerr << "Error: OpenCV kmeans failed to produce valid labels. Resulting image might be incorrect." << std::endl;
        if (!palette_vector_rgb_fallback.empty()) {
             quantized_image_bgr_cpu_fallback.setTo(cv::Vec3b(
                 palette_vector_rgb_fallback[0].b,
                 palette_vector_rgb_fallback[0].g,
                 palette_vector_rgb_fallback[0].r
             ));
        } else {
            quantized_image_bgr_cpu_fallback.setTo(cv::Vec3b(0,0,0));
        }
    } else {
        int* plabels = labels_cpu_fallback.ptr<int>(0);
        for (int r_idx = 0; r_idx < quantized_image_bgr_cpu_fallback.rows; ++r_idx) {
            for (int c_idx = 0; c_idx < quantized_image_bgr_cpu_fallback.cols; ++c_idx) {
                int sample_idx = r_idx * quantized_image_bgr_cpu_fallback.cols + c_idx;
                if (sample_idx < labels_cpu_fallback.rows && plabels) {
                    int cluster_idx = plabels[sample_idx];
                    if (cluster_idx >=0 && cluster_idx < centers_bgr_cpu_fallback.rows) {
                        cv::Vec3b& pixel = quantized_image_bgr_cpu_fallback.at<cv::Vec3b>(r_idx, c_idx);
                        pixel[0] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(cluster_idx, 0))));
                        pixel[1] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(cluster_idx, 1))));
                        pixel[2] = static_cast<unsigned char>(std::max(0.0f, std::min(255.0f, centers_bgr_cpu_fallback.at<float>(cluster_idx, 2))));
                    } else {
                         std::cerr << "Warning: OpenCV Kmeans returned invalid cluster_idx " << cluster_idx << " for pixel (" << r_idx << "," << c_idx << ")" << std::endl;
                         if (!palette_vector_rgb_fallback.empty()) {
                             cv::Vec3b& pixel = quantized_image_bgr_cpu_fallback.at<cv::Vec3b>(r_idx, c_idx);
                             pixel[0] = palette_vector_rgb_fallback[0].b;
                             pixel[1] = palette_vector_rgb_fallback[0].g;
                             pixel[2] = palette_vector_rgb_fallback[0].r;
                         } else {
                             quantized_image_bgr_cpu_fallback.at<cv::Vec3b>(r_idx, c_idx) = cv::Vec3b(0,0,0);
                         }
                    }
                } else {
                    std::cerr << "Warning: sample_idx out of bounds for labels_cpu_fallback. Pixel (" << r_idx << "," << c_idx << ")" << std::endl;
                    quantized_image_bgr_cpu_fallback.at<cv::Vec3b>(r_idx, c_idx) = cv::Vec3b(0,0,0);
                }
            }
        }
    }
    return {quantized_image_bgr_cpu_fallback, palette_vector_rgb_fallback};
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
    cv::Mat raw_img_rgb_data_wrapper(height, width, CV_8UC3, const_cast<unsigned char*>(raw_bitmap_data_rgb_ptr));
    cv::Mat raw_img_bgr;
    cv::cvtColor(raw_img_rgb_data_wrapper, raw_img_bgr, cv::COLOR_RGB2BGR);

    if (raw_img_bgr.empty()) {
        std::cerr << "Error: Input image data is empty or invalid after BGR conversion." << std::endl;
        return "<svg><text fill='red'>Error: Invalid input image.</text></svg>";
    }

    int actual_num_colors = num_colors_hint;
    auto quantization_result = perform_color_quantization_cpp(raw_img_bgr, actual_num_colors);

    cv::Mat quantized_img_bgr = quantization_result.first;
    std::vector<Color> palette_rgb = quantization_result.second;

    if (quantized_img_bgr.empty() || palette_rgb.empty()) {
        std::cerr << "Error: Color quantization failed." << std::endl;
        return "<svg><text fill='red'>Error: Color quantization failed.</text></svg>";
    }

    std::vector<SvgFeature> all_features;
    cv::Point2f image_center(static_cast<float>(quantized_img_bgr.cols) / 2.0f, static_cast<float>(quantized_img_bgr.rows) / 2.0f);
    double max_dist_from_center = cv::norm(cv::Point2f(0,0) - image_center);

    for (const auto& pal_color_rgb : palette_rgb) {
        cv::Vec3b target_cv_color_bgr(pal_color_rgb.b, pal_color_rgb.g, pal_color_rgb.r);
        cv::Mat mask;
        cv::inRange(quantized_img_bgr, target_cv_color_bgr, target_cv_color_bgr, mask);

        if (cv::countNonZero(mask) == 0) continue;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::string hex_color_str = compress_hex_color_cpp(pal_color_rgb.r, pal_color_rgb.g, pal_color_rgb.b);

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

    std::stringstream svg_ss;
    int viewbox_w = raw_img_bgr.cols;
    int viewbox_h = raw_img_bgr.rows;
    int svg_attr_width = (original_svg_width > 0) ? original_svg_width : viewbox_w;
    int svg_attr_height = (original_svg_height > 0) ? original_svg_height : viewbox_h;

    svg_ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_attr_width
           << "\" height=\"" << svg_attr_height
           << "\" viewBox=\"0 0 " << viewbox_w << " " << viewbox_h << "\">";

    cv::Scalar avg_color_bgr_scalar = cv::mean(raw_img_bgr);
    std::string bg_hex_color = compress_hex_color_cpp(
        static_cast<unsigned char>(avg_color_bgr_scalar[2]),
        static_cast<unsigned char>(avg_color_bgr_scalar[1]),
        static_cast<unsigned char>(avg_color_bgr_scalar[0])
    );
    svg_ss << "<rect width=\"" << viewbox_w
           << "\" height=\"" << viewbox_h
           << "\" fill=\"" << bg_hex_color << "\"/>";
    
    size_t features_rendered = 0;
    for (const auto& feature : all_features) {
        if (max_features_to_render > 0 && features_rendered >= static_cast<size_t>(max_features_to_render)) break;
        if (feature.points.empty()) continue;
        
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
        svg_ss << "\" fill=\"" << feature.color_hex << "\"/>";
        features_rendered++;
    }

    svg_ss << "</svg>";
    return svg_ss.str();
}
