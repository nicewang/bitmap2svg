#include "bitmap_to_svg.h"

#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip> // Required for std::hex, std::setw, std::setfill

#include <potracelib.h>


// Helper function to create potrace_bitmap_t from raw pixel data for a specific color mask.
// For grayscale (channels=1), it creates a binary mask based on a threshold.
// For color (channels>=3), it creates a binary mask based on exact color match (inefficient for many colors)
// or a proximity check (better). For simplicity, we'll do exact match or threshold.
potrace_bitmap_t *create_potrace_bitmap(int width, int height, const uint8_t* pixels, int channels, const std::vector<uint8_t>& target_color, int threshold = 128) {
    potrace_bitmap_t *bm = potrace_bitmap_new(width, height);
    if (!bm) return NULL;

    int stride = (width + 31) / 32 * 4; // Potrace stride is bytes per row, multiple of 4

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool is_target = false;
            int pixel_offset = (y * width + x) * channels;

            if (channels == 1) { // Grayscale
                if (pixels[pixel_offset] >= threshold) { // Use threshold for binary decision
                     is_target = true;
                }
            } else if (channels >= 3) { // Color (RGB or RGBA)
                 // Treat low alpha as transparent (background)
                 if (channels == 4 && pixels[pixel_offset+3] < 128) {
                     is_target = false;
                 } else {
                    // Simple exact RGB color match
                    if (pixels[pixel_offset] == target_color[0] &&
                        pixels[pixel_offset+1] == target_color[1] &&
                        pixels[pixel_offset+2] == target_color[2]) {
                        is_target = true;
                    }
                 }
            }

            if (is_target) {
                POTRACE_BM_PUT(bm, x, y, 1); // Set pixel to foreground
            } else {
                POTRACE_BM_PUT(bm, x, y, 0); // Set pixel to background
            }
        }
    }
    return bm;
}

// Helper to convert a Potrace path structure into an SVG path 'd' attribute string.
// Potrace provides paths composed of straight segments and cubic Bezier curves.
std::string path_to_svg_d(const potrace_path_t *path) {
    std::ostringstream d;

    for (const potrace_path_t *subpath = path; subpath; subpath = subpath->next) {
        // Start of a new subpath (vertex v[0] of this subpath)
        // Potrace paths start at the first vertex of the first segment.
        d << "M " << subpath->curve.v[0].x << "," << subpath->curve.v[0].y;

        // Iterate through the segments in this subpath
        for (int i = 0; i < subpath->curve.n; ++i) {
            const potrace_curve_t *curve = &subpath->curve;
            int j = (i + 1) % curve->n; // Index of the *next* vertex, which is the end of the current segment

            switch (curve->tag[j]) {
                case POTRACE_CORNER:
                    // Straight line segment from v[i] to v[j]
                    d << " L " << curve->v[j].x << "," << curve->v[j].y;
                    break;
                case POTRACE_CURVETO:
                    // Cubic Bezier curve from v[i] to v[j] with control points c[j] and c[j].m
                    // SVG Cubic Bezier command: C cx1 cy1 cx2 cy2 x y
                    d << " C " << curve->c[j].x << "," << curve->c[j].y
                      << " " << curve->c[j].m.x << "," << curve->c[j].m.y
                      << " " << curve->v[j].x << "," << curve->v[j].y;
                    break;
                default:
                    // Should not happen with standard Potrace output
                    // std::cerr << "Warning: Unexpected Potrace curve tag." << std::endl;
                    break;
            }
        }
        // Close the subpath
        d << " Z";
    }
    return d.str();
}


extern "C" {

// Python binding interface
// Takes raw pixel data (e.g., RGBA from PIL), width, height, and number of channels.
// Returns SVG code as a C-style string.
// NOTE: Uses a static string for the result, which is NOT thread-safe.
const char* convert_image_to_svg(int width, int height, const uint8_t* pixels, int channels) {
    static std::string result_svg; // Static buffer for the result

    if (!pixels || width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
         result_svg = "<svg></svg>"; // Return empty or error SVG
         return result_svg.c_str();
    }

    std::ostringstream svg;
    // Use viewBox to maintain aspect ratio if displayed size differs
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";

    // --- Color Handling / Layer Processing ---
    // This example uses a very basic color handling:
    // 1. For grayscale, it traces the foreground based on a threshold.
    // 2. For color, it attempts to find unique colors and trace each as a separate layer.
    //    Finding unique colors by iterating through pixels is inefficient for photos.
    //    A production-ready version should integrate proper color quantization (e.g., K-Means).

    if (channels == 1) {
        // Grayscale image: create one binary layer based on threshold
        potrace_bitmap_t *bm = create_potrace_bitmap(width, height, pixels, channels, {}, 128); // Use default threshold
        if (!bm) {
            result_svg = "<svg></svg>"; return result_svg.c_str(); // Error
        }

        // Trace the bitmap
        potrace_param_t *param = potrace_init();
        // Adjust Potrace parameters for desired quality/simplification
        // param->alphamax = 1.0; // Default 1.0: maximum segment length in the curve
        // param->opttolerance = 0.2; // Default 0.2: curve fitting tolerance
        // param->turnpolicy = POTRACE_TURNPOLICY_SLIGHTLYRIGHT; // How to handle ambiguous corners

        potrace_state_t *st = potrace_trace(param, bm);

        if (st && st->path) {
            std::string d_attr = path_to_svg_d(st->path);
             // Assume grayscale foreground should be black in SVG
            svg << "<path d=\"" << d_attr << "\" fill=\"black\" stroke=\"none\" />\n";
        }

        // Cleanup
        potrace_bitmap_free(bm);
        potrace_param_free(param);
        if (st) potrace_state_free(st);

    } else if (channels >= 3) {
        // Color image: attempt to process color layers.
        // WARNING: This unique color finding is very inefficient for images with many colors!
        std::set<std::vector<uint8_t>> unique_colors;
        for(int y=0; y<height; ++y) {
            for(int x=0; x<width; ++x) {
                int pixel_offset = (y * width + x) * channels;
                // Treat low alpha as background/transparent
                if (channels == 4 && pixels[pixel_offset + 3] < 128) continue;

                std::vector<uint8_t> color(3); // Store RGB only
                color[0] = pixels[pixel_offset];
                color[1] = pixels[pixel_offset+1];
                color[2] = pixels[pixel_offset+2];
                unique_colors.insert(color);
            }
        }

        // For each unique color, create a mask and trace
        for(const auto& color : unique_colors) {
            // Create binary bitmap for this color layer (foreground is this exact color, background is anything else or transparent)
             potrace_bitmap_t *bm = potrace_bitmap_new(width, height);
             if (!bm) continue; // Error creating bitmap

             for (int y = 0; y < height; ++y) {
                 for (int x = 0; x < width; ++x) {
                     int pixel_offset = (y * width + x) * channels;
                     bool is_this_color = true;
                     // Check RGB channels
                     if (pixels[pixel_offset] != color[0] ||
                         pixels[pixel_offset+1] != color[1] ||
                         pixels[pixel_offset+2] != color[2]) {
                         is_this_color = false;
                     }
                     // Check alpha if present
                      if (channels == 4 && pixels[pixel_offset + 3] < 128) is_this_color = false; // Transparent is background

                     if (is_this_color) {
                          POTRACE_BM_PUT(bm, x, y, 1);
                     } else {
                          POTRACE_BM_PUT(bm, x, y, 0);
                     }
                 }
             }

            // Trace the bitmap for this color layer
            potrace_param_t *param = potrace_init();
            // Adjust Potrace parameters if needed per layer (usually not)

            potrace_state_t *st = potrace_trace(param, bm);

            if (st && st->path) {
                std::string d_attr = path_to_svg_d(st->path);
                // Convert color vector to hex string (RGB)
                std::ostringstream color_hex;
                color_hex << "#";
                color_hex << std::hex << std::setw(2) << std::setfill('0') << (int)color[0];
                color_hex << std::hex << std::setw(2) << std::setfill('0') << (int)color[1];
                color_hex << std::hex << std::setw(2) << std::setfill('0') << (int)color[2];

                svg << "<path d=\"" << d_attr << "\" fill=\"" << color_hex.str() << "\" stroke=\"none\" />\n";
            }

            // Cleanup
            potrace_bitmap_free(bm);
            potrace_param_free(param);
            if (st) potrace_state_free(st);
        }
    }


    svg << "</svg>";
    result_svg = svg.str();
    return result_svg.c_str();
}

} // extern "C"
