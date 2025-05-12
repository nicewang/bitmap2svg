#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>

// Include Potrace header
#include <potracelib.h>

// Include pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // To handle std::vector

// Helper function to create potrace_bitmap_t (keep as is)
potrace_bitmap_t *create_potrace_bitmap(int width, int height, const uint8_t* pixels, int channels, const std::vector<uint8_t>& target_color, int threshold = 128) {
    potrace_bitmap_t *bm = potrace_bitmap_new(width, height);
    if (!bm) return NULL;

    // Potrace stride calculation is correct
    // int stride = (width + 31) / 32 * 4; // Potrace stride is bytes per row, multiple of 4 - Not needed for PUT macro

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool is_target = false;
            int pixel_offset = (y * width + x) * channels;

            if (channels == 1) { // Grayscale
                if (pixels[pixel_offset] >= threshold) {
                     is_target = true;
                }
            } else if (channels >= 3) { // Color (RGB or RGBA)
                 if (channels == 4 && pixels[pixel_offset+3] < 128) {
                     is_target = false; // Treat low alpha as transparent (background)
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

// Helper to convert a Potrace path structure into an SVG path 'd' attribute string (keep as is)
std::string path_to_svg_d(const potrace_path_t *path) {
    std::ostringstream d;
    d << std::fixed << std::setprecision(3); // Optional: control float precision

    for (const potrace_path_t *subpath = path; subpath; subpath = subpath->next) {
        d << "M " << subpath->curve.v[0].x << "," << subpath->curve.v[0].y;

        for (int i = 0; i < subpath->curve.n; ++i) {
            const potrace_curve_t *curve = &subpath->curve;
            int j = (i + 1) % curve->n;

            switch (curve->tag[j]) {
                case POTRACE_CORNER:
                    d << " L " << curve->v[j].x << "," << curve->v[j].y;
                    break;
                case POTRACE_CURVETO:
                    d << " C " << curve->c[j].x << "," << curve->c[j].y
                      << " " << curve->c[j].m.x << "," << curve->c[j].m.y
                      << " " << curve->v[j].x << "," << curve->v[j].y;
                    break;
                default:
                    break;
            }
        }
        d << " Z";
    }
    return d.str();
}

// Main conversion logic adapted for pybind11
std::string convert_image_to_svg_impl(int width, int height, const uint8_t* pixels, int channels) {
    if (!pixels || width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
        return "<svg></svg>"; // Return empty or error SVG
    }

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";

    if (channels == 1) {
        potrace_bitmap_t *bm = create_potrace_bitmap(width, height, pixels, channels, {}, 128);
        if (!bm) {
            return "<svg></svg>"; // Error
        }

        potrace_param_t *param = potrace_init();
        potrace_state_t *st = potrace_trace(param, bm);

        if (st && st->path) {
            std::string d_attr = path_to_svg_d(st->path);
            svg << "<path d=\"" << d_attr << "\" fill=\"black\" stroke=\"none\" />\n";
        }

        potrace_bitmap_free(bm);
        potrace_param_free(param);
        if (st) potrace_state_free(st);

    } else if (channels >= 3) {
        std::set<std::vector<uint8_t>> unique_colors;
        for(int y=0; y<height; ++y) {
            for(int x=0; x<width; ++x) {
                int pixel_offset = (y * width + x) * channels;
                if (channels == 4 && pixels[pixel_offset + 3] < 128) continue;

                std::vector<uint8_t> color(3);
                color[0] = pixels[pixel_offset];
                color[1] = pixels[pixel_offset+1];
                color[2] = pixels[pixel_offset+2];
                unique_colors.insert(color);
            }
        }

        for(const auto& color : unique_colors) {
            potrace_bitmap_t *bm = potrace_bitmap_new(width, height);
             if (!bm) continue;

            for (int y = 0; y < height; ++y) {
                 for (int x = 0; x < width; ++x) {
                     int pixel_offset = (y * width + x) * channels;
                     bool is_this_color = true;
                     if (pixels[pixel_offset] != color[0] ||
                          pixels[pixel_offset+1] != color[1] ||
                          pixels[pixel_offset+2] != color[2]) {
                          is_this_color = false;
                     }
                      if (channels == 4 && pixels[pixel_offset + 3] < 128) is_this_color = false;

                     if (is_this_color) {
                          POTRACE_BM_PUT(bm, x, y, 1);
                     } else {
                          POTRACE_BM_PUT(bm, x, y, 0);
                     }
                 }
             }

            potrace_param_t *param = potrace_init();
            potrace_state_t *st = potrace_trace(param, bm);

            if (st && st->path) {
                std::string d_attr = path_to_svg_d(st->path);
                std::ostringstream color_hex;
                color_hex << "#";
                color_hex << std::hex << std::setw(2) << std::setfill('0') << (int)color[0];
                color_hex << std::hex << std::setw(2) << std::setfill('0') << (int)color[1];
                color_hex << std::hex << std::setw(2) << std::setfill('0') << (int)color[2];

                svg << "<path d=\"" << d_attr << "\" fill=\"" << color_hex.str() << "\" stroke=\"none\" />\n";
            }

            potrace_bitmap_free(bm);
            potrace_param_free(param);
            if (st) potrace_state_free(st);
        }
    }

    svg << "</svg>";
    return svg.str();
}

// pybind11 wrapper
PYBIND11_MODULE(_bitmap2svg_core, m) {
    m.doc() = "Pybind11 wrapper for bitmap2svg using Potrace"; // Optional module docstring

    // Expose the C++ function to Python
    m.def("convert_image_to_svg", &convert_image_to_svg_impl,
          "Converts raw image pixel data to SVG using Potrace",
          pybind11::arg("width"),
          pybind11::arg("height"),
          pybind11::arg("pixels"), // pybind11 can handle pointer/buffer protocols
          pybind11::arg("channels"));
}
