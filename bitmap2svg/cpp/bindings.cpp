#include <pybind11/pybind11.h>
#include <pybind11/stl.h>    // For std::vector, std::string
#include <pybind11/numpy.h>  // For potential numpy array inputs

#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>

#include <potracelib.h>

potrace_bitmap_t* create_potrace_bitmap(int width, int height, const uint8_t* pixels, int channels, const std::vector<uint8_t>& target_color, int threshold = 0 /* Default for color mode, for grayscale it's passed explicitly*/) {
    potrace_bitmap_t* bm = potrace_bitmap_new(width, height);
    if (!bm) return nullptr;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool is_target = false;
            int offset = (y * width + x) * channels;

            if (channels == 1) {
                is_target = pixels[offset] >= threshold;
            } else if (channels >= 3) { // RGB or RGBA
                if (channels == 4 && pixels[offset + 3] < 128) { // Check alpha: if significantly transparent, treat as background
                    is_target = false;
                } else {
                    // target_color must not be empty for colored images if this branch is hit
                    if (target_color.size() >= 3) {
                         is_target = (pixels[offset]     == target_color[0] &&
                                      pixels[offset + 1] == target_color[1] &&
                                      pixels[offset + 2] == target_color[2]);
                    } else {
                        is_target = false;
                    }
                }
            }
            POTRACE_BM_PUT(bm, x, y, is_target ? 1 : 0);
        }
    }
    return bm;
}

std::string path_to_svg_d(const potrace_path_t* path) {
    std::ostringstream d;
    d << std::fixed << std::setprecision(3);

    for (const potrace_path_t* subpath = path; subpath; subpath = subpath->next) {
        if (subpath->curve.n == 0) continue; // Skip empty subpaths

        d << "M " << subpath->curve.v[0].x << "," << subpath->curve.v[0].y;
        for (int i = 0; i < subpath->curve.n; ++i) {
            const potrace_curve_t* curve = &subpath->curve;
            // j is the ENDPOINT of the current segment (from v[i] to v[j])
            int j = (i + 1) % curve->n;

            if (curve->tag[j] == POTRACE_CORNER) {
                d << " L " << curve->v[j].x << "," << curve->v[j].y;
            } else if (curve->tag[j] == POTRACE_CURVETO) {
                d << " C " << curve->c[j].x << "," << curve->c[j].y      // First control point
                  << " " << curve->c[j].m.x << "," << curve->c[j].m.y  // Second control point
                  << " " << curve->v[j].x << "," << curve->v[j].y;     // Endpoint
            }
        }
        d << " Z"; // Close path
    }
    return d.str();
}


std::string convert_image_to_svg_core(int width, int height, const uint8_t* pixels, int channels) {
    if (!pixels || width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
        return "<svg width=\"0\" height=\"0\" xmlns=\"http://www.w3.org/2000/svg\"></svg>"; // More informative empty SVG
    }

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";

    if (channels == 1) {
        // For grayscale, threshold is important. target_color is empty.
        auto* bm = create_potrace_bitmap(width, height, pixels, channels, {}, 128); // Default threshold 128
        if (!bm) {
            svg << "</svg>"; return svg.str(); // Early exit on bitmap creation failure
        }

        potrace_param_t* param = potrace_param_default(); // Use default parameters
        if (!param) {
            potrace_bitmap_free(bm);
            svg << "</svg>"; return svg.str();
        }
        // Example:
        // param->turdsize = 2;
        // param->turnpolicy = POTRACE_TURNPOLICY_MINORITY;

        potrace_state_t* st = potrace_trace(param, bm);

        if (st && st->status == POTRACE_STATUS_OK && st->plist) { // Use st->plist
            svg << "  <path d=\"" << path_to_svg_d(st->plist) << "\" fill=\"black\" stroke=\"none\" />\n";
        }

        potrace_bitmap_free(bm);
        potrace_param_free(param);
        if (st) potrace_state_free(st);

    } else { // RGB or RGBA (channels == 3 || channels == 4)
        std::set<std::vector<uint8_t>> distinct_colors;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int offset = (y * width + x) * channels;
                if (channels == 4 && pixels[offset + 3] < 128) { // Skip if significantly transparent
                    continue;
                }
                std::vector<uint8_t> current_color = {
                    pixels[offset], pixels[offset + 1], pixels[offset + 2]
                };
                distinct_colors.insert(current_color);
            }
        }

        for (const auto& color_vec : distinct_colors) {
            if (color_vec.size() < 3) continue; // Should not happen

            // The create_potrace_bitmap for color needs the target_color explicitly.
            // The threshold parameter is not used for color matching logic here.
            auto* bm = create_potrace_bitmap(width, height, pixels, channels, color_vec);
            if (!bm) continue;

            potrace_param_t* param = potrace_param_default();
            if (!param) {
                potrace_bitmap_free(bm);
                continue;
            }

            potrace_state_t* st = potrace_trace(param, bm);

            if (st && st->status == POTRACE_STATUS_OK && st->plist) {
                std::ostringstream hex_color_stream;
                hex_color_stream << "#" << std::hex << std::setfill('0');
                hex_color_stream << std::setw(2) << static_cast<int>(color_vec[0]);
                hex_color_stream << std::setw(2) << static_cast<int>(color_vec[1]);
                hex_color_stream << std::setw(2) << static_cast<int>(color_vec[2]);

                svg << "  <path d=\"" << path_to_svg_d(st->plist)
                    << "\" fill=\"" << hex_color_stream.str() << "\" stroke=\"none\" />\n";
            }

            potrace_bitmap_free(bm);
            potrace_param_free(param);
            if (st) potrace_state_free(st);
        }
    }

    svg << "</svg>";
    return svg.str();
}
// --- END: Your C++ code ---


namespace py = pybind11;

// Wrapper function to accept Python bytes or NumPy array for pixel data
std::string convert_image_to_svg_wrapper(int width, int height, py::object pixels_obj, int channels) {
    const uint8_t* pixels_ptr = nullptr;
    py::buffer_info info;

    if (py::isinstance<py::bytes>(pixels_obj)) {
        py::bytes pixels_py = pixels_obj.cast<py::bytes>();
        info = py::buffer(pixels_py).request();
    } else if (py::isinstance<py::array_t<uint8_t>>(pixels_obj)) {
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> pixels_np =
            pixels_obj.cast<py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>();
        info = pixels_np.request();
    } else {
        throw py::type_error("Pixel data must be bytes or a uint8 NumPy array.");
    }

    if (info.ptr == nullptr) {
        throw std::runtime_error("Failed to get buffer from pixels object.");
    }
    pixels_ptr = static_cast<const uint8_t*>(info.ptr);

    // Validate dimensions and size
    size_t expected_elements = static_cast<size_t>(width) * height * channels;
    // info.size is total number of elements if it's a NumPy array, or bytes if it's py::bytes.
    // info.itemsize is relevant for typed arrays. For bytes, itemsize is 1.
    size_t actual_elements = info.size * info.itemsize; // For bytes, itemsize is 1. For numpy array, itemsize is 1 (uint8_t).

    if (info.ndim != 1 && !(info.ndim == 3 && info.shape[0] == height && info.shape[1] == width && info.shape[2] == channels) ) {
         // Allow 1D flat array or HxWxC array for numpy
        if (! (py::isinstance<py::bytes>(pixels_obj) && info.ndim == 1) ) { // bytes should always be 1D buffer
            throw std::runtime_error("Pixel data must be a 1D flat array/bytes, or a HxWxC NumPy array.");
        }
    }
    
    // If numpy array is HxWxC, its total size in bytes will match expected_elements.
    // If it's a flat 1D array (or bytes), its size also matches expected_elements.
    if (actual_elements != expected_elements) {
        throw std::runtime_error(
            "Pixel data size mismatch. Expected total bytes: " + std::to_string(expected_elements) +
            ", Got: " + std::to_string(actual_elements) +
            ". Dimensions (WxHxC): " + std::to_string(width) + "x" + std::to_string(height) + "x" + std::to_string(channels)
        );
    }
    
    if (channels != 1 && channels != 3 && channels != 4) {
        throw std::invalid_argument("Channels must be 1 (grayscale), 3 (RGB), or 4 (RGBA).");
    }


    return convert_image_to_svg_core(width, height, pixels_ptr, channels);
}


PYBIND11_MODULE(_bitmap2svg_cpp, m) {
    m.doc() = R"pbdoc(
        Python bindings for C++ bitmap to SVG conversion using Potrace.
        Provides the function convert_image_to_svg.
    )pbdoc";

    m.def("convert_image_to_svg", &convert_image_to_svg_wrapper,
          "Converts bitmap image data (pixels) to an SVG string.",
          py::arg("width"),
          py::arg("height"),
          py::arg("pixels"), // py::bytes or numpy.ndarray[uint8]
          py::arg("channels"),
          R"pbdoc(
            Converts raw pixel data to an SVG string representation using Potrace.

            Args:
                width (int): Width of the image in pixels.
                height (int): Height of the image in pixels.
                pixels (Union[bytes, numpy.ndarray[numpy.uint8]]):
                    Flat byte array or NumPy array of pixel data.
                    Order should be row by row, e.g., RGBRGB... or GrayscaleValueGrayscaleValue...
                    If NumPy array, can be 1D (flat) or 3D (height, width, channels).
                channels (int): Number of color channels in the pixel data.
                    1 for grayscale.
                    3 for RGB.
                    4 for RGBA (Alpha channel is used for transparency; <128 alpha is background).

            Returns:
                str: A string containing the SVG representation of the image.
                     Returns a basic empty SVG string on internal Potrace errors or invalid input.

            Raises:
                TypeError: If pixels is not bytes or a uint8 NumPy array.
                RuntimeError: If pixel data size/dimensions are inconsistent.
                std::invalid_argument: If channels value is invalid.
          )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}