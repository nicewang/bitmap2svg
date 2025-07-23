#include <pybind11/pybind11.h>
#include <pybind11/stl.h>    // For std::vector conversions etc.
#include <pybind11/numpy.h>  // For NumPy array conversions
#include <iostream>
#include <vector>

// Include the C++ header for our core logic.
#include "bitmap_to_svg.h"

namespace py = pybind11;

// PYBIND11_MODULE defines the entry point for the Python C extension.
// The first argument is the module name as it will be imported in Python.
PYBIND11_MODULE(bitmap2svg_core, m) {
    m.doc() = "Pybind11 wrapper for C++ bitmap to SVG converter with internal color quantization, OpenCV enhancements, and optimization parameters.";

    // Bind the main C++ conversion function.
    m.def("convert_bitmap_to_svg_cpp",
          [](
            py::array_t<unsigned char, py::array::c_style | py::array::forcecast> raw_bitmap_data_rgb,
            int num_colors_hint,
            double simplification_epsilon_factor, // New parameter
            double min_contour_area,             // New parameter
            int max_features_to_render,          // New parameter
            py::object original_width_py,
            py::object original_height_py
          ) {
        // Request buffer information from the NumPy array.
        py::buffer_info info = raw_bitmap_data_rgb.request();

        // Validate dimensions: expecting a 3D array (height x width x 3 channels).
        if (info.ndim != 3) {
            throw py::value_error("Input raw_bitmap_data_rgb must be a 3D NumPy array (height x width x channels). Shape[0]=height, Shape[1]=width, Shape[2]=channels.");
        }
        if (info.shape[2] != 3) {
             throw py::value_error("Input raw_bitmap_data_rgb must have 3 channels (RGB). Shape[2] was " + std::to_string(info.shape[2]));
        }
        if (info.format != py::format_descriptor<unsigned char>::format()) {
            throw py::value_error("Input NumPy array must be of type unsigned char (uint8).");
        }
        if (info.ptr == nullptr) {
            throw py::value_error("Input NumPy array data pointer is null.");
        }
        if (info.itemsize != sizeof(unsigned char)) {
            throw py::value_error("Input NumPy array itemsize does not match unsigned char.");
        }

        int height = static_cast<int>(info.shape[0]);
        int width = static_cast<int>(info.shape[1]);
        const unsigned char* data_ptr = static_cast<const unsigned char*>(info.ptr);

        size_t expected_elements = static_cast<size_t>(height) * width * 3;
        if (info.size < expected_elements) {
            size_t actual_total_bytes = info.size * info.itemsize;
            size_t expected_total_bytes = expected_elements * info.itemsize;
            throw py::value_error(
                "Input NumPy array buffer is too small. "
                "Buffer has " + std::to_string(info.size) + " elements (total " + std::to_string(actual_total_bytes) + " bytes), "
                "but " + std::to_string(expected_elements) + " elements (total " + std::to_string(expected_total_bytes) + " bytes) are expected based on shape."
            );
        }

        // Handle optional original dimensions passed from Python.
        int call_original_svg_width = -1; // Default for C++ function if None
        if (!original_width_py.is_none()) {
            try {
                call_original_svg_width = original_width_py.cast<int>();
            } catch (const py::cast_error& e) {
                throw py::value_error("Invalid value for original_width_py, expected int or None.");
            }
        }
        int call_original_svg_height = -1; // Default for C++ function if None
        if (!original_height_py.is_none()) {
             try {
                call_original_svg_height = original_height_py.cast<int>();
            } catch (const py::cast_error& e) {
                throw py::value_error("Invalid value for original_height_py, expected int or None.");
            }
        }

        // Call the core C++ function
        return bitmapToSvg_with_internal_quantization(
            data_ptr,
            width,
            height,
            num_colors_hint,
            simplification_epsilon_factor,
            min_contour_area,
            max_features_to_render,
            call_original_svg_width,  // Corresponds to original_svg_width in C++ func
            call_original_svg_height  // Corresponds to original_svg_height in C++ func
        );

    }, py::arg("raw_bitmap_data_rgb"),
       py::arg("num_colors_hint") = 0, // Default: adaptive color selection in C++
       py::arg("simplification_epsilon_factor") = 0.015, // Adjusted default (was 0.005 hardcoded, 0.0075 or 0.015 reasonable start)
       py::arg("min_contour_area") = 30.0,              // Default minimum area
       py::arg("max_features_to_render") = 0,           // Default: 0 means unlimited features
       py::arg("original_width_py") = py::none(),       // Default to None if not provided
       py::arg("original_height_py") = py::none(),      // Default to None if not provided
       "Converts a raw RGB bitmap (3D NumPy array HxWx3) to an SVG string.\n"
       "Includes color quantization, feature extraction, and optimization parameters.\n"
       " - num_colors_hint: Target colors (<=0 for adaptive).\n"
       " - simplification_epsilon_factor: Controls polygon simplification (higher means more simplification).\n"
       " - min_contour_area: Minimum area for a polygon to be included.\n"
       " - max_features_to_render: Max number of polygons (0 for unlimited).\n"
       " - original_width_py/original_height_py: Optional SVG header dimensions."
    );
}
