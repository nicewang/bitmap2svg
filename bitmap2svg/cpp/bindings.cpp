#include <pybind11/pybind11.h>
#include <pybind11/stl.h>    // For std::vector conversions
#include <pybind11/numpy.h>  // For NumPy array conversions
#include <iostream>
#include <vector>

// Include the C++ header for our core logic.
#include "bitmap_to_svg.h"

namespace py = pybind11;

// PYBIND11_MODULE defines the entry point for the Python C extension.
// The first argument is the module name as it will be imported in Python.
PYBIND11_MODULE(bitmap2svg_core, m) {
    m.doc() = "Pybind11 wrapper for C++ bitmap to SVG converter with internal color quantization and OpenCV enhancements.";

    // Optional: If need to expose the Color struct to Python for some reason (e.g., debugging, or if Python constructs it)
    // py::class_<Color>(m, "Color_cpp") // Renamed to avoid conflict if Python has 'Color'
    //     .def(py::init<unsigned char, unsigned char, unsigned char>())
    //     .def_readwrite("r", &Color::r)
    //     .def_readwrite("g", &Color::g)
    //     .def_readwrite("b", &Color::b)
    //     .def("__repr__", [](const Color &c) {
    //         return "<Color_cpp r=" + std::to_string(c.r) +
    //                " g=" + std::to_string(c.g) +
    //                " b=" + std::to_string(c.b) + ">";
    //     });

    // Bind the main C++ conversion function.
    // It takes a NumPy array (raw RGB image data), a hint for the number of colors,
    // and optional original dimensions for the SVG attributes.
    m.def("convert_bitmap_to_svg_cpp", // Renamed for clarity in Python if needed
          [](
            // Input NumPy array: expects a C-style contiguous array of unsigned chars.
            // py::array::forcecast allows some flexibility in input type if convertible.
            py::array_t<unsigned char, py::array::c_style | py::array::forcecast> raw_bitmap_data_rgb,
            int num_colors_hint, // Hint for C++: <=0 for adaptive, >0 for fixed number.
            py::object original_width_py,  // Original width for SVG 'width' attribute.
            py::object original_height_py  // Original height for SVG 'height' attribute.
          ) {
        // Request buffer information from the NumPy array.
        py::buffer_info info = raw_bitmap_data_rgb.request();

        // Validate dimensions: expecting a 3D array (height x width x 3 channels).
        if (info.ndim != 3 || info.shape[2] != 3) {
            throw py::value_error("Input raw_bitmap_data_rgb must be a 3D NumPy array (height x width x 3 channels) of unsigned chars.");
        }
        if (info.format != py::format_descriptor<unsigned char>::format()) {
            throw py::value_error("Input NumPy array must be of type unsigned char (uint8).");
        }


        int height = static_cast<int>(info.shape[0]); // Processed image height (from potentially resized NumPy array)
        int width = static_cast<int>(info.shape[1]);  // Processed image width
        auto* data_ptr = static_cast<const unsigned char*>(info.ptr); // Pointer to raw data

        // Handle optional original dimensions passed from Python.
        // If Python passes None, these will be <=0, and C++ will use processed dimensions for SVG attributes.
        int svg_attr_width = -1;
        if (!original_width_py.is_none()) {
            svg_attr_width = original_width_py.cast<int>();
        }
        int svg_attr_height = -1;
        if (!original_height_py.is_none()) {
            svg_attr_height = original_height_py.cast<int>();
        }
        
        // Default simplification factor, can be made a parameter if needed.
        double simplification_factor = 0.005; 

        // Call the core C++ function.
        return bitmapToSvg_with_internal_quantization(
            data_ptr,
            width,
            height,
            num_colors_hint,
            simplification_factor,
            svg_attr_width,
            svg_attr_height
        );

    }, py::arg("raw_bitmap_data_rgb"),       // Argument name in Python
       py::arg("num_colors_hint"),           // Argument name in Python
       py::arg("original_width_py") = py::none(), // Default to None if not provided
       py::arg("original_height_py") = py::none(),// Default to None if not provided
       "Converts a raw RGB bitmap (3D NumPy array) to an SVG string. "
       "Color quantization and feature extraction are performed in C++ using OpenCV. "
       "Pass num_colors_hint <= 0 for adaptive color selection."
    );
}
