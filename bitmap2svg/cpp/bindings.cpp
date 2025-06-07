#include <pybind11/pybind11.h>
#include <pybind11/stl.h>    // For std::vector conversions etc.
#include <pybind11/numpy.h>  // For NumPy array conversions
#include <iostream>
#include <vector>
#include <string>

// Include the C++ header for core logic.
#include "bitmap_to_svg.h" 

namespace py = pybind11;

// PYBIND11_MODULE defines the entry point for the Python C extension.
// The first argument is the module name as it will be imported in Python.
PYBIND11_MODULE(bitmap2svg_core, m) {
    m.doc() = "Pybind11 wrapper for C++ bitmap to SVG converter. "
              "Features internal color quantization (CPU or GPU if available), "
              "OpenCV enhancements, and various optimization parameters.";

    // Bind the main C++ conversion function.
    // The Python-exposed name is "convert_bitmap_to_svg_cpp".
    m.def("convert_bitmap_to_svg_cpp",
          [](
            py::array_t<unsigned char, py::array::c_style | py::array::forcecast> raw_bitmap_data_rgb,
            int num_colors_hint,
            double simplification_epsilon_factor,
            double min_contour_area,
            int max_features_to_render,
            py::object original_width_py,  // Using py::object to allow None
            py::object original_height_py  // Using py::object to allow None
          ) -> std::string { // Explicit return type for clarity
        // Request buffer information from the NumPy array.
        py::buffer_info info = raw_bitmap_data_rgb.request();

        // --- Validate NumPy array properties ---
        if (info.ndim != 3) {
            throw py::value_error(
                "Input 'raw_bitmap_data_rgb' must be a 3D NumPy array (height x width x channels). "
                "Received ndim = " + std::to_string(info.ndim) + "."
            );
        }
        if (info.shape[2] != 3) {
             throw py::value_error(
                "Input 'raw_bitmap_data_rgb' must have 3 channels (RGB). "
                "Received shape[2] (channels) = " + std::to_string(info.shape[2]) + "."
            );
        }
        if (info.format != py::format_descriptor<unsigned char>::format()) {
            // This check ensures the dtype is uint8.
            throw py::value_error("Input NumPy array must be of type 'unsigned char' (uint8).");
        }
        if (info.ptr == nullptr) {
            throw py::value_error("Input NumPy array data pointer is null. This should not happen if array is valid.");
        }

        // Extract dimensions and data pointer.
        int height = static_cast<int>(info.shape[0]);
        int width = static_cast<int>(info.shape[1]);
        const unsigned char* data_ptr = static_cast<const unsigned char*>(info.ptr);

        if (width <= 0 || height <= 0) {
            throw py::value_error(
                "Input NumPy array dimensions (height, width) must be positive. "
                "Received height=" + std::to_string(height) + ", width=" + std::to_string(width) + "."
            );
        }

        size_t expected_elements = static_cast<size_t>(height) * static_cast<size_t>(width) * 3;
        if (static_cast<size_t>(info.size) < expected_elements) {
            throw py::value_error(
                "Input NumPy array buffer is smaller than expected from its shape. "
                "Buffer has " + std::to_string(info.size) + " elements, "
                "but " + std::to_string(expected_elements) + " elements are expected based on shape (" +
                std::to_string(height) + "x" + std::to_string(width) + "x3)."
            );
        }

        int call_original_svg_width = 0;
        if (!original_width_py.is_none()) {
            try {
                call_original_svg_width = original_width_py.cast<int>();
                if (call_original_svg_width <= 0) {
                    // If user explicitly provides a non-positive width, it's an invalid input
                    // for this binding layer's convention (use None for default C++ behavior).
                    throw py::value_error(
                        "If 'original_width_py' is provided (i.e., not None), "
                        "it must be a positive integer. Pass None to use processed image width."
                    );
                }
            } catch (const py::cast_error& e) {
                throw py::value_error("Invalid value for 'original_width_py': must be an integer or None.");
            }
        }

        int call_original_svg_height = 0;
        if (!original_height_py.is_none()) {
             try {
                call_original_svg_height = original_height_py.cast<int>();
                if (call_original_svg_height <= 0) {
                    throw py::value_error(
                        "If 'original_height_py' is provided (i.e., not None), "
                        "it must be a positive integer. Pass None to use processed image height."
                    );
                }
            } catch (const py::cast_error& e) {
                throw py::value_error("Invalid value for 'original_height_py': must be an integer or None.");
            }
        }

        return bitmapToSvg_with_internal_quantization(
            data_ptr,
            width,
            height,
            num_colors_hint,
            simplification_epsilon_factor,
            min_contour_area,
            max_features_to_render,
            call_original_svg_width,
            call_original_svg_height
        );

    }, py::arg("raw_bitmap_data_rgb"),
       py::arg("num_colors_hint") = 0,
       py::arg("simplification_epsilon_factor") = 0.009, // Matched to bitmap_to_svg.h default
       py::arg("min_contour_area") = 10.0,               // Matched to bitmap_to_svg.h default
       py::arg("max_features_to_render") = 0,
       py::arg("original_width_py") = py::none(),
       py::arg("original_height_py") = py::none(),
       R"pbdoc(
        Converts a raw RGB bitmap (3D NumPy array HxWx3 of uint8) to an SVG string.

        This function is the C++ core of the bitmap-to-SVG conversion process.
        It includes adaptive color quantization, contour detection, polygon simplification,
        and various optimization parameters to control the output SVG.
        The underlying implementation may use GPU acceleration if available and compiled with support.

        Args:
            raw_bitmap_data_rgb: 3D NumPy array (uint8, HxWx3) representing the RGB image.
            num_colors_hint: Target number of colors for quantization. If <= 0, an
                             adaptive number of colors will be chosen by the C++ backend.
            simplification_epsilon_factor: Factor for cv::approxPolyDP polygon simplification.
                                           Smaller values mean less simplification.
            min_contour_area: Minimum area for a detected contour to be included.
            max_features_to_render: Maximum number of polygon features to render.
                                    If 0, all important features are rendered (respecting
                                    SVG size constraints).
            original_width_py: Optional. If an integer is provided, it must be positive and
                               sets the 'width' attribute of the SVG root tag.
                               If None, the processed image width is used by the C++ backend.
            original_height_py: Optional. If an integer is provided, it must be positive and
                                sets the 'height' attribute of the SVG root tag.
                                If None, the processed image height is used by the C++ backend.

        Returns:
            A string containing the generated SVG code.

        Raises:
            ValueError: If input arguments are invalid (e.g., wrong NumPy array shape/type,
                        non-positive integer for optional dimensions when not None).
            RuntimeError: For errors during C++ processing.
       )pbdoc"
    );
}
