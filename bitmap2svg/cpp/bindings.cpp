#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // To handle std::vector conversions
#include <pybind11/numpy.h> // To handle NumPy arrays
#include <iostream> 
#include <vector> 

// Include the C++ header for our core logic.
#include "cpp_svg_converter.h"

namespace py = pybind11;

// Define the Python module and bind the C++ functions.
// The module name here must match the extension name specified in setup.py.
// We use bitmap2svg_core to match our setup.py and internal module naming.
PYBIND11_MODULE(bitmap2svg_core, m) {
    std::cerr << "PYBIND11_MODULE bitmap2svg_core initializing..." << std::endl;
    m.doc() = "Pybind11 wrapper for C++ bitmap to SVG converter"; // Optional module docstring

    // Bind the Color struct so it can be used in Python.
    py::class_<Color>(m, "Color")
        .def(py::init<unsigned char, unsigned char, unsigned char>())
        .def_readwrite("r", &Color::r)
        .def_readwrite("g", &Color::g)
        .def_readwrite("b", &Color::b);

    // Bind the main conversion function, named 'convert' in Python.
    // It takes a NumPy array (RGB) and a list of Color objects (palette).
    m.def("convert", [](py::array_t<unsigned char> bitmap_data, const std::vector<Color>& palette) {
        std::cerr << "C++ convert (bitmap_to_svg_cpp) called!" << std::endl;

        // Request a contiguous buffer from the NumPy array.
        py::buffer_info info = bitmap_data.request();

        // Ensure the buffer has the expected dimensions and format (height x width x 3 for RGB).
        // Our C++ function expects RGB data.
        if (info.ndim != 3 || info.shape[2] != 3) {
             throw py::value_error("Input bitmap_data must be a 3D NumPy array (height x width x 3) of unsigned chars.");
        }

        int height = info.shape[0];
        int width = info.shape[1];
        // Get the pointer to the raw data.
        auto* data_ptr = static_cast<const unsigned char*>(info.ptr);

        // Call the core C++ function with data, dimensions, and palette.
        return bitmapToSvg(data_ptr, width, height, palette);
    }, py::arg("bitmap_data"), py::arg("palette"),
       "Convert an RGB bitmap (3D numpy array) and color palette to SVG string");

    std::cerr << "PYBIND11_MODULE bitmap2svg_core initialized successfully." << std::endl;
}
