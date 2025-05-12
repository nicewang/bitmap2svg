#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/bytes.h>
#include <string>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include "bitmap_to_svg.h"

namespace py = pybind11;

std::string convert_image_to_svg_pybind(int width, int height, py::bytes pixel_bytes, int channels) {
    const char* pixels_ptr = pixel_bytes.c_str();
    size_t size = static_cast<size_t>(py::len(pixel_bytes));

    size_t expected_size = static_cast<size_t>(width) * height * channels;
    if (size != expected_size) {
         throw py::value_error("Pixel data size mismatch: expected " + std::to_string(expected_size) + ", got " + std::to_string(size) +
                               ", for dimensions " + std::to_string(width) + "x" + std::to_string(height) + " with " + std::to_string(channels) + " channels.");
    }

    const uint8_t* pixels = reinterpret_cast<const uint8_t*>(pixels_ptr);

    return convert_image_to_svg_core(width, height, pixels, channels);
}

PYBIND11_MODULE(_bitmap2svg_core, m) {
    m.doc() = "Pybind11 plugin for bitmap to SVG tracing using Potrace.";

    m.def("convert_image_to_svg", &convert_image_to_svg_pybind,
          "Convert image pixels (bytes) to SVG using Potrace. Expects raw pixel data as bytes (width * height * channels).");
}
