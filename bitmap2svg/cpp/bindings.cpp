#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/bytes.h>
#include "bitmap_to_svg.cpp"

namespace py = pybind11;

std::string convert_image_to_svg_pybind(int width, int height, py::bytes pixel_bytes, int channels) {
    const char* pixels_ptr = pixel_bytes.c_str();
    size_t size = py::len(pixel_bytes);

    if (size != (size_t)width * height * channels) {
         throw py::value_error("Pixel data size mismatch: expected " + std::to_string((size_t)width * height * channels) + ", got " + std::to_string(size));
    }

    const uint8_t* pixels = reinterpret_cast<const uint8_t*>(pixels_ptr);

    return convert_image_to_svg_core(width, height, pixels, channels);
}

PYBIND11_MODULE(_tracer, m) {
    m.doc() = "Pybind11 plugin for bitmap to SVG tracing.";

    m.def("convert_image_to_svg", &convert_image_to_svg_pybind,
          "Convert image pixels to SVG using Potrace.");
}
