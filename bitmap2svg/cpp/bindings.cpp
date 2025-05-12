#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bitmap_to_svg.cpp"

namespace py = pybind11;

std::string convert_numpy_to_svg(py::array_t<uint8_t> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 3) {
        throw std::runtime_error("Image must be a 3D NumPy array (H, W, C)");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.shape[2];

    const uint8_t* data_ptr = static_cast<uint8_t*>(buf.ptr);
    return convert_image_to_svg_core(width, height, data_ptr, channels);
}

PYBIND11_MODULE(bitmap2svg_cpp, m) {
    m.def("convert_image_to_svg", &convert_numpy_to_svg, "Convert image (NumPy array) to SVG string");
}
