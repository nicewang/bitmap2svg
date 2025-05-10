#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
extern "C" const char* convert_bitmap_to_svg(int, int, const uint8_t*);

namespace py = pybind11;

PYBIND11_MODULE(bitmap2svg_core, m) {
    m.def("convert", [](py::array_t<uint8_t> arr) {
        py::buffer_info info = arr.request();
        if (info.ndim != 2)
            throw std::runtime_error("Only 2D grayscale images are supported");

        int height = info.shape[0];
        int width = info.shape[1];
        auto* data = static_cast<uint8_t*>(info.ptr);
        return std::string(convert_bitmap_to_svg(width, height, data));
    }, "Convert a grayscale bitmap (2D numpy array) to SVG string");
}
