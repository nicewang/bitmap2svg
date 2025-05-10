#include <string>
#include <fstream>
#include <sstream>
#include <vector>

extern "C" {

struct Bitmap {
    int width, height;
    std::vector<uint8_t> pixels; // 1 byte per pixel, grayscale
};

// Simplified logic: assume binary image, output each white pixel as a 1x1 SVG rectangle
std::string bitmap_to_svg(const Bitmap& bmp) {
    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << bmp.width
        << "\" height=\"" << bmp.height << "\">";

    for (int y = 0; y < bmp.height; ++y) {
        for (int x = 0; x < bmp.width; ++x) {
            if (bmp.pixels[y * bmp.width + x] < 128) continue;  // skip black pixels
            svg << "<rect x=\"" << x << "\" y=\"" << y
                << "\" width=\"1\" height=\"1\" fill=\"black\" />";
        }
    }

    svg << "</svg>";
    return svg.str();
}

// Python binding interface (receives raw pixels)
const char* convert_bitmap_to_svg(int width, int height, const uint8_t* pixels) {
    static std::string result;
    Bitmap bmp{width, height, std::vector<uint8_t>(pixels, pixels + width * height)};
    result = bitmap_to_svg(bmp);
    return result.c_str();
}

}
