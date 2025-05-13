#ifndef CPP_SVG_CONVERTER_H
#define CPP_SVG_CONVERTER_H

#include <vector>
#include <string>

// Structure to represent a color (e.g., RGB)
struct Color {
    unsigned char r, g, b;
};

// Function to convert bitmap data to SVG string
// bitmap_data: Pointer to the raw pixel data (e.g., R, G, B, R, G, B, ...)
// width: Width of the image
// height: Height of the image
// palette: Vector of colors used in the bitmap
// Returns: SVG string representation of the bitmap
std::string bitmapToSvg(const unsigned char* bitmap_data, int width, int height, const std::vector<Color>& palette);

#endif // CPP_SVG_CONVERTER_H
