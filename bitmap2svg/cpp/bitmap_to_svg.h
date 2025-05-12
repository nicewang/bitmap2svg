// bitmap2svg/cpp/bitmap_to_svg.h

#ifndef BITMAP_TO_SVG_H
#define BITMAP_TO_SVG_H

#include <string>
#include <cstdint>
#include <potracelib.h>

std::string convert_image_to_svg_core(int width, int height, const uint8_t* pixels, int channels);

#endif
