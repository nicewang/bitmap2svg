#pragma once

#include <string>
#include <cstdint>

std::string convert_image_to_svg_core(int width, int height, const uint8_t* pixels, int channels);
