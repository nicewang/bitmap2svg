#ifndef BITMAP_TO_SVG_H
#define BITMAP_TO_SVG_H

#include <cstdint> // For uint8_t

extern "C" {
    /**
     * Converts raw pixel data to SVG using Potrace.
     *
     * Args:
     * width: Image width in pixels.
     * height: Image height in pixels.
     * pixels: Pointer to the raw pixel data (e.g., L, RGB, RGBA).
     * channels: Number of color channels (1 for grayscale, 3 for RGB, 4 for RGBA).
     *
     * Returns:
     * A C-style string containing the SVG XML code.
     * NOTE: Uses a static buffer, NOT thread-safe. The caller should copy the string if needed.
     * Returns "<svg></svg>" on error or invalid input.
     */
    const char* convert_image_to_svg(int width, int height, const uint8_t* pixels, int channels);
}

#endif // BITMAP_TO_SVG_H