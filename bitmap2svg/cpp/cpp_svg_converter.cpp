#include "cpp_svg_converter.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

// Helper function to get pixel color
Color getPixelColor(const unsigned char* data, int x, int y, int width, int height) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        // Return a default color or handle out of bounds as needed
        return {0, 0, 0}; // Black for out of bounds
    }
    int index = (y * width + x) * 3; // Assuming 3 channels (RGB)
    return {data[index], data[index + 1], data[index + 2]};
}

// Helper function to check if two colors are equal
bool colorsEqual(const Color& c1, const Color& c2) {
    return c1.r == c2.r && c1.g == c2.g && c1.b == c2.b;
}

// Basic contour tracing algorithm (simplified)
// This is a basic implementation. A more robust one would handle holes and complex shapes better.
std::vector<std::pair<int, int>> traceContour(const unsigned char* data, int width, int height, const Color& target_color, std::vector<std::vector<bool>>& visited, int start_x, int start_y) {
    std::vector<std::pair<int, int>> contour;
    int x = start_x;
    int y = start_y;
    int dx[] = {1, 0, -1, 0}; // Right, Down, Left, Up
    int dy[] = {0, 1, 0, -1};
    int dir = 0; // Start moving right

    // Find a starting point on the boundary
    bool found_start = false;
    for (int j = start_y; j < height; ++j) {
        for (int i = start_x; i < width; ++i) {
            if (colorsEqual(getPixelColor(data, i, j, width, height), target_color) &&
                (i == 0 || !colorsEqual(getPixelColor(data, i - 1, j, width, height), target_color) ||
                 j == 0 || !colorsEqual(getPixelColor(data, i, j - 1, width, height), target_color) ||
                 i == width - 1 || !colorsEqual(getPixelColor(data, i + 1, j, width, height), target_color) ||
                 j == height - 1 || !colorsEqual(getPixelColor(data, i, j + 1, width, height), target_color))) {
                x = i;
                y = j;
                found_start = true;
                break;
            }
        }
        if (found_start) break;
    }

    if (!found_start) return contour; // No contour found for this color starting here

    int current_x = x;
    int current_y = y;
    int initial_x = x;
    int initial_y = y;
    bool first_step = true;

    while ((current_x != initial_x || current_y != initial_y) || first_step) {
        first_step = false;
        contour.push_back({current_x, current_y});
        visited[current_y][current_x] = true; // Mark as visited

        // Try to find the next boundary pixel (right-hand rule)
        int next_dir = (dir + 3) % 4; // Start checking from the direction to the right of current

        bool moved = false;
        for (int i = 0; i < 4; ++i) {
            int check_dir = (next_dir + i) % 4;
            int next_x = current_x + dx[check_dir];
            int next_y = current_y + dy[check_dir];

            if (next_x >= 0 && next_x < width && next_y >= 0 && next_y < height) {
                if (colorsEqual(getPixelColor(data, next_x, next_y, width, height), target_color)) {
                     // Check if the pixel to the left of the movement direction is NOT the target color
                    int left_of_move_dir = (check_dir + 1) % 4;
                    int pixel_left_x = current_x + dx[left_of_move_dir];
                    int pixel_left_y = current_y + dy[left_of_move_dir];

                    if (pixel_left_x < 0 || pixel_left_x >= width || pixel_left_y < 0 || pixel_left_y >= height ||
                        !colorsEqual(getPixelColor(data, pixel_left_x, pixel_left_y, width, height), target_color)) {
                         current_x = next_x;
                         current_y = next_y;
                         dir = check_dir;
                         moved = true;
                         break;
                    }
                }
            }
        }
        if (!moved) {
             // Could not find next boundary pixel, might be a single pixel or error
             break;
        }
    }

    return contour;
}


// Simple curve fitting (detects horizontal/vertical lines and tries to fit arcs)
// This is a very basic attempt at curve fitting. More advanced algorithms exist.
std::string pointsToSvgPath(const std::vector<std::pair<int, int>>& points) {
    if (points.empty()) {
        return "";
    }

    std::stringstream ss;
    ss << "M " << points[0].first << "," << points[0].second;

    // Keep track of previous point and segment type
    int prev_x = points[0].first;
    int prev_y = points[0].second;
    enum SegmentType { LINE, ARC, UNKNOWN };
    SegmentType current_segment_type = UNKNOWN;

    for (size_t i = 1; i < points.size(); ++i) {
        int current_x = points[i].first;
        int current_y = points[i].second;

        int dx = current_x - prev_x;
        int dy = current_y - prev_y;

        // Simple line detection
        if (dx == 0 || dy == 0) {
            if (current_segment_type != LINE) {
                ss << " L";
                current_segment_type = LINE;
            }
            ss << " " << current_x << "," << current_y;
        } else {
            // Basic arc detection (looking for changes in direction)
            // This is a very naive approach. A proper implementation would analyze curvature.
            if (i > 1) {
                int prev_dx = prev_x - points[i-2].first;
                int prev_dy = prev_y - points[i-2].second;

                // Check if direction changed significantly
                if ((dx * prev_dx <= 0 && dy * prev_dy <= 0) && (std::abs(dx) > 1 || std::abs(dy) > 1)) {
                     if (current_segment_type != ARC) {
                        // Start a new arc segment
                        // This is a placeholder. Proper arc fitting requires more math.
                        // For now, we'll just output a line and mark it as potentially part of an arc.
                        ss << " L";
                        current_segment_type = ARC; // Mark as potential arc segment
                     }
                      ss << " " << current_x << "," << current_y;
                } else {
                    // If direction didn't change or changed slightly, treat as line
                    if (current_segment_type != LINE) {
                        ss << " L";
                        current_segment_type = LINE;
                    }
                    ss << " " << current_x << "," << current_y;
                }
            } else {
                 // First segment after M, treat as line
                 if (current_segment_type != LINE) {
                    ss << " L";
                    current_segment_type = LINE;
                 }
                 ss << " " << current_x << "," << current_y;
            }
        }

        prev_x = current_x;
        prev_y = current_y;
    }

    // Close the path if it's not empty
    if (!points.empty()) {
         ss << " Z";
    }


    return ss.str();
}


std::string bitmapToSvg(const unsigned char* bitmap_data, int width, int height, const std::vector<Color>& palette) {
    std::stringstream svg_ss;
    svg_ss << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";

    // Add a background rectangle (optional, depends on how palette[0] is used)
    // Assuming the first color in the palette is the background
     if (!palette.empty()) {
         svg_ss << "<rect width=\"" << width << "\" height=\"" << height << "\" fill=\"rgb(" << (int)palette[0].r << "," << (int)palette[0].g << "," << (int)palette[0].b << ")\"/>\n";
     }


    // Keep track of visited pixels to avoid re-tracing contours
    std::vector<std::vector<bool>> visited(height, std::vector<bool>(width, false));

    // Iterate through each color in the palette (skip background if added as rect)
    for (size_t p_idx = 0; p_idx < palette.size(); ++p_idx) {
        const Color& target_color = palette[p_idx];

        // Find all contours for this color
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Start tracing if the pixel has the target color and hasn't been visited
                // and is potentially a boundary pixel (check neighbors)
                if (colorsEqual(getPixelColor(bitmap_data, x, y, width, height), target_color) && !visited[y][x]) {
                     bool is_boundary = false;
                     for(int dy = -1; dy <= 1; ++dy) {
                         for(int dx = -1; dx <= 1; ++dx) {
                             if (dx == 0 && dy == 0) continue;
                             int nx = x + dx;
                             int ny = y + dy;
                             if (nx < 0 || nx >= width || ny < 0 || ny >= height ||
                                 !colorsEqual(getPixelColor(bitmap_data, nx, ny, width, height), target_color)) {
                                 is_boundary = true;
                                 break;
                             }
                         }
                         if(is_boundary) break;
                     }

                     if (is_boundary) {
                        std::vector<std::pair<int, int>> contour = traceContour(bitmap_data, width, height, target_color, visited, x, y);

                        if (!contour.empty()) {
                            // Convert contour points to SVG path data
                            std::string path_data = pointsToSvgPath(contour);

                            // Add path element to SVG
                            svg_ss << "<path d=\"" << path_data << "\" fill=\"rgb(" << (int)target_color.r << "," << (int)target_color.g << "," << (int)target_color.b << ")\"/>\n";
                        }
                     } else {
                         // If it's not a boundary, mark it as visited to avoid re-checking
                         visited[y][x] = true;
                     }
                }
            }
        }
    }


    svg_ss << "</svg>";

    return svg_ss.str();
}
