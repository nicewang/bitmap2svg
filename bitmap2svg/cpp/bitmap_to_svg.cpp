#include <string>
#include <vector>
#include <sstream>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>

#include <potracelib.h>

potrace_bitmap_t* create_potrace_bitmap(int width, int height, const uint8_t* pixels, int channels, const std::vector<uint8_t>& target_color, int threshold) {
    potrace_bitmap_t* bm = potrace_bitmap_new(width, height); 
    if (!bm) return nullptr;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool is_target = false;
            int offset = (y * width + x) * channels;

            if (channels == 1) {
                is_target = pixels[offset] >= threshold;
            } else if (channels >= 3) {
                if (channels == 4 && pixels[offset + 3] < 128) {
                     is_target = false;
                } else {
                    is_target = (pixels[offset]     == target_color[0] &&
                                 pixels[offset + 1] == target_color[1] &&
                                 pixels[offset + 2] == target_color[2]);
                }
            }
            POTRACE_BM_PUT(bm, x, y, is_target ? 1 : 0);
        }
    }
    return bm;
}

std::string path_to_svg_d(const potrace_path_t* path) {
    std::ostringstream d;
    d << std::fixed << std::setprecision(3); 

    for (const potrace_path_t* subpath = path; subpath; subpath = subpath->next) {
        const potrace_curve_t* curve = &subpath->curve;
        int n = curve->n; 
        if (n == 0) continue; 
        d << "M " << curve->c[n - 1][2].x << "," << curve->c[n - 1][2].y;

        for (int i = 0; i < n; ++i) {
            if (curve->tag[i] == POTRACE_CORNER) {
                d << " L " << curve->c[i][1].x << "," << curve->c[i][1].y;
            } else if (curve->tag[i] == POTRACE_CURVETO) {
                d << " C " << curve->c[i][0].x << "," << curve->c[i][0].y
                  << " " << curve->c[i][1].x << "," << curve->c[i][1].y
                  << " " << curve->c[i][2].x << "," << curve->c[i][2].y;
            }
        }
        d << " Z"; 
    }

    return d.str();
}


std::string convert_image_to_svg_core(int width, int height, const uint8_t* pixels, int channels) {
    if (!pixels || width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
        return "<svg></svg>"; 
    }

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";

    if (channels == 1) {
        auto* bm = create_potrace_bitmap(width, height, pixels, channels, {}, 128);
        if (!bm) return "<svg></svg>"; 

        auto* param = potrace_param_default();
        if (!param) {
             potrace_bitmap_delete(bm); 
             return "<svg></svg>"; 
        }

        auto* st = potrace_trace(param, bm); 

        if (st && st->status == POTRACE_STATUS_OK && st->plist) { 
            svg << "<path d=\"" << path_to_svg_d(st->plist) << "\" fill=\"black\" stroke=\"none\" />\n";
        }

        potrace_bitmap_delete(bm);
        potrace_param_free(param);
        if (st) potrace_state_free(st);

    } else {
        std::set<std::vector<uint8_t>> colors;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int offset = (y * width + x) * channels;
                if (channels == 4 && pixels[offset + 3] < 128) continue; // 跳过透明度低的像素

                std::vector<uint8_t> color = {
                    pixels[offset], pixels[offset + 1], pixels[offset + 2]
                };
                colors.insert(color);
            }
        }

        for (const auto& color : colors) {
            auto* bm = create_potrace_bitmap(width, height, pixels, channels, color, 0);
            if (!bm) continue; 

            auto* param = potrace_param_default();
             if (!param) {
                potrace_bitmap_delete(bm);
                continue; 

            auto* st = potrace_trace(param, bm); 

            if (st && st->status == POTRACE_STATUS_OK && st->plist) { 
                std::ostringstream hex;
                hex << "#" << std::hex << std::setfill('0');
                for (uint8_t c : color) {
                    hex << std::setw(2) << static_cast<int>(c);
                }

                svg << "<path d=\"" << path_to_svg_d(st->plist)
                    << "\" fill=\"" << hex.str() << "\" stroke=\"none\" />\n";
            }

            potrace_bitmap_delete(bm);
            potrace_param_free(param);
            if (st) potrace_state_free(st);
        }
    }

    svg << "</svg>";
    return svg.str();
}
