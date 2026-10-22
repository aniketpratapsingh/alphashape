#include "image/operators/downsample.hpp"

#include <cmath>

#include "base/utils/error.hpp"

namespace alphashape {

namespace {

/// Power-of-two >= 1 check, shared by both overloads.
void require_power_of_two(int factor) {
    if (factor < 1 || (factor & (factor - 1)) != 0) {
        throw ValueError("downsample_box: factor must be a power of two >= 1");
    }
}

}  // namespace

FloatImage downsample_box(const FloatImage& src, int factor) {
    require_power_of_two(factor);
    const int out_w = src.width() / factor;
    const int out_h = src.height() / factor;
    if (factor == 1) {
        return src;  // a copy
    }
    FloatImage out(out_w, out_h);
    const double inv = 1.0 / (static_cast<double>(factor) * static_cast<double>(factor));
    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            double sum = 0.0;
            for (int dy = 0; dy < factor; ++dy) {
                const int sy = oy * factor + dy;
                for (int dx = 0; dx < factor; ++dx) {
                    sum += static_cast<double>(src.at(ox * factor + dx, sy));
                }
            }
            out.at(ox, oy) = static_cast<float>(sum * inv);
        }
    }
    return out;
}

Image downsample_box(const Image& src, int factor) {
    require_power_of_two(factor);
    const int out_w = src.width() / factor;
    const int out_h = src.height() / factor;
    if (factor == 1) {
        return src;  // a copy
    }
    const int channels = src.channels();
    Image out(out_w, out_h, channels);
    const double inv = 1.0 / (static_cast<double>(factor) * static_cast<double>(factor));
    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            for (int c = 0; c < channels; ++c) {
                double sum = 0.0;
                for (int dy = 0; dy < factor; ++dy) {
                    const int sy = oy * factor + dy;
                    for (int dx = 0; dx < factor; ++dx) {
                        sum += static_cast<double>(src.at(ox * factor + dx, sy, c));
                    }
                }
                // Round half away from zero; values are non-negative so lround
                // is the intended nearest-integer.
                out.at(ox, oy, c) = static_cast<std::uint8_t>(std::lround(sum * inv));
            }
        }
    }
    return out;
}

}  // namespace alphashape
