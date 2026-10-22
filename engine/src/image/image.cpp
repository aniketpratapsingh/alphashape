#include "image/image.hpp"

#include <algorithm>
#include <cmath>

#include "base/utils/error.hpp"

namespace alphashape {

FloatImage::FloatImage(int width, int height, float fill)
    : width_(width), height_(height) {
    if (width < 0 || height < 0) {
        throw ValueError("image dimensions must be non-negative");
    }
    pixels_.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), fill);
}

float FloatImage::at_clamped(int x, int y) const {
    x = std::clamp(x, 0, width_ - 1);
    y = std::clamp(y, 0, height_ - 1);
    return at(x, y);
}

float FloatImage::sample(float x, float y) const {
    const float fx = std::floor(x);
    const float fy = std::floor(y);
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const float ax = x - fx;
    const float ay = y - fy;
    const float top = at_clamped(x0, y0) * (1.0F - ax) + at_clamped(x0 + 1, y0) * ax;
    const float bottom = at_clamped(x0, y0 + 1) * (1.0F - ax) + at_clamped(x0 + 1, y0 + 1) * ax;
    return top * (1.0F - ay) + bottom * ay;
}

Image::Image(int width, int height, int channels)
    : width_(width), height_(height), channels_(channels) {
    if (width < 0 || height < 0) {
        throw ValueError("image dimensions must be non-negative");
    }
    if (channels != 1 && channels != 3) {
        throw ValueError("images have 1 or 3 channels");
    }
    data_.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                     static_cast<std::size_t>(channels),
                 0);
}

std::size_t Image::index(int x, int y, int c) const {
    return (static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x)) *
               static_cast<std::size_t>(channels_) +
           static_cast<std::size_t>(c);
}

std::uint8_t& Image::at(int x, int y, int c) { return data_[index(x, y, c)]; }

std::uint8_t Image::at(int x, int y, int c) const { return data_[index(x, y, c)]; }

FloatImage Image::to_luminance() const {
    FloatImage out(width_, height_);
    constexpr float kScale = 1.0F / 255.0F;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            float value = 0.0F;
            if (channels_ == 1) {
                value = static_cast<float>(at(x, y));
            } else {
                value = 0.2126F * static_cast<float>(at(x, y, 0)) +
                        0.7152F * static_cast<float>(at(x, y, 1)) +
                        0.0722F * static_cast<float>(at(x, y, 2));
            }
            out.at(x, y) = value * kScale;
        }
    }
    return out;
}

}  // namespace alphashape
