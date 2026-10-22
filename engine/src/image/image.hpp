#pragma once

// Image containers.
//
// Image: 8-bit, 1 or 3 interleaved channels — what codecs produce.
// FloatImage: single-channel float — what the algorithms work on (scale
// spaces, gradients, depth maps). Keeping the two separate means algorithms
// never care what format a photo arrived in.

#include <cstdint>
#include <vector>

namespace alphashape {

class FloatImage {
public:
    FloatImage() = default;
    FloatImage(int width, int height, float fill = 0.0F);

    int width() const { return width_; }
    int height() const { return height_; }
    bool empty() const { return pixels_.empty(); }

    float& at(int x, int y) { return pixels_[index(x, y)]; }
    float at(int x, int y) const { return pixels_[index(x, y)]; }

    /// Clamped read: coordinates outside the image return the nearest edge.
    float at_clamped(int x, int y) const;

    /// Bilinear sample with edge clamping, at continuous coordinates where
    /// (0,0) is the centre of the top-left pixel.
    float sample(float x, float y) const;

    std::vector<float>& pixels() { return pixels_; }
    const std::vector<float>& pixels() const { return pixels_; }

private:
    std::size_t index(int x, int y) const {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x);
    }
    int width_ = 0;
    int height_ = 0;
    std::vector<float> pixels_;
};

class Image {
public:
    Image() = default;
    Image(int width, int height, int channels);

    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    bool empty() const { return data_.empty(); }

    std::uint8_t* data() { return data_.data(); }
    const std::uint8_t* data() const { return data_.data(); }
    std::size_t size_bytes() const { return data_.size(); }

    std::uint8_t& at(int x, int y, int c = 0);
    std::uint8_t at(int x, int y, int c = 0) const;

    /// Luminance in [0, 1]. Colour uses Rec. 709 weights.
    FloatImage to_luminance() const;

private:
    std::size_t index(int x, int y, int c) const;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    std::vector<std::uint8_t> data_;
};

}  // namespace alphashape
