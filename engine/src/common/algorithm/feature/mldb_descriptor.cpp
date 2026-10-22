#include "common/algorithm/feature/mldb_descriptor.hpp"

#include <array>
#include <bit>
#include <cmath>

#include "base/utils/error.hpp"

namespace alphashape {

namespace {

constexpr int kChannels = 3;  // intensity, d/du, d/dv

class BitWriter {
public:
    explicit BitWriter(Descriptor& d) : d_(d) {}
    void push(bool bit) {
        if (bit_ >= Descriptor::kBytes * 8) {
            throw ValueError("descriptor overflow");
        }
        if (bit) {
            d_.bytes[bit_ / 8] = static_cast<std::uint8_t>(d_.bytes[bit_ / 8] | (1U << (bit_ % 8)));
        }
        ++bit_;
    }
    std::size_t count() const { return bit_; }

private:
    Descriptor& d_;
    std::size_t bit_ = 0;
};

}  // namespace

Descriptor compute_mldb(const ScaleSpace& space, const Detection& det, const MldbParams& params) {
    const int n = params.samples;
    if (n % 12 != 0) {
        throw ValueError("MLDB samples per side must be divisible by 2, 3 and 4");
    }
    const FloatImage& g = space.gaussians.at(static_cast<std::size_t>(det.octave))
                              .at(static_cast<std::size_t>(det.layer));

    // Sample an n x n grid on a square aligned with the keypoint orientation.
    // Derivatives are then taken along the grid axes, so all three channels
    // are expressed in the keypoint's frame: that is the rotation invariance.
    const float radius = params.patch_radius * det.octave_sigma;
    const float c = std::cos(det.keypoint.orientation);
    const float s = std::sin(det.keypoint.orientation);
    std::vector<float> patch(static_cast<std::size_t>(n * n));
    const auto P = [&](int i, int j) -> float& { return patch[static_cast<std::size_t>(j * n + i)]; };
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const float u = ((static_cast<float>(i) + 0.5F) / static_cast<float>(n) * 2.0F - 1.0F) * radius;
            const float v = ((static_cast<float>(j) + 0.5F) / static_cast<float>(n) * 2.0F - 1.0F) * radius;
            P(i, j) = g.sample(det.octave_x + c * u - s * v, det.octave_y + s * u + c * v);
        }
    }
    std::vector<float> du(patch.size()), dv(patch.size()), mag(patch.size());
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const std::size_t k = static_cast<std::size_t>(j * n + i);
            du[k] = P(std::min(i + 1, n - 1), j) - P(std::max(i - 1, 0), j);
            dv[k] = P(i, std::min(j + 1, n - 1)) - P(i, std::max(j - 1, 0));
            mag[k] = std::sqrt(du[k] * du[k] + dv[k] * dv[k]);
        }
    }

    // Cell means for a grid of side `cells`.
    const auto cell_means = [&](int cells, const std::vector<float>& channel) {
        const int step = n / cells;
        std::vector<float> means(static_cast<std::size_t>(cells * cells), 0.0F);
        for (int cy = 0; cy < cells; ++cy) {
            for (int cx = 0; cx < cells; ++cx) {
                float sum = 0.0F;
                for (int j = cy * step; j < (cy + 1) * step; ++j)
                    for (int i = cx * step; i < (cx + 1) * step; ++i) sum += channel[static_cast<std::size_t>(j * n + i)];
                means[static_cast<std::size_t>(cy * cells + cx)] = sum / static_cast<float>(step * step);
            }
        }
        return means;
    };

    Descriptor d;
    BitWriter bits(d);
    const std::array<const std::vector<float>*, kChannels> channels = {&patch, &du, &dv};
    for (const int cells : {2, 3, 4}) {
        std::array<std::vector<float>, kChannels> means;
        for (std::size_t ch = 0; ch < kChannels; ++ch) means[ch] = cell_means(cells, *channels[ch]);
        const std::size_t count = static_cast<std::size_t>(cells * cells);
        for (std::size_t a = 0; a < count; ++a)
            for (std::size_t b = a + 1; b < count; ++b)
                for (std::size_t ch = 0; ch < kChannels; ++ch) bits.push(means[ch][a] > means[ch][b]);
    }

    const auto mean_of = [](const std::vector<float>& v) {
        float sum = 0.0F;
        for (const float x : v) sum += x;
        return sum / static_cast<float>(v.size());
    };
    const float patch_mean = mean_of(patch);
    const float mag_mean = mean_of(mag);
    for (const float m : cell_means(3, patch)) bits.push(m > patch_mean);
    for (const float m : cell_means(3, mag)) bits.push(m > mag_mean);

    if (bits.count() != Descriptor::kBytes * 8) {
        throw ValueError("MLDB layout produced " + std::to_string(bits.count()) + " bits, expected " +
                         std::to_string(Descriptor::kBytes * 8));
    }
    return d;
}

std::vector<Descriptor> compute_mldb(const ScaleSpace& space, const std::vector<Detection>& detections,
                                     const MldbParams& params) {
    std::vector<Descriptor> out;
    out.reserve(detections.size());
    for (const Detection& det : detections) out.push_back(compute_mldb(space, det, params));
    return out;
}

int hamming_distance(const Descriptor& a, const Descriptor& b) {
    int distance = 0;
    for (std::size_t i = 0; i < Descriptor::kBytes; ++i) {
        distance += std::popcount(static_cast<unsigned>(a.bytes[i] ^ b.bytes[i]));
    }
    return distance;
}

}  // namespace alphashape
