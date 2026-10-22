#include "common/algorithm/feature/dog_detector.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

#include <Eigen/Dense>

#include "base/utils/error.hpp"

namespace alphashape {

namespace {

constexpr float kTwoPi = 2.0F * std::numbers::pi_v<float>;

FloatImage downsample_half(const FloatImage& in) {
    FloatImage out(std::max(1, in.width() / 2), std::max(1, in.height() / 2));
    for (int y = 0; y < out.height(); ++y) {
        for (int x = 0; x < out.width(); ++x) {
            out.at(x, y) = in.at(2 * x, 2 * y);
        }
    }
    return out;
}

FloatImage subtract(const FloatImage& a, const FloatImage& b) {
    FloatImage out(a.width(), a.height());
    for (std::size_t i = 0; i < out.pixels().size(); ++i) {
        out.pixels()[i] = a.pixels()[i] - b.pixels()[i];
    }
    return out;
}

bool is_extremum(const std::vector<FloatImage>& dogs, int s, int x, int y, float v) {
    for (int ds = -1; ds <= 1; ++ds) {
        const FloatImage& level = dogs[static_cast<std::size_t>(s + ds)];
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (ds == 0 && dx == 0 && dy == 0) continue;
                const float n = level.at(x + dx, y + dy);
                if (v > 0.0F ? n > v : n < v) return false;
            }
        }
    }
    return true;
}

}  // namespace

FloatImage gaussian_blur(const FloatImage& image, float sigma) {
    if (sigma <= 0.0F || image.empty()) {
        return image;
    }
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0F * sigma)));
    std::vector<float> kernel(static_cast<std::size_t>(2 * radius + 1));
    float sum = 0.0F;
    for (int i = -radius; i <= radius; ++i) {
        const float w = std::exp(-static_cast<float>(i * i) / (2.0F * sigma * sigma));
        kernel[static_cast<std::size_t>(i + radius)] = w;
        sum += w;
    }
    for (float& w : kernel) w /= sum;

    FloatImage tmp(image.width(), image.height());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float acc = 0.0F;
            for (int i = -radius; i <= radius; ++i) {
                acc += kernel[static_cast<std::size_t>(i + radius)] * image.at_clamped(x + i, y);
            }
            tmp.at(x, y) = acc;
        }
    }
    FloatImage out(image.width(), image.height());
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            float acc = 0.0F;
            for (int i = -radius; i <= radius; ++i) {
                acc += kernel[static_cast<std::size_t>(i + radius)] * tmp.at_clamped(x, y + i);
            }
            out.at(x, y) = acc;
        }
    }
    return out;
}

DogDetector::DogDetector(DogParams params) : params_(params) {
    if (params_.scales_per_octave < 1) throw ValueError("scales_per_octave must be >= 1");
    if (params_.base_sigma <= params_.assumed_blur) throw ValueError("base_sigma must exceed assumed_blur");
}

ScaleSpace DogDetector::build_scale_space(const FloatImage& luminance) const {
    const int s_count = params_.scales_per_octave;
    ScaleSpace space;
    space.scales_per_octave = s_count;
    space.base_sigma = params_.base_sigma;

    // Stop while the smallest octave still has room for the border and a 3x3x3
    // neighbourhood.
    const int min_side = std::min(luminance.width(), luminance.height());
    int octaves = 0;
    for (int side = min_side; side >= 2 * params_.border + 3 && octaves < params_.max_octaves; side /= 2) {
        ++octaves;
    }
    if (octaves == 0) {
        return space;  // image too small to detect anything
    }

    const float k = std::pow(2.0F, 1.0F / static_cast<float>(s_count));
    const float first = std::sqrt(params_.base_sigma * params_.base_sigma -
                                  params_.assumed_blur * params_.assumed_blur);
    FloatImage base = gaussian_blur(luminance, first);

    for (int o = 0; o < octaves; ++o) {
        std::vector<FloatImage> levels;
        levels.reserve(static_cast<std::size_t>(s_count + 3));
        levels.push_back(base);
        float previous = params_.base_sigma;
        for (int i = 1; i < s_count + 3; ++i) {
            const float total = params_.base_sigma * std::pow(k, static_cast<float>(i));
            levels.push_back(gaussian_blur(levels.back(), std::sqrt(total * total - previous * previous)));
            previous = total;
        }
        std::vector<FloatImage> dogs;
        dogs.reserve(static_cast<std::size_t>(s_count + 2));
        for (std::size_t i = 0; i + 1 < levels.size(); ++i) {
            dogs.push_back(subtract(levels[i + 1], levels[i]));
        }
        // Level S has twice the base blur, so halving it gives the next
        // octave's base at exactly base_sigma in the new pixel units.
        base = downsample_half(levels[static_cast<std::size_t>(s_count)]);
        space.gaussians.push_back(std::move(levels));
        space.dogs.push_back(std::move(dogs));
    }
    return space;
}

std::vector<Detection> DogDetector::detect(const ScaleSpace& space) const {
    std::vector<Detection> out;
    const int s_count = space.scales_per_octave;
    const float prelim = 0.5F * params_.contrast_threshold / static_cast<float>(s_count);
    const float r = params_.edge_ratio;

    for (int o = 0; o < space.octaves(); ++o) {
        const auto& dogs = space.dogs[static_cast<std::size_t>(o)];
        const int w = dogs[0].width();
        const int h = dogs[0].height();
        const float octave_factor = std::ldexp(1.0F, o);

        for (int s = 1; s <= s_count; ++s) {
            const FloatImage& level = dogs[static_cast<std::size_t>(s)];
            for (int y = params_.border; y < h - params_.border; ++y) {
                for (int x = params_.border; x < w - params_.border; ++x) {
                    const float v = level.at(x, y);
                    if (std::abs(v) <= prelim || !is_extremum(dogs, s, x, y, v)) continue;

                    // Subpixel refinement: fit a quadratic in (x, y, scale) and
                    // move to the neighbour sample while the offset exceeds 0.5.
                    int xi = x, yi = y, si = s;
                    Eigen::Vector3f offset = Eigen::Vector3f::Zero();
                    Eigen::Vector3f grad = Eigen::Vector3f::Zero();
                    bool converged = false;
                    for (int it = 0; it < params_.refine_iterations; ++it) {
                        const FloatImage& prev = dogs[static_cast<std::size_t>(si - 1)];
                        const FloatImage& cur = dogs[static_cast<std::size_t>(si)];
                        const FloatImage& next = dogs[static_cast<std::size_t>(si + 1)];
                        const float c = cur.at(xi, yi);
                        grad << 0.5F * (cur.at(xi + 1, yi) - cur.at(xi - 1, yi)),
                            0.5F * (cur.at(xi, yi + 1) - cur.at(xi, yi - 1)),
                            0.5F * (next.at(xi, yi) - prev.at(xi, yi));
                        const float dxx = cur.at(xi + 1, yi) + cur.at(xi - 1, yi) - 2.0F * c;
                        const float dyy = cur.at(xi, yi + 1) + cur.at(xi, yi - 1) - 2.0F * c;
                        const float dss = next.at(xi, yi) + prev.at(xi, yi) - 2.0F * c;
                        const float dxy = 0.25F * (cur.at(xi + 1, yi + 1) - cur.at(xi - 1, yi + 1) -
                                                   cur.at(xi + 1, yi - 1) + cur.at(xi - 1, yi - 1));
                        const float dxs = 0.25F * (next.at(xi + 1, yi) - next.at(xi - 1, yi) -
                                                   prev.at(xi + 1, yi) + prev.at(xi - 1, yi));
                        const float dys = 0.25F * (next.at(xi, yi + 1) - next.at(xi, yi - 1) -
                                                   prev.at(xi, yi + 1) + prev.at(xi, yi - 1));
                        Eigen::Matrix3f hess;
                        hess << dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss;
                        offset = -hess.colPivHouseholderQr().solve(grad);
                        if (!offset.allFinite()) break;
                        if (offset.cwiseAbs().maxCoeff() < 0.5F) {
                            converged = true;
                            break;
                        }
                        xi += static_cast<int>(std::lround(offset.x()));
                        yi += static_cast<int>(std::lround(offset.y()));
                        si += static_cast<int>(std::lround(offset.z()));
                        if (si < 1 || si > s_count || xi < params_.border || xi >= w - params_.border ||
                            yi < params_.border || yi >= h - params_.border) {
                            break;
                        }
                    }
                    if (!converged) continue;

                    const FloatImage& cur = dogs[static_cast<std::size_t>(si)];
                    const float contrast = cur.at(xi, yi) + 0.5F * grad.dot(offset);
                    if (std::abs(contrast) * static_cast<float>(s_count) < params_.contrast_threshold) continue;

                    // Edge rejection (Lowe 2004 §4.1): ratio of principal
                    // curvatures from the 2x2 spatial Hessian.
                    const float c = cur.at(xi, yi);
                    const float dxx = cur.at(xi + 1, yi) + cur.at(xi - 1, yi) - 2.0F * c;
                    const float dyy = cur.at(xi, yi + 1) + cur.at(xi, yi - 1) - 2.0F * c;
                    const float dxy = 0.25F * (cur.at(xi + 1, yi + 1) - cur.at(xi - 1, yi + 1) -
                                               cur.at(xi + 1, yi - 1) + cur.at(xi - 1, yi - 1));
                    const float tr = dxx + dyy;
                    const float det = dxx * dyy - dxy * dxy;
                    if (det <= 0.0F || tr * tr * r >= (r + 1.0F) * (r + 1.0F) * det) continue;

                    Detection d;
                    d.octave = o;
                    d.layer = si;
                    d.octave_x = static_cast<float>(xi) + offset.x();
                    d.octave_y = static_cast<float>(yi) + offset.y();
                    d.octave_sigma = space.base_sigma *
                                     std::pow(2.0F, (static_cast<float>(si) + offset.z()) / static_cast<float>(s_count));
                    d.response = std::abs(contrast);
                    d.keypoint.x = d.octave_x * octave_factor;
                    d.keypoint.y = d.octave_y * octave_factor;
                    d.keypoint.scale = d.octave_sigma * octave_factor;

                    // Dominant orientation(s): gradient histogram over a
                    // Gaussian-weighted window (Lowe 2004 §5).
                    const FloatImage& g = space.gaussians[static_cast<std::size_t>(o)][static_cast<std::size_t>(si)];
                    const float win_sigma = 1.5F * d.octave_sigma;
                    const int radius = static_cast<int>(std::lround(3.0F * win_sigma));
                    const int bins = params_.orientation_bins;
                    std::vector<float> hist(static_cast<std::size_t>(bins), 0.0F);
                    for (int dy = -radius; dy <= radius; ++dy) {
                        for (int dx = -radius; dx <= radius; ++dx) {
                            const int px = xi + dx, py = yi + dy;
                            if (px <= 0 || py <= 0 || px >= w - 1 || py >= h - 1) continue;
                            const float gx = g.at(px + 1, py) - g.at(px - 1, py);
                            const float gy = g.at(px, py + 1) - g.at(px, py - 1);
                            const float weight = std::exp(-static_cast<float>(dx * dx + dy * dy) /
                                                          (2.0F * win_sigma * win_sigma));
                            float angle = std::atan2(gy, gx);
                            if (angle < 0.0F) angle += kTwoPi;
                            int bin = static_cast<int>(std::floor(angle / kTwoPi * static_cast<float>(bins)));
                            bin = std::clamp(bin, 0, bins - 1);
                            hist[static_cast<std::size_t>(bin)] += weight * std::sqrt(gx * gx + gy * gy);
                        }
                    }
                    // Circular [1 4 6 4 1]/16 smoothing before peak picking.
                    std::vector<float> smooth(hist.size());
                    for (int i = 0; i < bins; ++i) {
                        const auto at = [&](int j) { return hist[static_cast<std::size_t>((j + bins) % bins)]; };
                        smooth[static_cast<std::size_t>(i)] =
                            (at(i - 2) + at(i + 2)) * (1.0F / 16.0F) + (at(i - 1) + at(i + 1)) * (4.0F / 16.0F) +
                            at(i) * (6.0F / 16.0F);
                    }
                    const float peak = *std::max_element(smooth.begin(), smooth.end());
                    if (peak <= 0.0F) continue;
                    for (int i = 0; i < bins; ++i) {
                        const float left = smooth[static_cast<std::size_t>((i - 1 + bins) % bins)];
                        const float mid = smooth[static_cast<std::size_t>(i)];
                        const float right = smooth[static_cast<std::size_t>((i + 1) % bins)];
                        if (mid < params_.orientation_peak_ratio * peak || mid <= left || mid <= right) continue;
                        const float shift = 0.5F * (left - right) / (left - 2.0F * mid + right);
                        float angle = (static_cast<float>(i) + 0.5F + shift) * kTwoPi / static_cast<float>(bins);
                        angle = std::fmod(angle + kTwoPi, kTwoPi);
                        Detection oriented = d;
                        oriented.keypoint.orientation = angle;
                        out.push_back(oriented);
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace alphashape
