#include "model/camera/film_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

void require_complete(const FilmSettings& film, const char* what) {
    if (!film_geometry_complete(film)) {
        throw ValueError(std::string(what) +
                         ": the film frame needs a positive resolution and "
                         "extent (run CalibrateFiducials first)");
    }
}

/// True when a finite, strictly positive scalar.
bool positive(double value) { return std::isfinite(value) && value > 0.0; }

}  // namespace

bool film_geometry_complete(const FilmSettings& film) {
    const Eigen::Vector2d extent = film_extent_mm(film);
    return positive(film.resolution_mm_per_px) && positive(extent.x()) &&
           positive(extent.y());
}

Eigen::Vector2d film_extent_mm(const FilmSettings& film) {
    if (positive(film.film_size_mm.x()) && positive(film.film_size_mm.y())) {
        return film.film_size_mm;
    }
    if (film.fiducials.empty()) {
        return Eigen::Vector2d::Zero();
    }
    Eigen::Vector2d minimum =
        Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector2d maximum =
        Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
    for (const Fiducial& fiducial : film.fiducials) {
        minimum = minimum.cwiseMin(fiducial.film_mm);
        maximum = maximum.cwiseMax(fiducial.film_mm);
    }
    return (maximum - minimum).cwiseMax(0.0);
}

FilmSize film_normalised_size(const FilmSettings& film) {
    require_complete(film, "film_normalised_size");
    const Eigen::Vector2d extent = film_extent_mm(film);
    const int width = static_cast<int>(
        std::lround(extent.x() / film.resolution_mm_per_px));
    const int height = static_cast<int>(
        std::lround(extent.y() / film.resolution_mm_per_px));
    if (width <= 0 || height <= 0) {
        throw ValueError("film_normalised_size: the normalised image is empty");
    }
    return FilmSize{width, height};
}

Eigen::Vector2d film_mm_from_normalised(const FilmSettings& film,
                                        const Eigen::Vector2d& normalised) {
    const FilmSize size = film_normalised_size(film);
    const double res = film.resolution_mm_per_px;
    return Eigen::Vector2d(
        (normalised.x() - 0.5 * static_cast<double>(size.first)) * res,
        -(normalised.y() - 0.5 * static_cast<double>(size.second)) * res);
}

Eigen::Vector2d normalised_from_film_mm(const FilmSettings& film,
                                        const Eigen::Vector2d& film_mm) {
    const FilmSize size = film_normalised_size(film);
    const double res = film.resolution_mm_per_px;
    return Eigen::Vector2d(
        film_mm.x() / res + 0.5 * static_cast<double>(size.first),
        -film_mm.y() / res + 0.5 * static_cast<double>(size.second));
}

std::optional<Eigen::Vector2d> scan_pixel_from_normalised(
    const Eigen::Matrix3d& film_transform, const FilmSettings& film,
    const Eigen::Vector2d& normalised) {
    const Eigen::Vector2d mm = film_mm_from_normalised(film, normalised);
    const Eigen::Vector3d h =
        film_transform * Eigen::Vector3d(mm.x(), mm.y(), 1.0);
    if (!h.allFinite() || std::abs(h.z()) < 1.0e-12) {
        return std::nullopt;
    }
    const Eigen::Vector2d pixel(h.x() / h.z(), h.y() / h.z());
    if (!pixel.allFinite()) {
        return std::nullopt;
    }
    return pixel;
}

namespace {

std::vector<float> to_float_buffer(const Image& image) {
    std::vector<float> out(image.size_bytes());
    const std::uint8_t* src = image.data();
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>(src[i]);
    }
    return out;
}

std::vector<double> gaussian_kernel(double sigma) {
    const int radius =
        std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        const double d = static_cast<double>(i);
        const double v = std::exp(-0.5 * d * d / (sigma * sigma));
        kernel[static_cast<std::size_t>(i + radius)] = v;
        sum += v;
    }
    for (double& v : kernel) {
        v /= sum;
    }
    return kernel;
}

/// Separable Gaussian blur over an interleaved float buffer, edge clamping.
std::vector<float> gaussian_blur(const std::vector<float>& src, int width,
                                 int height, int channels, double sigma) {
    const std::vector<double> kernel = gaussian_kernel(sigma);
    const int radius = static_cast<int>(kernel.size() / 2);
    const std::size_t c = static_cast<std::size_t>(channels);
    const std::size_t w = static_cast<std::size_t>(width);
    const std::size_t h = static_cast<std::size_t>(height);
    std::vector<double> tmp(src.size(), 0.0);
    std::vector<double> out(src.size(), 0.0);
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            for (std::size_t ch = 0; ch < c; ++ch) {
                double sum = 0.0;
                for (int t = -radius; t <= radius; ++t) {
                    const int xx = std::clamp(static_cast<int>(x) + t, 0, width - 1);
                    sum += kernel[static_cast<std::size_t>(t + radius)] *
                           static_cast<double>(src[(y * w + static_cast<std::size_t>(xx)) * c + ch]);
                }
                tmp[(y * w + x) * c + ch] = sum;
            }
        }
    }
    for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
            for (std::size_t ch = 0; ch < c; ++ch) {
                double sum = 0.0;
                for (int t = -radius; t <= radius; ++t) {
                    const int yy = std::clamp(static_cast<int>(y) + t, 0, height - 1);
                    sum += kernel[static_cast<std::size_t>(t + radius)] *
                           tmp[(static_cast<std::size_t>(yy) * w + x) * c + ch];
                }
                out[(y * w + x) * c + ch] = sum;
            }
        }
    }
    std::vector<float> result(out.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        result[i] = static_cast<float>(out[i]);
    }
    return result;
}

/// Bilinear sample in array coordinates (pixel index i's centre is at i).
/// Taps outside the image contribute 0, so the normalised border is black.
double sample_bilinear(const std::vector<float>& src, int width, int height,
                       int channels, double qx, double qy, int channel) {
    if (!(qx >= -1.0 && qy >= -1.0 && qx <= static_cast<double>(width) &&
          qy <= static_cast<double>(height))) {
        return 0.0;  // also rejects NaN
    }
    const int x0 = static_cast<int>(std::floor(qx));
    const int y0 = static_cast<int>(std::floor(qy));
    const double fx = qx - static_cast<double>(x0);
    const double fy = qy - static_cast<double>(y0);
    const std::size_t c = static_cast<std::size_t>(channels);
    double sum = 0.0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int x = x0 + dx;
            const int y = y0 + dy;
            if (x < 0 || x >= width || y < 0 || y >= height) {
                continue;
            }
            const double weight = (dx == 0 ? 1.0 - fx : fx) *
                                  (dy == 0 ? 1.0 - fy : fy);
            const std::size_t index =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)) *
                    c +
                static_cast<std::size_t>(channel);
            sum += weight * static_cast<double>(src[index]);
        }
    }
    return sum;
}

}  // namespace

Image resample_film_image(const Image& scan,
                          const Eigen::Matrix3d& film_transform,
                          const FilmSettings& film) {
    const FilmSize size = film_normalised_size(film);
    if (scan.empty() || scan.width() <= 0 || scan.height() <= 0 ||
        scan.channels() <= 0) {
        throw ValueError("resample_film_image: the scan is empty");
    }
    const int width = scan.width();
    const int height = scan.height();
    const int channels = scan.channels();
    std::vector<float> source = to_float_buffer(scan);
    // One normalised pixel covers this many scan pixels; past 1.5 the bilinear
    // tap undersamples the scan, so prefilter to the output's Nyquist limit.
    const Eigen::Matrix2d linear = film_transform.topLeftCorner<2, 2>();
    const double factor =
        film.resolution_mm_per_px * std::sqrt(std::abs(linear.determinant()));
    if (std::isfinite(factor) && factor > 1.5) {
        source = gaussian_blur(source, width, height, channels, 0.5 * factor);
    }
    Image out(size.first, size.second, channels);
    for (int y = 0; y < size.second; ++y) {
        for (int x = 0; x < size.first; ++x) {
            const Eigen::Vector2d normalised(static_cast<double>(x) + 0.5,
                                             static_cast<double>(y) + 0.5);
            const std::optional<Eigen::Vector2d> pixel =
                scan_pixel_from_normalised(film_transform, film, normalised);
            if (!pixel.has_value()) {
                for (int ch = 0; ch < channels; ++ch) {
                    out.at(x, y, ch) = 0;
                }
                continue;
            }
            // Scan pixel coordinates also use the (0.5, 0.5) centre convention;
            // convert to the array index whose centre is the integer index.
            const double qx = pixel->x() - 0.5;
            const double qy = pixel->y() - 0.5;
            for (int ch = 0; ch < channels; ++ch) {
                const double value =
                    sample_bilinear(source, width, height, channels, qx, qy, ch);
                const long rounded =
                    std::lround(std::clamp(value, 0.0, 255.0));
                out.at(x, y, ch) = static_cast<std::uint8_t>(rounded);
            }
        }
    }
    return out;
}

}  // namespace alphashape
