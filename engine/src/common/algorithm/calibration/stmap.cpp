#include "common/algorithm/calibration/stmap.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

#include "base/utils/error.hpp"

namespace alphashape {
namespace {

/// Normalised storage value for a source coordinate, or NaN when the projection
/// produced anything a resampler cannot use.
void store_source(double u_src, double v_src, double width, double height,
                  float& s, float& t) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(u_src) || !std::isfinite(v_src)) {
        s = static_cast<float>(nan);
        t = static_cast<float>(nan);
        return;
    }
    const double s_value = u_src / width;
    const double t_value = 1.0 - v_src / height;
    if (!std::isfinite(s_value) || !std::isfinite(t_value)) {
        s = static_cast<float>(nan);
        t = static_cast<float>(nan);
        return;
    }
    s = static_cast<float>(s_value);
    t = static_cast<float>(t_value);
}

}  // namespace

STMap compute_stmap(const Calibration& calibration, int width, int height) {
    if (calibration.width <= 0 || calibration.height <= 0) {
        throw ValueError("compute_stmap: calibration geometry must be positive");
    }
    if (!(calibration.f > 0.0) || !std::isfinite(calibration.f)) {
        throw ValueError("compute_stmap: focal length must be positive and finite");
    }
    if (width == 0) {
        width = calibration.width;
    }
    if (height == 0) {
        height = calibration.height;
    }
    if (width <= 0 || height <= 0) {
        throw ValueError("compute_stmap: map size must be positive");
    }

    const double sensor_width = static_cast<double>(calibration.width);
    const double sensor_height = static_cast<double>(calibration.height);
    const double centre_x = sensor_width / 2.0 + calibration.cx;
    const double centre_y = sensor_height / 2.0 + calibration.cy;

    STMap map;
    map.width = width;
    map.height = height;
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    map.s.resize(count);
    map.t.resize(count);

    for (int row = 0; row < height; ++row) {
        const double v_out = (static_cast<double>(row) + 0.5) * sensor_height /
                             static_cast<double>(height);
        for (int col = 0; col < width; ++col) {
            const double u_out = (static_cast<double>(col) + 0.5) * sensor_width /
                                 static_cast<double>(width);
            const double x = (u_out - centre_x) / calibration.f;
            const double y = (v_out - centre_y) / calibration.f;
            const std::size_t index =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(col);
            double u_src = std::numeric_limits<double>::quiet_NaN();
            double v_src = std::numeric_limits<double>::quiet_NaN();
            try {
                const Eigen::Vector2d source =
                    calibration.project(Eigen::Vector3d(x, y, 1.0));
                u_src = source.x();
                v_src = source.y();
            } catch (const Error&) {
                // A projection the model cannot define stays a hole.
            }
            store_source(u_src, v_src, sensor_width, sensor_height,
                         map.s[index], map.t[index]);
        }
    }
    return map;
}

}  // namespace alphashape
