#include "common/algorithm/calibration/stmap.hpp"

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <tiffio.h>

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

/// libtiff prints warnings and errors to stderr by default; the library must
/// not. Both handlers swallow the message and report success to the caller,
/// which learns about the failure from a return code instead.
int swallow_error(TIFF*, void*, const char*, const char*, va_list) {
    return 1;  // handled: libtiff prints nothing
}

int swallow_warning(TIFF*, void*, const char*, const char*, va_list) {
    return 1;  // handled: libtiff prints nothing
}

/// Owns a TIFFOpenOptions block and installs the silent handlers.
class SilentOpenOptions {
public:
    SilentOpenOptions() : options_(TIFFOpenOptionsAlloc()) {
        if (options_ != nullptr) {
            TIFFOpenOptionsSetErrorHandlerExtR(options_, swallow_error,
                                               nullptr);
            TIFFOpenOptionsSetWarningHandlerExtR(options_, swallow_warning,
                                                 nullptr);
        }
    }
    ~SilentOpenOptions() {
        if (options_ != nullptr) {
            TIFFOpenOptionsFree(options_);
        }
    }
    SilentOpenOptions(const SilentOpenOptions&) = delete;
    SilentOpenOptions& operator=(const SilentOpenOptions&) = delete;

    TIFFOpenOptions* get() const { return options_; }

private:
    TIFFOpenOptions* options_ = nullptr;
};

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

void write_stmap_tiff(const std::string& path, const STMap& map) {
    if (map.width <= 0 || map.height <= 0) {
        throw ValueError("write_stmap_tiff: map size must be positive");
    }
    const std::size_t count =
        static_cast<std::size_t>(map.width) * static_cast<std::size_t>(map.height);
    if (map.s.size() != count || map.t.size() != count) {
        throw ValueError("write_stmap_tiff: channel size does not match the map");
    }

    SilentOpenOptions options;
    if (options.get() == nullptr) {
        throw Error("write_stmap_tiff: cannot allocate TIFF open options");
    }
    TIFF* raw = TIFFOpenExt(path.c_str(), "w", options.get());
    if (raw == nullptr) {
        throw Error("write_stmap_tiff: cannot open '" + path + "'");
    }
    std::unique_ptr<TIFF, decltype(&TIFFClose)> tif(raw, &TIFFClose);

    bool ok = true;
    auto set = [&ok, &tif](uint32_t tag, auto value) {
        if (TIFFSetField(tif.get(), tag, value) != 1) {
            ok = false;
        }
    };
    set(TIFFTAG_IMAGEWIDTH, static_cast<uint32_t>(map.width));
    set(TIFFTAG_IMAGELENGTH, static_cast<uint32_t>(map.height));
    set(TIFFTAG_SAMPLESPERPIXEL, static_cast<uint16_t>(2));
    set(TIFFTAG_BITSPERSAMPLE, static_cast<uint16_t>(32));
    set(TIFFTAG_SAMPLEFORMAT, static_cast<uint16_t>(SAMPLEFORMAT_IEEEFP));
    set(TIFFTAG_PHOTOMETRIC, static_cast<uint16_t>(PHOTOMETRIC_MINISBLACK));
    set(TIFFTAG_PLANARCONFIG, static_cast<uint16_t>(PLANARCONFIG_CONTIG));
    set(TIFFTAG_ORIENTATION, static_cast<uint16_t>(ORIENTATION_TOPLEFT));
    uint16_t extra = EXTRASAMPLE_UNSPECIFIED;
    if (TIFFSetField(tif.get(), TIFFTAG_EXTRASAMPLES, 1, &extra) != 1) {
        ok = false;
    }
    set(TIFFTAG_COMPRESSION, static_cast<uint16_t>(COMPRESSION_ADOBE_DEFLATE));
    const uint32_t rows =
        static_cast<uint32_t>(TIFFDefaultStripSize(tif.get(), 0));
    if (!ok || TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, rows) != 1) {
        throw Error("write_stmap_tiff: cannot configure '" + path + "'");
    }

    std::vector<float> row(static_cast<std::size_t>(map.width) * 2U);
    for (int y = 0; y < map.height && ok; ++y) {
        const std::size_t base =
            static_cast<std::size_t>(y) * static_cast<std::size_t>(map.width);
        for (int x = 0; x < map.width; ++x) {
            const std::size_t index = base + static_cast<std::size_t>(x);
            const std::size_t out = static_cast<std::size_t>(x) * 2U;
            row[out] = map.s[index];
            row[out + 1U] = map.t[index];
        }
        if (TIFFWriteScanline(tif.get(), row.data(), static_cast<uint32_t>(y),
                              0) < 0) {
            ok = false;
        }
    }
    if (!ok) {
        throw Error("write_stmap_tiff: writing '" + path + "' failed");
    }
}

}  // namespace alphashape
