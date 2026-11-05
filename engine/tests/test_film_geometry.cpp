// Tests for WP-13.18b: the film-normalised frame that the pipeline runs in.
//
// Each scanned film is resampled once into a metric pixel grid, so every later
// stage sees an ordinary image and an ordinary pixel calibration. These tests
// pin the frame mapping, the resampling quality, the central loader's three
// behaviours, fiducial exclusion during detection, and persistence of the new
// FilmSettings fields (the doc.xml case lives in test_fiducials.cpp).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "base/platform/temp.hpp"
#include "base/utils/error.hpp"
#include "common/algorithm/feature/detect_features_task.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/io/camera_image.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "model/camera/calibration.hpp"
#include "model/camera/film_geometry.hpp"

namespace alphashape {
namespace {

/// A square layout: extent 2*half_mm per side and four marks at 80% of it.
FilmSettings square_layout(double half_mm, double resolution) {
    FilmSettings film;
    film.enabled = true;
    film.resolution_mm_per_px = resolution;
    film.film_size_mm = Eigen::Vector2d(2.0 * half_mm, 2.0 * half_mm);
    const double d = 0.8 * half_mm;
    for (int i = 0; i < 4; ++i) {
        Fiducial fiducial;
        fiducial.label = "m" + std::to_string(i);
        fiducial.film_mm = Eigen::Vector2d((i == 0 || i == 3) ? -d : d,
                                           (i < 2) ? d : -d);
        film.fiducials.push_back(fiducial);
    }
    return film;
}

/// Film millimetres -> scan pixels, centred on the scan, rotated and scaled.
/// Scan pixels use the Calibration convention: the top-left pixel centre is
/// (0.5, 0.5), so a scan of width W is centred at W/2.
Eigen::Matrix3d canonical_transform(int scan_w, int scan_h, double px_per_mm,
                                    double rotation_deg) {
    const double angle = rotation_deg * std::numbers::pi_v<double> / 180.0;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double cx = 0.5 * static_cast<double>(scan_w);
    const double cy = 0.5 * static_cast<double>(scan_h);
    Eigen::Matrix3d transform;
    transform << px_per_mm * c, px_per_mm * s, cx,
                 px_per_mm * s, -px_per_mm * c, cy,
                 0.0, 0.0, 1.0;
    return transform;
}

void paint_square(Image& image, int centre_x, int centre_y, int half,
                  std::uint8_t value) {
    for (int y = centre_y - half; y <= centre_y + half; ++y) {
        for (int x = centre_x - half; x <= centre_x + half; ++x) {
            if (x >= 0 && x < image.width() && y >= 0 && y < image.height()) {
                image.at(x, y) = value;
            }
        }
    }
}

/// A smooth 8-bit test pattern in the normalised frame with one bright square,
/// whose continuous centre is (feature_u, feature_v).
Image render_canonical(const FilmSettings& film, Eigen::Vector2d& feature_uv) {
    const FilmSize size = film_normalised_size(film);
    Image image(size.first, size.second, 1);
    for (int y = 0; y < size.second; ++y) {
        for (int x = 0; x < size.first; ++x) {
            const double u = static_cast<double>(x) + 0.5;
            const double v = static_cast<double>(y) + 0.5;
            const double value =
                110.0 + 40.0 * std::sin(u * 0.04) * std::cos(v * 0.035);
            image.at(x, y) = static_cast<std::uint8_t>(
                std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
    const int fx = (size.first * 3) / 4;
    const int fy = (size.second * 3) / 10;
    paint_square(image, fx, fy, 4, static_cast<std::uint8_t>(250));
    feature_uv = Eigen::Vector2d(static_cast<double>(fx) + 0.5,
                                 static_cast<double>(fy) + 0.5);
    return image;
}

/// Bilinear sample of an 8-bit image at continuous coordinates whose origin
/// is the top-left pixel centre at (0.5, 0.5).
double sample_continuous(const Image& image, double u, double v) {
    const double qx = u - 0.5;
    const double qy = v - 0.5;
    if (!(qx > -1.0 && qy > -1.0 &&
          qx < static_cast<double>(image.width()) &&
          qy < static_cast<double>(image.height()))) {
        return 0.0;
    }
    const int x0 = static_cast<int>(std::floor(qx));
    const int y0 = static_cast<int>(std::floor(qy));
    const double fx = qx - static_cast<double>(x0);
    const double fy = qy - static_cast<double>(y0);
    double sum = 0.0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int x = x0 + dx;
            const int y = y0 + dy;
            if (x < 0 || x >= image.width() || y < 0 || y >= image.height()) {
                continue;
            }
            const double weight =
                (dx == 0 ? 1.0 - fx : fx) * (dy == 0 ? 1.0 - fy : fy);
            sum += weight * static_cast<double>(image.at(x, y));
        }
    }
    return sum;
}

/// Warp the normalised test pattern into a scan through `transform`.
Image warp_to_scan(const Image& canonical, const FilmSettings& film,
                   const Eigen::Matrix3d& transform, int scan_w, int scan_h) {
    const Eigen::Matrix3d inverse = transform.inverse();
    Image scan(scan_w, scan_h, 1);
    for (int y = 0; y < scan_h; ++y) {
        for (int x = 0; x < scan_w; ++x) {
            const Eigen::Vector3d h =
                inverse * Eigen::Vector3d(static_cast<double>(x) + 0.5,
                                          static_cast<double>(y) + 0.5, 1.0);
            if (!h.allFinite() || std::abs(h.z()) < 1.0e-12) {
                scan.at(x, y) = 0;
                continue;
            }
            const Eigen::Vector2d mm(h.x() / h.z(), h.y() / h.z());
            const Eigen::Vector2d uv = normalised_from_film_mm(film, mm);
            const double value = sample_continuous(canonical, uv.x(), uv.y());
            scan.at(x, y) = static_cast<std::uint8_t>(
                std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
    return scan;
}

TEST(FilmGeometry, RoundTripMappingsInvert) {
    const FilmSettings film = square_layout(20.0, 0.05);
    const FilmSize size = film_normalised_size(film);
    ASSERT_EQ(size.first, 800);
    ASSERT_EQ(size.second, 800);
    const std::vector<Eigen::Vector2d> points = {
        Eigen::Vector2d(100.5, 200.5), Eigen::Vector2d(700.0, 50.0),
        Eigen::Vector2d(400.0, 400.0)};
    for (int kind = 0; kind < 2; ++kind) {
        Eigen::Matrix3d transform = canonical_transform(1200, 1200, 18.0, 11.0);
        if (kind == 1) {
            transform(2, 0) = 1.0e-5;
            transform(2, 1) = -2.0e-5;
        }
        const Eigen::Matrix3d inverse = transform.inverse();
        for (const Eigen::Vector2d& normalised : points) {
            const Eigen::Vector2d mm = film_mm_from_normalised(film, normalised);
            const Eigen::Vector2d back_normalised =
                normalised_from_film_mm(film, mm);
            EXPECT_LT((back_normalised - normalised).norm(), 1.0e-9);

            const std::optional<Eigen::Vector2d> pixel =
                scan_pixel_from_normalised(transform, film, normalised);
            ASSERT_TRUE(pixel.has_value());
            const Eigen::Vector3d h =
                inverse * Eigen::Vector3d(pixel->x(), pixel->y(), 1.0);
            ASSERT_GT(std::abs(h.z()), 1.0e-12);
            const Eigen::Vector2d mm_back(h.x() / h.z(), h.y() / h.z());
            const Eigen::Vector2d pixel_normalised =
                normalised_from_film_mm(film, mm_back);
            EXPECT_LT((pixel_normalised - normalised).norm(), 1.0e-9)
                << "kind " << kind;
        }
    }
}

TEST(FilmGeometry, ResampleRecoversCanonicalImage) {
    const FilmSettings film = square_layout(20.0, 0.05);
    Eigen::Vector2d feature(0.0, 0.0);
    const Image canonical = render_canonical(film, feature);
    Eigen::Matrix3d shear = Eigen::Matrix3d::Identity();
    shear(0, 1) = 0.04;
    const Eigen::Matrix3d transform =
        canonical_transform(1000, 1000, 20.0, 7.0) * shear;
    const Image scan =
        warp_to_scan(canonical, film, transform, 1000, 1000);
    const Image back = resample_film_image(scan, transform, film);
    ASSERT_EQ(back.width(), canonical.width());
    ASSERT_EQ(back.height(), canonical.height());

    double sum = 0.0;
    std::size_t count = 0;
    for (int y = 80; y < canonical.height() - 80; ++y) {
        for (int x = 80; x < canonical.width() - 80; ++x) {
            sum += std::abs(static_cast<double>(back.at(x, y)) -
                            static_cast<double>(canonical.at(x, y)));
            ++count;
        }
    }
    ASSERT_GT(count, 0U);
    EXPECT_LT(sum / static_cast<double>(count), 2.0);

    // The bright feature's centroid must land within 0.1 px.
    const int fx = static_cast<int>(std::floor(feature.x() - 0.5));
    const int fy = static_cast<int>(std::floor(feature.y() - 0.5));
    double weight = 0.0;
    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for (int y = fy - 8; y <= fy + 8; ++y) {
        for (int x = fx - 8; x <= fx + 8; ++x) {
            if (x < 0 || x >= back.width() || y < 0 || y >= back.height()) {
                continue;
            }
            const double value = static_cast<double>(back.at(x, y));
            if (value < 200.0) {
                continue;
            }
            centroid += value * Eigen::Vector2d(static_cast<double>(x) + 0.5,
                                                static_cast<double>(y) + 0.5);
            weight += value;
        }
    }
    ASSERT_GT(weight, 0.0);
    centroid /= weight;
    EXPECT_LT((centroid - feature).norm(), 0.1);
}

std::string write_temp_png(const Image& image, const std::string& name) {
    const std::string path =
        platform::unique_temp_path("alphashape_filmg_" + name + "_", ".png");
    write_png(path, image);
    return path;
}

/// A chunk with one sensor/camera/frame. `transform` nullopt means the film
/// camera has no fitted scan transform yet.
Chunk& add_camera(Document& document, const FilmSettings& film, int width,
                  int height, const std::string& path,
                  const std::optional<Eigen::Matrix3d>& transform) {
    Chunk& chunk = document.workspace().add_chunk("film");
    document.set_active_chunk_id(chunk.id());
    Sensor sensor;
    sensor.label = "film-camera";
    sensor.type = CameraType::Frame;
    sensor.width = width;
    sensor.height = height;
    sensor.calibration =
        Calibration::initial(CameraType::Frame, width, height, 1000.0);
    sensor.film = film;
    chunk.sensors().push_back(sensor);
    Camera camera;
    camera.id = chunk.allocate_id();
    camera.sensor_id = 0;
    camera.label = "scan";
    camera.enabled = true;
    camera.film_transform = transform;
    chunk.cameras().push_back(camera);
    auto frame = std::make_shared<Frame>(path, width, height);
    frame->set_id(chunk.allocate_id());
    frame->set_camera_id(camera.id);
    chunk.frames().push_back(frame);
    return chunk;
}

/// The transform that makes the scan grid identical to the normalised grid:
/// film mm -> pixel is the inverse of the normalised mapping.
Eigen::Matrix3d identity_scan_transform(const FilmSettings& film) {
    const FilmSize size = film_normalised_size(film);
    Eigen::Matrix3d transform = Eigen::Matrix3d::Zero();
    transform(0, 0) = 1.0 / film.resolution_mm_per_px;
    transform(1, 1) = -1.0 / film.resolution_mm_per_px;
    transform(0, 2) = 0.5 * static_cast<double>(size.first);
    transform(1, 2) = 0.5 * static_cast<double>(size.second);
    transform(2, 2) = 1.0;
    return transform;
}

TEST(FilmPipeline, LoaderResamplesOnlyFilmCameras) {
    const FilmSettings film = square_layout(20.0, 0.05);  // 800 x 800
    Image scan(1000, 1000, 1);
    for (int y = 0; y < scan.height(); ++y) {
        for (int x = 0; x < scan.width(); ++x) {
            scan.at(x, y) = static_cast<std::uint8_t>((x + y) % 251);
        }
    }
    const std::string path = write_temp_png(scan, "pipeline");
    const Eigen::Matrix3d transform = canonical_transform(1000, 1000, 20.0, 3.0);

    {
        Document document;
        Chunk& chunk = add_camera(document, film, 1000, 1000, path, transform);
        const Image loaded =
            load_camera_image(document, chunk, chunk.cameras()[0]);
        EXPECT_EQ(loaded.width(), 800);
        EXPECT_EQ(loaded.height(), 800);
        EXPECT_EQ(loaded.channels(), 1);
    }
    {
        // A film camera without a fitted transform must not silently use the
        // raw scan: the error names the camera and the fixing task.
        Document document;
        Chunk& chunk =
            add_camera(document, film, 1000, 1000, path, std::nullopt);
        try {
            (void)load_camera_image(document, chunk, chunk.cameras()[0]);
            FAIL() << "expected a film camera without a transform to throw";
        } catch (const Error& error) {
            EXPECT_NE(std::string(error.what()).find("DetectFiducials"),
                      std::string::npos);
        }
    }
    {
        // A digital camera stays bit-identical.
        Document document;
        Chunk& chunk = add_camera(document, FilmSettings{}, 1000, 1000, path,
                                  std::nullopt);
        const Image loaded =
            load_camera_image(document, chunk, chunk.cameras()[0]);
        const Image raw = read_image(path);
        ASSERT_EQ(loaded.size_bytes(), raw.size_bytes());
        EXPECT_EQ(std::memcmp(loaded.data(), raw.data(), raw.size_bytes()), 0);
    }
    {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
}

TEST(FilmFeatures, FiducialsAndBorderExcluded) {
    const FilmSettings film = square_layout(30.0, 0.1);  // 600 x 600
    const FilmSize size = film_normalised_size(film);
    Image image(size.first, size.second, 1);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            image.at(x, y) = static_cast<std::uint8_t>(120 + (x / 40) % 2 * 15);
        }
    }
    std::vector<Eigen::Vector2d> marks;
    for (const Fiducial& fiducial : film.fiducials) {
        const Eigen::Vector2d uv = normalised_from_film_mm(film, fiducial.film_mm);
        marks.push_back(uv);
        paint_square(image, static_cast<int>(std::floor(uv.x() - 0.5)),
                     static_cast<int>(std::floor(uv.y() - 0.5)), 4,
                     static_cast<std::uint8_t>(250));
    }
    // A strong feature away from every mark and from the border must survive.
    paint_square(image, size.first / 2, size.second / 2, 6,
                 static_cast<std::uint8_t>(250));
    const std::string path = write_temp_png(image, "features");
    Document document;
    Chunk& chunk = add_camera(document, film, size.first, size.second, path,
                              identity_scan_transform(film));
    Frame* frame = chunk.frames()[0].get();
    const KeypointSet set =
        detect_keypoints_for_frame(document, chunk, *frame, 40000, 1000, 5.0);
    EXPECT_FALSE(set.keypoints.empty());
    const double radius = 5.0 / film.resolution_mm_per_px;
    const double max_x = static_cast<double>(size.first) - radius;
    const double max_y = static_cast<double>(size.second) - radius;
    bool centre_feature = false;
    const Eigen::Vector2d centre(0.5 * static_cast<double>(size.first),
                                 0.5 * static_cast<double>(size.second));
    for (const Keypoint& keypoint : set.keypoints) {
        const Eigen::Vector2d point(static_cast<double>(keypoint.x) + 0.5,
                                    static_cast<double>(keypoint.y) + 0.5);
        EXPECT_GE(point.x(), radius);
        EXPECT_GE(point.y(), radius);
        EXPECT_LE(point.x(), max_x);
        EXPECT_LE(point.y(), max_y);
        for (const Eigen::Vector2d& mark : marks) {
            EXPECT_GE((point - mark).norm(), radius);
        }
        if ((point - centre).norm() < 20.0) {
            centre_feature = true;
        }
    }
    EXPECT_TRUE(centre_feature);
    std::error_code error;
    std::filesystem::remove(path, error);
}

}  // namespace
}  // namespace alphashape
