#pragma once

// Synthetic marker sequence for the TrackMarkers tests (WP-13.10).
//
// A textured ground plane carrying three coded circular targets and one
// checker cross is inverse-mapped through the plane homography for every
// pixel, so the printed pattern is sampled analytically rather than warped
// from a bitmap. A smoothly moving camera with a small rotation renders the
// sequence; blur and noise are added afterwards. Capture times can be written
// into PNG eXIf chunks in an order unrelated to import order, which is what
// makes ordering-by-time observable.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include "base/platform/temp.hpp"
#include "base/utils/portable_rng.hpp"
#include "common/algorithm/feature/dog_detector.hpp"
#include "common/algorithm/markers/project_markers.hpp"
#include "common/algorithm/markers/target_patterns.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "common/core/marker.hpp"
#include "common/core/types.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"
#include "make_circular_target.hpp"
#include "make_marker_sheet.hpp"

namespace alphashape::test {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;

struct MarkerSequenceTarget {
    bool cross = false;
    Vec3 centre = Vec3::Zero();  ///< on the ground plane, y = 0
    double radius = 0.03;        ///< physical outer radius, metres
    double rotation = 0.0;
    int bits = 12;
    int code = 0;                ///< 2 * table index + parity
    std::uint32_t word = 0;
};

struct MarkerSequenceSpec {
    int frames = 20;
    int width = 640;
    int height = 480;
    double focal = 600.0;
    /// Temporal indices in import order; empty means 0, 1, ..., frames-1.
    std::vector<int> import_order;
    /// When set, the capture time is the reverse of the temporal index, so a
    /// time-ordered run and an import-ordered run differ.
    bool reverse_capture_times = false;
    int missing_time_temporal = -1;
    /// Every frame shares one whole second; SubSecTimeOriginal carries the
    /// temporal index, so ordering depends on the subsecond field.
    bool burst_capture_times = false;
    /// Temporal index whose DateTimeOriginal is deliberately malformed.
    int malformed_time_temporal = -1;
    int occlude_first = -1;
    int occlude_last = -1;
    int occluded_target = 0;
    std::uint32_t seed = 7;
};

struct MarkerSequenceFrame {
    int temporal_index = 0;
    int camera_id = -1;
    int import_index = 0;
    std::string path;
    Transform pose = Transform::Identity();
    Calibration calibration;
};

struct MarkerSequence {
    Document document;
    Chunk* chunk = nullptr;
    std::vector<MarkerSequenceTarget> targets;
    std::vector<MarkerSequenceFrame> frames;  ///< temporal order
    std::vector<std::string> paths;
    std::vector<int> import_order;

    MarkerSequence() = default;
    MarkerSequence(MarkerSequence&&) = default;
    MarkerSequence& operator=(MarkerSequence&&) = default;
    MarkerSequence(const MarkerSequence&) = delete;
    MarkerSequence& operator=(const MarkerSequence&) = delete;
    ~MarkerSequence();

    const Camera& camera(std::size_t temporal) const;
    const Calibration& calibration(std::size_t temporal) const;
    std::optional<Vec2> truth(std::size_t target, std::size_t temporal) const;
};

MarkerSequence make_marker_sequence(const MarkerSequenceSpec& spec);

inline std::uint32_t crc32_of(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<std::uint32_t>(data[i]);
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^
                  (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

/// Minimal little-endian TIFF block with IFD0 -> ExifIFD -> DateTimeOriginal
/// and, when `subsec` is non-empty, SubSecTimeOriginal.
inline std::vector<std::uint8_t> datetime_tiff_block(
    const std::string& when,
    const std::string& subsec = std::string()) {
    std::vector<std::uint8_t> block(96, 0);
    const auto put16 = [&block](std::size_t at, std::uint16_t value) {
        block[at] = static_cast<std::uint8_t>(value & 0xFFu);
        block[at + 1] = static_cast<std::uint8_t>(value >> 8);
    };
    const auto put32 = [&block](std::size_t at, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            block[at + static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(
                (value >> (8 * i)) & 0xFFu);
        }
    };
    block[0] = 'I';
    block[1] = 'I';
    put16(2, 42);
    put32(4, 8);
    // IFD0: one entry, the ExifIFD pointer.
    put16(8, 1);
    put16(10, 0x8769);
    put16(12, 4);
    put32(14, 1);
    put32(18, 26);
    put32(22, 0);
    // ExifIFD: DateTimeOriginal, then optionally SubSecTimeOriginal.
    const std::uint16_t entries = subsec.empty() ? 1 : 2;
    put16(26, entries);
    put16(28, 0x9003);
    put16(30, 2);
    put32(32, 20);
    put32(36, 56);
    put32(40, 0);
    if (!subsec.empty()) {
        put16(40, 0x9291);
        put16(42, 2);
        put32(44, static_cast<std::uint32_t>(subsec.size() + 1));
        put32(48, 76);
        put32(52, 0);
    }
    const std::string text = when.substr(0, 19);
    for (std::size_t i = 0; i < text.size(); ++i) {
        block[56 + i] = static_cast<std::uint8_t>(text[i]);
    }
    for (std::size_t i = 0; i < subsec.size(); ++i) {
        block[76 + i] = static_cast<std::uint8_t>(subsec[i]);
    }
    return block;
}

/// Write an 8-bit grayscale PNG, optionally with a DateTimeOriginal eXIf
/// chunk. `read_exif` parses the block back, so the test exercises the same
/// capture-time path a real photo uses.
inline void write_gray_png_with_time(
    const std::string& path, const FloatImage& image,
    const std::optional<std::string>& when,
    const std::optional<std::string>& subsec = std::nullopt) {
    Image out(image.width(), image.height(), 1);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const double value = static_cast<double>(image.at(x, y)) * 255.0;
            out.at(x, y, 0) = static_cast<std::uint8_t>(
                std::lround(std::clamp(value, 0.0, 255.0)));
        }
    }
    write_png(path, out);
    if (!when) {
        return;
    }
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        return;
    }
    const std::streamoff end = in.tellg();
    if (end < 12) {
        return;
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    const std::vector<std::uint8_t> tiff =
        datetime_tiff_block(*when, subsec.value_or(std::string()));
    std::vector<std::uint8_t> chunk;
    const std::uint32_t length = static_cast<std::uint32_t>(tiff.size());
    for (int i = 3; i >= 0; --i) {
        chunk.push_back(static_cast<std::uint8_t>((length >> (8 * i)) & 0xFFu));
    }
    for (const char c : std::string("eXIf")) {
        chunk.push_back(static_cast<std::uint8_t>(c));
    }
    chunk.insert(chunk.end(), tiff.begin(), tiff.end());
    const std::uint32_t crc = crc32_of(chunk.data() + 4, chunk.size() - 4);
    for (int i = 3; i >= 0; --i) {
        chunk.push_back(static_cast<std::uint8_t>((crc >> (8 * i)) & 0xFFu));
    }
    const std::size_t iend = bytes.size() - 12;
    bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(iend), chunk.begin(),
                 chunk.end());
    std::ofstream out_file(path, std::ios::binary | std::ios::trunc);
    out_file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
}

/// Deterministic, continuous ground texture in world coordinates.
inline float sequence_ground_shade(double x, double z) {
    const double wave = 0.04 * std::sin(41.0 * x + 0.6) +
                        0.03 * std::cos(33.0 * z - 0.3) +
                        0.02 * std::sin(57.0 * (x + z));
    const auto hash = [](int ix, int iz) {
        std::uint32_t value = static_cast<std::uint32_t>(ix) * 73856093U ^
                              static_cast<std::uint32_t>(iz) * 19349663U;
        value ^= value >> 13;
        value *= 0x5bd1e995U;
        value ^= value >> 15;
        return static_cast<double>(value & 0xffffU) / 65535.0 - 0.5;
    };
    const double grain = 0.04 * hash(static_cast<int>(std::floor(x * 55.0)),
                                     static_cast<int>(std::floor(z * 55.0)));
    // Light, paper-like ground keeps the coded target's dark outer border
    // separated from its quiet zone, which is what the decoder assumes.
    return static_cast<float>(
        std::clamp(0.78 + wave + grain, 0.66, 0.90));
}

/// Target intensity at a world point, or nullopt when the point is ground.
inline std::optional<float> sequence_target_shade(
    const MarkerSequenceTarget& target, double x, double z) {
    const double dx = x - target.centre.x();
    const double dz = z - target.centre.z();
    const double extent = target.radius * 1.5;
    if (std::abs(dx) > extent || std::abs(dz) > extent) {
        return std::nullopt;
    }
    const double c = std::cos(target.rotation);
    const double s = std::sin(target.rotation);
    const double u = dx * c + dz * s;
    const double v = -dx * s + dz * c;
    // A camera above the ground sees image x follow -z and image y follow -x
    // (the pixel Jacobian of the plane homography), so the camera-facing
    // (right, down) frame is (-v, -u). Its determinant against image (x right,
    // y down) is positive: a printed target is only rotated by the view, never
    // mirrored. The ring, the cross and the target rotation all use this one
    // basis.
    const Vec2 local(-v / target.radius, -u / target.radius);
    if (target.cross) {
        const double half = 0.9;
        if (std::abs(local.x()) > half || std::abs(local.y()) > half) {
            return std::nullopt;
        }
        return local.x() * local.y() > 0.0 ? 0.95F : 0.05F;
    }
    const double rho = local.norm();
    if (rho > 1.0) {
        return std::nullopt;
    }
    CircularTruth truth;
    truth.bits = target.bits;
    truth.word = target.word;
    truth.parity = 1;
    const double sector_width =
        2.0 * std::numbers::pi_v<double> / static_cast<double>(target.bits);
    return circular_black_at(local, truth, sector_width) ? 0.05F : 0.95F;
}

/// Smoothly translating camera aimed at the ground, with a small sinusoidal
/// rotation about each camera axis. The motion is slow enough that one
/// frame-to-frame Lucas-Kanade translation tracks the surface.
inline Transform sequence_pose(int index, int count, double height) {
    const double u = count > 1 ? static_cast<double>(index) /
                                     static_cast<double>(count - 1)
                               : 0.5;
    const Vec3 centre(-0.02 + 0.05 * u, -height, -0.03 + 0.06 * u);
    const Vec3 aim(0.02 * std::sin(2.0 * std::numbers::pi_v<double> * u), 0.0,
                   0.01 * std::cos(2.0 * std::numbers::pi_v<double> * u));
    const Vec3 forward = (aim - centre).normalized();
    Vec3 reference = Vec3::UnitX();
    if (std::abs(forward.dot(reference)) > 0.9) {
        reference = Vec3::UnitZ();
    }
    const Vec3 right = forward.cross(reference).normalized();
    const Vec3 down = forward.cross(right);
    Eigen::Matrix3d base;
    base.row(0) = right.transpose();
    base.row(1) = down.transpose();
    base.row(2) = forward.transpose();
    const double phase = 2.0 * std::numbers::pi_v<double> * u;
    const double yaw = 0.012 * std::sin(phase);
    const double pitch = 0.008 * std::sin(phase + 1.1);
    const double roll = 0.005 * std::cos(phase);
    const Eigen::Matrix3d delta =
        (Eigen::AngleAxisd(yaw, Vec3::UnitY()).toRotationMatrix() *
         Eigen::AngleAxisd(pitch, Vec3::UnitX()).toRotationMatrix() *
         Eigen::AngleAxisd(roll, Vec3::UnitZ()).toRotationMatrix());
    const Eigen::Matrix3d rotation = delta * base;
    Transform pose = Transform::Identity();
    pose.block<3, 3>(0, 0) = rotation;
    pose.block<3, 1>(0, 3) = -rotation * centre;
    return pose;
}

/// Render one frame by inverse-mapping every subpixel through the ground
/// homography. Painting happens in world coordinates, so a target keeps its
/// printed shape under motion and rotation.
inline FloatImage render_sequence_view(
    const Transform& pose, const Calibration& calibration,
    const std::vector<MarkerSequenceTarget>& targets, int temporal,
    const MarkerSequenceSpec& spec) {
    const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
    const Vec3 translation = pose.block<3, 1>(0, 3);
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();
    intrinsics(0, 0) = calibration.f + calibration.b1;
    intrinsics(0, 1) = calibration.b2;
    intrinsics(0, 2) = calibration.width * 0.5 + calibration.cx;
    intrinsics(1, 1) = calibration.f;
    intrinsics(1, 2) = calibration.height * 0.5 + calibration.cy;
    Eigen::Matrix3d plane;
    plane.col(0) = rotation.col(0);
    plane.col(1) = rotation.col(2);
    plane.col(2) = translation;
    const Eigen::Matrix3d inverse = (intrinsics * plane).inverse();
    FloatImage image(calibration.width, calibration.height, 0.5F);
    for (int y = 0; y < calibration.height; ++y) {
        for (int x = 0; x < calibration.width; ++x) {
            double sum = 0.0;
            for (int sy = 0; sy < 2; ++sy) {
                for (int sx = 0; sx < 2; ++sx) {
                    const double px = static_cast<double>(x) + 0.25 + 0.5 * sx;
                    const double py = static_cast<double>(y) + 0.25 + 0.5 * sy;
                    const Vec3 q = inverse * Vec3(px, py, 1.0);
                    double shade = 0.5;
                    const double wx = q.x() / q.z();
                    const double wz = q.y() / q.z();
                    const double depth =
                        rotation.row(2).dot(Vec3(wx, 0.0, wz)) +
                        translation.z();
                    if (depth > 1e-9) {
                        shade = static_cast<double>(
                            sequence_ground_shade(wx, wz));
                        for (const MarkerSequenceTarget& target : targets) {
                            if (const auto value =
                                    sequence_target_shade(target, wx, wz)) {
                                shade = static_cast<double>(*value);
                                break;
                            }
                        }
                    }
                    sum += shade;
                }
            }
            image.at(x, y) = static_cast<float>(sum / 4.0);
        }
    }
    if (spec.occlude_first >= 0 && temporal >= spec.occlude_first &&
        temporal <= spec.occlude_last &&
        spec.occluded_target >= 0 &&
        spec.occluded_target < static_cast<int>(targets.size())) {
        const MarkerSequenceTarget& target =
            targets[static_cast<std::size_t>(spec.occluded_target)];
        const Vec3 centre_camera = rotation * target.centre + translation;
        const Vec3 edge_camera =
            rotation * (target.centre + Vec3(target.radius, 0.0, 0.0)) +
            translation;
        if (centre_camera.z() > 0.0 && edge_camera.z() > 0.0) {
            const Vec2 centre = calibration.project(centre_camera);
            const double radius =
                (calibration.project(edge_camera) - centre).norm() + 4.0;
            const int x0 = std::max(
                0, static_cast<int>(std::floor(centre.x() - radius)));
            const int x1 = std::min(
                calibration.width - 1,
                static_cast<int>(std::ceil(centre.x() + radius)));
            const int y0 = std::max(
                0, static_cast<int>(std::floor(centre.y() - radius)));
            const int y1 = std::min(
                calibration.height - 1,
                static_cast<int>(std::ceil(centre.y() + radius)));
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    image.at(x, y) = 0.45F;
                }
            }
        }
    }
    image = gaussian_blur(image, 0.8F);
    add_noise(image, 0.008, spec.seed + static_cast<std::uint32_t>(temporal));
    return image;
}

inline MarkerSequence::~MarkerSequence() {
    for (const std::string& path : paths) {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
}

inline const Camera& MarkerSequence::camera(std::size_t temporal) const {
    const MarkerSequenceFrame& frame = frames.at(temporal);
    return chunk->cameras().at(static_cast<std::size_t>(frame.import_index));
}

inline const Calibration& MarkerSequence::calibration(
    std::size_t temporal) const {
    return frames.at(temporal).calibration;
}

inline std::optional<Vec2> MarkerSequence::truth(std::size_t target,
                                                 std::size_t temporal) const {
    return project_marker(targets.at(target).centre, camera(temporal),
                          calibration(temporal), false);
}

inline MarkerSequence make_marker_sequence(const MarkerSequenceSpec& spec) {
    MarkerSequence scene;
    Chunk& chunk = scene.document.workspace().add_chunk("Chunk 1");
    scene.document.set_active_chunk_id(chunk.id());
    scene.chunk = &chunk;
    const std::vector<std::uint32_t>& codes = circular_codes(12);
    const int code_indices[3] = {1, 5, 11};
    const Vec3 positions[4] = {Vec3(-0.18, 0.0, 0.12), Vec3(0.16, 0.0, 0.09),
                               Vec3(-0.02, 0.0, -0.17),
                               Vec3(0.18, 0.0, -0.14)};
    for (int i = 0; i < 4; ++i) {
        MarkerSequenceTarget target;
        target.cross = i == 3;
        target.centre = positions[i];
        target.radius = 0.060;
        target.rotation = 0.21 * static_cast<double>(i);
        if (!target.cross) {
            const int index =
                std::clamp(code_indices[i], 0,
                           static_cast<int>(codes.size()) - 1);
            target.bits = 12;
            target.word = codes[static_cast<std::size_t>(index)];
            target.code = 2 * index + 1;
        }
        scene.targets.push_back(target);
    }
    Sensor sensor;
    sensor.label = "sensor";
    sensor.type = CameraType::Frame;
    sensor.width = spec.width;
    sensor.height = spec.height;
    sensor.calibration =
        Calibration::initial(CameraType::Frame, spec.width, spec.height,
                             spec.focal);
    chunk.sensors().push_back(sensor);
    if (spec.import_order.empty()) {
        scene.import_order.resize(static_cast<std::size_t>(spec.frames));
        for (int i = 0; i < spec.frames; ++i) {
            scene.import_order[static_cast<std::size_t>(i)] = i;
        }
    } else {
        scene.import_order = spec.import_order;
    }
    scene.frames.resize(static_cast<std::size_t>(spec.frames));
    scene.paths.reserve(scene.import_order.size());
    for (std::size_t import_pos = 0; import_pos < scene.import_order.size();
         ++import_pos) {
        const int temporal = scene.import_order[import_pos];
        if (temporal < 0 || temporal >= spec.frames) {
            continue;
        }
        const Transform pose = sequence_pose(temporal, spec.frames, 1.15);
        const Calibration calibration = chunk.sensors()[0].calibration;
        Camera camera;
        camera.id = chunk.allocate_id();
        camera.sensor_id = 0;
        camera.label = "cam" + std::to_string(temporal);
        camera.enabled = true;
        camera.pose = pose;
        chunk.cameras().push_back(camera);
        const std::string path =
            platform::unique_temp_path("alphashape_track_", ".png");
        auto frame = std::make_shared<Frame>(path, spec.width, spec.height);
        frame->set_id(chunk.allocate_id());
        frame->set_camera_id(camera.id);
        chunk.frames().push_back(frame);
        const FloatImage image =
            render_sequence_view(pose, calibration, scene.targets, temporal,
                                 spec);
        std::optional<std::string> when;
        std::optional<std::string> subsec;
        const int time_index =
            spec.reverse_capture_times ? spec.frames - 1 - temporal : temporal;
        if (spec.burst_capture_times) {
            when = "2026:05:01 10:00:00";
            char fraction[8];
            std::snprintf(fraction, sizeof(fraction), "%02d", 1 + time_index);
            subsec = std::string(fraction);
        } else if (temporal != spec.missing_time_temporal) {
            char stamp[32];
            const int hours = 10 + time_index / 3600;
            const int minutes = (time_index / 60) % 60;
            const int seconds = time_index % 60;
            std::snprintf(stamp, sizeof(stamp), "2026:05:01 %02d:%02d:%02d",
                          hours, minutes, seconds);
            when = std::string(stamp);
        }
        if (temporal == spec.malformed_time_temporal) {
            when = "not a valid stamp";
            subsec.reset();
        }
        write_gray_png_with_time(path, image, when, subsec);
        MarkerSequenceFrame record;
        record.temporal_index = temporal;
        record.camera_id = camera.id;
        record.import_index = static_cast<int>(chunk.cameras().size()) - 1;
        record.path = path;
        record.pose = pose;
        record.calibration = calibration;
        scene.frames[static_cast<std::size_t>(temporal)] = record;
        scene.paths.push_back(path);
    }
    return scene;
}

}  // namespace alphashape::test
